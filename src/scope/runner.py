"""Experiment grid: models × methods × seeds, with dependencies and two launchers.

* ``dpo`` and ``gcd_sft`` start from the SFT checkpoint of the same model and seed;
  the SFT job is added automatically if missing.
* ``vanilla`` and ``gcd`` use greedy decoding of the un-trained model and are fully
  deterministic, so by default they run once per model (first seed).
* Completed runs (``test_metrics.json`` present) are skipped, so re-launching the same
  grid resumes it.

Launchers:  local  — sequential subprocesses on this node (one GPU job at a time)
            slurm  — one sbatch job per run, chained with --dependency=afterok
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import CONFIG_ROOT, run_dir_for
from .pipeline import is_complete
from .utils.common import log

DEPENDS_ON = {"dpo": "sft", "gcd_sft": "sft", "gcd_cfg_sft": "sft"}
DETERMINISTIC = {"vanilla", "gcd", "gcd_cfg"}
ALL_METHODS = ["vanilla", "sft", "dpo", "gcd", "gcd_sft", "gcd_cfg", "gcd_cfg_sft", "scope"]


@dataclass
class Job:
    domain: str
    model: str
    method: str
    seed: int
    output_root: str
    overrides: List[str] = field(default_factory=list)
    dep: Optional["Job"] = None
    label: str = ""                      # run-directory name; defaults to the method (ablations: method__variant)

    @property
    def tag(self) -> str:
        return self.label or self.method

    @property
    def run_dir(self) -> Path:
        return run_dir_for(self.output_root, self.domain, self.model, self.tag, self.seed)

    @property
    def name(self) -> str:
        return f"{self.domain}-{self.model}-{self.tag}-s{self.seed}"

    def command(self, force: bool = False, python: str = sys.executable, discard: bool = False) -> List[str]:
        cmd = [python, "-m", "scope.cli", "experiment", "--domain", self.domain, "--model", self.model,
               "--method", self.method, "--seed", str(self.seed), "--set", f"run.output_root={self.output_root}"]
        if self.label and self.label != self.method:
            cmd += ["--set", f"run.method={self.label}"]
        if self.dep is not None:
            cmd += ["--set", f"run.init_from={self.dep.run_dir / 'checkpoint-best'}"]
        for o in self.overrides:
            cmd += ["--set", o]
        if force:
            cmd.append("--force")
            if discard:
                cmd.append("--discard")
        return cmd


def plan(domain: str, models: List[str], methods: List[str], seeds: List[int], output_root: str,
         overrides: Optional[List[str]] = None, all_seeds_for_deterministic: bool = False) -> List[Job]:
    jobs: Dict[tuple, Job] = {}

    def add(model, method, seed) -> Job:
        key = (model, method, seed)
        if key in jobs:
            return jobs[key]
        dep = add(model, DEPENDS_ON[method], seed) if method in DEPENDS_ON else None
        jobs[key] = Job(domain, model, method, seed, output_root, list(overrides or []), dep)
        return jobs[key]

    for model in models:
        for method in methods:
            run_seeds = seeds[:1] if (method in DETERMINISTIC and not all_seeds_for_deterministic) else seeds
            for seed in run_seeds:
                add(model, method, seed)
    return list(jobs.values())   # insertion order is a valid topological order


def load_ablation_spec(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_ROOT / "ablations" / f"{name_or_path}.yaml"
    spec = yaml.safe_load(open(p))
    spec.setdefault("base_method", "scope")
    return spec


def plan_ablation(spec: dict, domain: str, models: List[str], seeds: List[int], output_root: str,
                  overrides: Optional[List[str]] = None, only: Optional[List[str]] = None) -> List[Job]:
    """Each variant = base method (or its own ``method``) + ``set`` overrides, stored under
    ``<method>__<variant>``.  ``reuse: <method>`` points a variant at an existing main-grid
    method (no new run), ``init_from: <variant>`` starts it from another variant's checkpoint."""
    variants: Dict[str, dict] = spec["variants"]
    if only:
        keep = set(only)
        for v in list(keep):                       # keep variants this one depends on
            while variants.get(v, {}).get("init_from") in variants:
                v = variants[v]["init_from"]
                keep.add(v)
        variants = {k: v for k, v in variants.items() if k in keep}
    jobs: Dict[tuple, Job] = {}

    def add_plain(model, method, seed) -> Job:
        key = (model, method, seed)
        if key not in jobs:
            dep = add_plain(model, DEPENDS_ON[method], seed) if method in DEPENDS_ON else None
            jobs[key] = Job(domain, model, method, seed, output_root, list(overrides or []), dep)
        return jobs[key]

    def add_variant(model, vname, seed) -> Job:
        v = variants[vname] or {}
        if v.get("reuse"):
            return add_plain(model, v["reuse"], seed)
        key = (model, f"variant:{vname}", seed)
        if key in jobs:
            return jobs[key]
        method = v.get("method", spec["base_method"])
        dep = None
        if v.get("init_from"):
            dep = add_variant(model, v["init_from"], seed) if v["init_from"] in variants \
                else add_plain(model, v["init_from"], seed)
        elif method in DEPENDS_ON:
            dep = add_plain(model, DEPENDS_ON[method], seed)
        label = f"{spec['base_method']}__{vname}"
        jobs[key] = Job(domain, model, method, seed, output_root,
                        list(overrides or []) + list(spec.get("set", [])) + list(v.get("set", [])), dep, label)
        return jobs[key]

    for model in models:
        for vname in variants:
            v = variants[vname] or {}
            one = v.get("deterministic") or v.get("reuse") in DETERMINISTIC
            for seed in (seeds[:1] if one else seeds):
                add_variant(model, vname, seed)
    return list(jobs.values())


def run_local(jobs: List[Job], force: bool = False, dry_run: bool = False, discard: bool = False) -> int:
    failed: set = set()
    for j in jobs:
        if j.dep is not None and j.dep.name in failed:
            log(f"✗ {j.name}: skipped (dependency {j.dep.name} failed)")
            failed.add(j.name)
            continue
        if is_complete(j.run_dir) and not force:
            log(f"✓ {j.name}: complete")
            continue
        cmd = j.command(force, discard=discard)
        log(f"▶ {j.name}\n   {shlex.join(cmd)}")
        if dry_run:
            continue
        if subprocess.run(cmd).returncode != 0:
            log(f"✗ {j.name} failed")
            failed.add(j.name)
    log(f"Finished: {len(jobs) - len(failed)}/{len(jobs)} ok" + (f"; failed: {sorted(failed)}" if failed else ""))
    return 1 if failed else 0


def run_slurm(jobs: List[Job], slurm_cfg: str, force: bool = False, dry_run: bool = False,
              discard: bool = False) -> int:
    spec = yaml.safe_load(open(Path(slurm_cfg) if Path(slurm_cfg).exists()
                               else CONFIG_ROOT / "slurm" / f"{slurm_cfg}.yaml"))
    ids: Dict[str, str] = {}
    script_dir = Path(jobs[0].output_root) / "_slurm" if jobs else Path("_slurm")
    script_dir.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        if is_complete(j.run_dir) and not force:
            log(f"✓ {j.name}: complete")
            continue
        sb = dict(spec.get("sbatch", {}))
        sb["time"] = spec.get("time_overrides", {}).get(j.method, sb.get("time"))
        sb["job-name"] = j.name
        sb["output"] = str((script_dir / f"{j.name}-%j.out").resolve())
        lines = ["#!/bin/bash"] + [f"#SBATCH --{k}={v}" for k, v in sb.items() if v not in (None, "")]
        lines += ["set -eo pipefail", f"cd {Path.cwd().resolve()}", spec.get("setup", ""),
                  shlex.join(j.command(force, python=spec.get("python", "python"), discard=discard))]
        path = script_dir / f"{j.name}.sbatch"
        path.write_text("\n".join(lines) + "\n")
        cmd = ["sbatch", "--parsable"]
        if j.dep is not None and j.dep.name in ids:
            cmd.append(f"--dependency=afterok:{ids[j.dep.name]}")
        cmd.append(str(path))
        if dry_run:
            log(f"[dry-run] {shlex.join(cmd)}")
            ids[j.name] = f"<{j.name}>"
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            log(f"✗ sbatch failed for {j.name}: {out.stderr.strip()}")
            return 1
        ids[j.name] = out.stdout.strip().split(";")[0]
        log(f"submitted {j.name} → job {ids[j.name]}" + (f" (after {j.dep.name})" if j.dep and j.dep.name in ids else ""))
    return 0
