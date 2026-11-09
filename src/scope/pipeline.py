"""One experiment = one (domain, model, method, seed).

    [load]  →  [DPO: build preference pairs from the SFT model]
            →  [train, select best epoch on validation]   (trained methods)
            →  [test: generate (± GCD) and score every example]

Outputs in ``results/<domain>/<model>/<method>/s<seed>/``::

    config.yaml               fully resolved config
    run.log                   console log
    steps.csv / epochs.csv    training curves (also in TensorBoard and W&B)
    checkpoint-best/          selected weights (trained methods)
    test_predictions.jsonl    one row per test example with all per-example metrics
    test_metrics.json         aggregate test metrics  ← read by `scope table` / `scope stats`
"""
from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import torch

from .config import ExperimentConfig, resolve_run_dir
from .data import load_splits
from .domain import Artefacts, GrammarScorer
from .evaluation import evaluate
from .models import load_model, load_tokenizer
from .tracking import Tracker
from .training import DPOTrainer, ObjectiveTrainer, build_preference_pairs
from .utils.common import Tee, fmt_elapsed, log, mirror_to_depot, set_seed, write_json, write_jsonl
from .utils.profiling import Profiler, hardware, n_params, peak_memory_gb, reset_peak_memory


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "test_metrics.json").exists()


def _free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def archive_run_dir(run_dir: Path) -> Path:
    """Move a previous attempt aside as ``s<seed>.<YYYYmmdd-HHMMSS>`` (never overwritten)."""
    from datetime import datetime
    stamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    dest = run_dir.with_name(f"{run_dir.name}.{stamp}")
    i = 1
    while dest.exists():
        dest = run_dir.with_name(f"{run_dir.name}.{stamp}-{i}")
        i += 1
    run_dir.rename(dest)
    return dest


def run_experiment(cfg: ExperimentConfig, force: bool = False, discard: bool = False) -> dict:
    """force: re-run, moving the previous attempt to ``s<seed>.<timestamp>``;
    discard (with force): delete the previous attempt instead."""
    run_dir = resolve_run_dir(cfg)
    if is_complete(run_dir) and not force:
        log(f"✓ {run_dir} already complete — skipping (use --force to redo)")
        import json
        return json.load(open(run_dir / "test_metrics.json"))
    archived = None
    if force and run_dir.exists():                   # explicit re-run: start clean
        if discard:
            shutil.rmtree(run_dir)
        else:
            archived = archive_run_dir(run_dir)
    elif run_dir.exists() and not cfg.run.resume_from:
        steps = sorted(run_dir.glob("step_*"))
        if steps:                                    # crashed/pre-empted job: resume automatically
            cfg.run.resume_from = str(steps[-1])
        else:                                        # restart from scratch: drop stale curves/logs
            for f in ("steps.csv", "epochs.csv"):
                (run_dir / f).unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(run_dir / "run.log")
    t0 = time.time()
    r = cfg.run
    try:
        if archived is not None:
            log(f"Previous attempt archived to {archived}")
        cfg.save(run_dir / "config.yaml")
        set_seed(r.seed)
        log(f"═══ {r.domain} | {r.model} ({cfg.model.hf_id}) | {r.method} | seed {r.seed} ═══")
        tracker = Tracker(cfg, run_dir, job_type=r.method)

        art = Artefacts.load(cfg.artefacts)
        log(f"Artefacts: {art.stats}")
        splits = load_splits(cfg.data)
        log(f"Data: {splits.describe()}")
        grammar = GrammarScorer(art.grammar_path, cfg.eval.grammar_workers, cfg.eval.grammar_timeout)

        init = r.init_from or None
        if init and not Path(init).exists():
            raise FileNotFoundError(f"init_from checkpoint not found: {init} "
                                    f"(run the SFT job for this model/seed first)")
        tok = load_tokenizer(cfg.model, init)
        model = load_model(cfg.model, init, trainable=cfg.train.enabled)

        best_epoch = None
        compute = {"hardware": hardware(), "params": n_params(model), "method": r.method, "model": r.model,
                   "seed": r.seed}
        if cfg.train.enabled:
            reset_peak_memory()
            t_train = time.time()
            prof = Profiler()
            if cfg.train.trainer == "dpo":
                pref = build_preference_pairs(model, tok, cfg, splits.train, grammar, run_dir / "dpo_pairs.jsonl",
                                              prof=prof)
                if not pref:
                    raise RuntimeError("No DPO preference pairs could be built")
                trainer = DPOTrainer(cfg, model, tok, splits, art, grammar, tracker, pref_rows=pref)
            else:
                trainer = ObjectiveTrainer(cfg, model, tok, splits, art, grammar, tracker)
            trainer.prof = prof
            best = trainer.train()
            best_epoch = trainer.best_epoch
            wall = time.time() - t_train
            n_dev = max(1, compute["hardware"]["n_gpus"])
            compute["training"] = {
                "wall_seconds": wall, "gpu_hours": wall * n_dev / 3600 if compute["hardware"]["n_gpus"] else 0.0,
                "train_loop_seconds": trainer.train_seconds, "optimizer_steps": trainer.opt_step,
                "samples_seen": trainer.samples_seen,
                "samples_per_second": trainer.samples_seen / max(1e-9, trainer.train_seconds),
                **peak_memory_gb(), **prof.summary(compute["params"])}
            del trainer
            _free(model)
            log(f"Reloading best checkpoint for testing: {best}")
            model = load_model(cfg.model, str(best), trainable=False)

        log(f"Test evaluation (GCD={'on, ' + cfg.gcd.level if cfg.gcd.enabled else 'off'})")
        reset_peak_memory()
        rows, summary = evaluate(model, tok, splits.test, cfg, art, grammar, use_gcd=cfg.gcd.enabled,
                                 with_bertscore=True, max_pairs=cfg.eval.test_max_pairs)
        write_jsonl(run_dir / "test_predictions.jsonl", rows)
        summary.update({"domain": r.domain, "model": r.model, "hf_id": cfg.model.hf_id, "method": r.method,
                        "seed": r.seed, "best_epoch": best_epoch, "elapsed_sec": round(time.time() - t0)})
        compute["inference"] = {**summary.pop("inference"), **peak_memory_gb()}
        inf = compute["inference"]   # prefill + decode, × beams; dense-layer estimate
        inf["flops_estimate"] = 2.0 * compute["params"] * max(1, inf["num_beams"]) * (
            inf["prompt_tokens"] + inf["generated_tokens"])
        write_json(run_dir / "compute.json", compute)
        write_json(run_dir / "test_metrics.json", summary)
        tracker.summary({f"compute/{k}": v for sec in ("training", "inference") for k, v in compute.get(sec, {}).items()
                         if isinstance(v, (int, float))})
        tracker.summary({f"test/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        tracker.log_examples("test/examples", [{k: x.get(k) for k in
                                                ("request", "reference", "generated", "c_bar", "slot_f1")}
                                               for x in rows[: cfg.logging.log_examples]])
        log("TEST | " + " ".join(f"{k}={summary[k]:.4f}" for k in
                                 ("c_tok", "c_phr", "c_cfg", "c_bar", "slot_f1", "da_f1", "halluc_pct",
                                  "bertscore_d", "bertscore_g") if k in summary))
        grammar.close()
        tracker.close()
        mirror_to_depot(run_dir, r.depot_path, f"{r.domain}/{r.model}/{r.method}/s{r.seed}")
        if cfg.train.enabled and not cfg.train.keep_checkpoint:
            shutil.rmtree(run_dir / "checkpoint-best", ignore_errors=True)
        log(f"Done in {fmt_elapsed(t0)} → {run_dir}")
        return summary
    finally:
        tee.close()
