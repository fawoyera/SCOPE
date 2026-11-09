"""Add BERTScore to finished runs without regenerating (``scope bertscore``).

Reads each run's ``test_predictions.jsonl``, computes BERTSc_D (domain encoder) and
BERTSc_G (general encoder) in one batched call per encoder, writes the per-example
scores back, and updates ``test_metrics.json`` so ``scope table`` / ``scope stats`` pick
them up.  Other metrics are left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..config import iter_run_dirs, load_saved_config
from ..metrics.bertscore import bertscore_f1
from ..utils.common import log, read_jsonl, write_json, write_jsonl
from .evaluator import METRICS


def rescore_run(run_dir: Path, domain_model: Optional[str], general_model: Optional[str],
                device: Optional[str], force: bool = False) -> bool:
    preds, mpath = run_dir / "test_predictions.jsonl", run_dir / "test_metrics.json"
    if not (preds.exists() and mpath.exists()):
        return False
    metrics = json.load(open(mpath))
    if not force and metrics.get("bertscore_d") is not None and metrics.get("bertscore_g") is not None:
        log(f"✓ {run_dir} already has BERTScore")
        return True
    bs = load_saved_config(run_dir / "config.yaml").eval.bertscore
    rows = read_jsonl(preds)
    cands, refs = [r["generated"] for r in rows], [r["reference"] for r in rows]
    ok = True
    for key, model in (("bertscore_d", domain_model or bs.domain_model),
                       ("bertscore_g", general_model or bs.general_model)):
        vals = bertscore_f1(cands, refs, model, bs.num_layers, device or bs.device, bs.batch_size)
        if vals is None:
            ok = False
            continue
        for r, v in zip(rows, vals):
            r[key] = v
        metrics[key] = sum(vals) / len(vals)
    write_jsonl(preds, rows)
    write_json(mpath, metrics)
    log(f"{run_dir}: BERTSc_D={metrics.get('bertscore_d', float('nan')):.4f} "
        f"BERTSc_G={metrics.get('bertscore_g', float('nan')):.4f}")
    return ok


def rescore_all(root: str, domain: str, models: Optional[List[str]] = None, methods: Optional[List[str]] = None,
                domain_model: Optional[str] = None, general_model: Optional[str] = None,
                device: Optional[str] = None, force: bool = False) -> int:
    dirs = [f.parent for *_, f in iter_run_dirs(root, domain, "test_metrics.json")]
    dirs = [d for d in dirs if (not models or d.parts[-3] in models) and (not methods or d.parts[-2] in methods)]
    failed = [d for d in dirs if not rescore_run(d, domain_model, general_model, device, force)]
    if failed:
        log(f"BERTScore missing for {len(failed)} run(s) — check that bert-score is installed and both "
            f"encoders load (run `scope bert-mlm` for the domain encoder; bert-base-uncased needs Hub access "
            f"or a local copy passed with --general-model).")
    return 1 if failed else 0
