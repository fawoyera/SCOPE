"""Per-request latency benchmark (batch size 1), for the latency/compute section.

For a finished run it reloads exactly what was tested (``checkpoint-best`` for trained
methods, the base model otherwise) with the run's decoding (unconstrained, GCD†, or
grammar-constrained GCD), then times the same first-N test utterances one at a time
after a warm-up.  Reported per utterance: end-to-end latency, generated tokens, time to
first token (prefill + 1 step, measured separately), and the share spent computing
grammar masks.  Results go to ``<run_dir>/latency.json``.

Train-time compute (GPU-hours, FLOPs, phase breakdown) is recorded automatically during
training in ``compute.json``; this module only covers inference.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..config import load_saved_config
from ..data import load_splits, render
from ..domain import Artefacts
from ..models import load_model, load_tokenizer
from ..utils.common import log, write_json
from ..utils.profiling import _sync, hardware, n_params, peak_memory_gb, reset_peak_memory
from .evaluator import generate


def _pct(x: List[float]) -> Dict[str, float]:
    a = np.asarray(x, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def benchmark_run(run_dir: Path, n: int = 200, warmup: int = 10, ttft: bool = True) -> Dict:
    cfg = load_saved_config(run_dir / "config.yaml")
    cfg.eval.batch_size = 1
    ckpt = run_dir / "checkpoint-best"
    src = str(ckpt) if (cfg.train.enabled and ckpt.exists()) else (cfg.run.init_from or None)
    if cfg.train.enabled and not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing (was train.keep_checkpoint=false?)")
    tok = load_tokenizer(cfg.model, src)
    model = load_model(cfg.model, src, trainable=False)
    art = Artefacts.load(cfg.artefacts)
    test = load_splits(cfg.data).test[: n + warmup]
    rendered = [render(p, tok, cfg) for p in test]
    prompts, roles = [r["prompt"] for r in rendered], [r["role"] for r in rendered]
    use_gcd = cfg.gcd.enabled
    log(f"Latency: {run_dir} | decoding={'gcd-' + cfg.gcd.level if use_gcd else 'unconstrained'} | "
        f"n={len(prompts) - warmup} (+{warmup} warm-up)")
    generate(model, tok, prompts[:warmup], cfg, art, use_gcd, roles=roles[:warmup])   # warm-up (also builds GCD tables)
    reset_peak_memory()

    lat, toks, mask, first = [], [], [], []
    max_new = cfg.eval.max_new_tokens
    for p, r in zip(prompts[warmup:], roles[warmup:]):
        t: Dict = {}
        _sync()
        t0 = time.perf_counter()
        generate(model, tok, [p], cfg, art, use_gcd, roles=[r], timing=t)
        _sync()
        lat.append(1000 * (time.perf_counter() - t0))
        toks.append(t["per_example_tokens"][0])
        mask.append(1000 * t.get("mask_seconds", 0.0))
        if ttft:
            cfg.eval.max_new_tokens = 1
            _sync()
            t1 = time.perf_counter()
            generate(model, tok, [p], cfg, art, use_gcd, roles=[r])
            _sync()
            first.append(1000 * (time.perf_counter() - t1))
            cfg.eval.max_new_tokens = max_new
    per_tok = [l / max(1, k) for l, k in zip(lat, toks)]
    out = {"run_dir": str(run_dir), "model": cfg.run.model, "method": cfg.run.method, "seed": cfg.run.seed,
           "decoding": f"gcd-{cfg.gcd.level}" if use_gcd else "unconstrained",
           "num_beams": cfg.gcd.num_beams if use_gcd else 1, "n": len(lat), "params": n_params(model),
           "hardware": hardware(), "latency_ms": _pct(lat), "ms_per_token": _pct(per_tok),
           "tokens": _pct(toks), "mask_ms": _pct(mask),
           "mask_fraction": float(np.sum(mask) / max(1e-9, np.sum(lat))),
           **({"ttft_ms": _pct(first)} if first else {}), **peak_memory_gb(),
           "per_example": [{"latency_ms": a, "tokens": b, "mask_ms": c} for a, b, c in zip(lat, toks, mask)]}
    write_json(run_dir / "latency.json", out)
    log(f"  latency p50={out['latency_ms']['p50']:.1f} ms p95={out['latency_ms']['p95']:.1f} ms | "
        f"{out['ms_per_token']['mean']:.2f} ms/token | mask share {out['mask_fraction']:.1%}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
