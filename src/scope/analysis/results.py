"""Collect finished runs from ``results/<domain>/<model>/<method>/s<seed>/``."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import iter_run_dirs

# metric key, column header, higher-is-better, scale for display
METRIC_SPECS = [
    ("c_tok", r"\Ctok", True, 1.0),
    ("c_phr", r"\Cphr", True, 1.0),
    ("c_cfg", r"\Ccfg", True, 1.0),
    ("c_bar", r"\Cbar", True, 1.0),
    ("slot_f1", "Slot-F1", True, 1.0),
    ("da_f1", "DA-F1", True, 1.0),
    ("halluc", r"Hall.\%$\downarrow$", False, 100.0),
    ("bertscore_d", r"$_\mathrm{D}$", True, 1.0),
    ("bertscore_g", r"$_\mathrm{G}$", True, 1.0),
]
METRIC_KEYS = [m[0] for m in METRIC_SPECS]
HIGHER_BETTER = {m[0]: m[2] for m in METRIC_SPECS}
SCALE = {m[0]: m[3] for m in METRIC_SPECS}

METHOD_LABELS = {"vanilla": "Vanilla", "sft": "SFT", "dpo": "DPO", "gcd": r"GCD$^\dagger$",
                 "gcd_sft": r"GCD$^\dagger$+SFT",
                 "gcd_cfg": "GCD", "gcd_cfg_sft": "GCD+SFT", "scope": r"\textbf{SCOPE (ours)}"}
MODEL_LABELS = {"llama31_8b": "Llama-3.1-8B", "llama31_8b_instruct": "Llama-3.1-8B-Inst.",
                "qwen3_8b": "Qwen3-8B", "gpt2_large": "GPT-2 774M", "gemma2_9b_it": "Gemma-2-9B",
                "gemma3_1b_it": "Gemma-3-1B", "mistral_7b_instruct": "Mistral-7B"}


def tex_escape(s: str) -> str:
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m) or tex_escape(m)


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m) or tex_escape(m)


def _runs(root: Path, domain: str):
    yield from iter_run_dirs(root, domain, "test_metrics.json")


def collect_runs(root: str, domain: str) -> pd.DataFrame:
    """One row per finished run with its aggregate test metrics."""
    rows = []
    for model, method, seed, f in _runs(Path(root), domain):
        m = json.load(open(f))
        rows.append({"model": model, "method": method, "seed": seed,
                     **{k: m.get(k) for k in METRIC_KEYS + ["n", "c_phr_macro"]}})
    return pd.DataFrame(rows)


def collect_examples(root: str, domain: str, models: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per (run, test example) with per-example metrics — input to paired tests."""
    frames = []
    for model, method, seed, f in _runs(Path(root), domain):
        if models and model not in models:
            continue
        p = f.parent / "test_predictions.jsonl"
        if not p.exists():
            continue
        df = pd.read_json(p, lines=True)
        keep = ["pair_id"] + [k for k in METRIC_KEYS if k in df.columns]
        df = df[keep].copy()
        df["model"], df["method"], df["seed"] = model, method, seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def order(values, preferred: List[str]) -> List[str]:
    vals = list(dict.fromkeys(values))
    return [v for v in preferred if v in vals] + sorted(v for v in vals if v not in preferred)
