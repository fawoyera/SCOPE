"""Batched BERTScore (F1) with a domain-adapted and a general encoder.

BERTSc_D uses an ATC-adapted BERT (``scope bert-mlm`` builds it once);
BERTSc_G uses ``bert-base-uncased``.  Scores are computed in one batched call
per encoder instead of one call per example.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import BERTScoreConfig, resolve_path
from ..utils.common import log


def bertscore_f1(cands: List[str], refs: List[str], model: str, num_layers: int,
                 device: str = "cpu", batch_size: int = 64) -> Optional[List[float]]:
    if not cands:
        return []
    try:
        from bert_score import score
    except ImportError:
        log("bert-score not installed — skipping BERTScore")
        return None
    _ensure_max_length(resolve_path(model))
    try:
        # empty candidates make bert_score warn/NaN; score them against a placeholder
        cands = [c if c.strip() else "." for c in cands]
        _, _, f1 = score(cands, refs, model_type=resolve_path(model), num_layers=num_layers,
                         lang="en", device=device, batch_size=batch_size, verbose=False)
        return [float(x) for x in f1.tolist()]
    except Exception as e:
        log(f"BERTScore with {model} failed: {e}")
        return None


def _ensure_max_length(path: str, default: int = 512) -> None:
    """bert-score truncates to tokenizer.model_max_length; a local tokenizer saved without it
    reports a sentinel ~1e30 and crashes ("int too big to convert").  Pin it to 512."""
    import json
    from pathlib import Path
    cfg = Path(path) / "tokenizer_config.json"
    if not cfg.exists():
        return
    d = json.load(open(cfg))
    if not isinstance(d.get("model_max_length"), int) or d["model_max_length"] > 100_000:
        d["model_max_length"] = default
        json.dump(d, open(cfg, "w"), indent=2)


def bertscore_both(cands: List[str], refs: List[str], cfg: BERTScoreConfig):
    if not cfg.enabled:
        return None, None
    d = bertscore_f1(cands, refs, cfg.domain_model, cfg.num_layers, cfg.device, cfg.batch_size)
    g = bertscore_f1(cands, refs, cfg.general_model, cfg.num_layers, cfg.device, cfg.batch_size)
    return d, g
