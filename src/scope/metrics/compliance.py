"""Lexical and phraseological compliance (Defs. 1–2).  C_cfg lives in domain/grammar.py."""
from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple


def words(text: str) -> List[str]:
    return text.upper().split()


def c_tok(tokens: List[str], vocab: Iterable[str]) -> float:
    """C_tok = (1/T) Σ_t 1[ŷ_t ∈ V]."""
    if not tokens:
        return 0.0
    return sum(t in vocab for t in tokens) / len(tokens)


def c_phr(tokens: List[str], ngrams: Dict[int, Set[Tuple[str, ...]]]) -> float:
    """Phrase compliance: fraction of all order-{2,3,4} windows found in P.

    Windows of all orders are pooled (micro-average), matching the released
    training code.  ``c_phr_macro`` gives the per-order uniform average of Eq. (cphr).
    """
    if not tokens:
        return 0.0
    total = hits = 0
    for n, wl in ngrams.items():
        for i in range(len(tokens) - n + 1):
            total += 1
            hits += tuple(tokens[i:i + n]) in wl
    return hits / total if total else 0.0


def c_phr_macro(tokens: List[str], ngrams: Dict[int, Set[Tuple[str, ...]]]) -> float:
    """Eq. (cphr): (1/3) Σ_n C_phr^(n); orders with no window contribute 0."""
    if not tokens:
        return 0.0
    vals = []
    for n, wl in ngrams.items():
        win = len(tokens) - n + 1
        vals.append(sum(tuple(tokens[i:i + n]) in wl for i in range(win)) / win if win > 0 else 0.0)
    return sum(vals) / len(vals)
