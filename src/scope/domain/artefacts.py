"""Compliance artefacts (V, P, G) for one domain."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set, Tuple

from ..config import ArtefactConfig, resolve_path

NGRAM_KEYS = {2: "bigrams", 3: "trigrams", 4: "4grams"}


@dataclass
class Artefacts:
    vocab: FrozenSet[str]                       # V
    ngrams: Dict[int, Set[Tuple[str, ...]]]     # P = P^(2) ∪ P^(3) ∪ P^(4)
    grammar_path: str                           # G (Lark source)
    system_prompt: str = "atc"
    stats: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: ArtefactConfig) -> "Artefacts":
        with open(resolve_path(cfg.vocab)) as f:
            vocab = frozenset(w.upper() for w in json.load(f))
        with open(resolve_path(cfg.phrases)) as f:
            raw = json.load(f)
        ngrams = {n: {tuple(w.upper() for w in g) for g in raw.get(k, [])}
                  for n, k in NGRAM_KEYS.items()}
        stats = {"|V|": len(vocab), **{f"|P{n}|": len(s) for n, s in ngrams.items()}}
        return cls(vocab, ngrams, resolve_path(cfg.grammar), cfg.system_prompt, stats)
