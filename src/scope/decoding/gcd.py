"""GCD†: vocabulary (+ bigram) constrained decoding as a Hugging Face LogitsProcessor.

At every step a token is allowed only if appending its surface string keeps the
response inside the constraint set:

* every *completed* word is in V                                   (level = vocab, bigram)
* every completed word w following word u satisfies (u, w) ∈ P^(2) (level = bigram)
* the trailing partial word is a prefix of some admissible next word
* EOS is allowed only at a word boundary (partial word empty or complete)

Strict Earley (CFG) masking degenerates on the LDC corpus (see paper), so the
token-level constraint uses V and P^(2); C_cfg is still measured afterwards.
Masks are cached per decoder state, which makes this fast in practice.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch

_WORD_PIECE = re.compile(r"^ ?[A-Z][A-Z'\-]*$|^ ?[A-Z'\-]+$")


def token_surfaces(tokenizer) -> List[str]:
    """Surface string of every token *as it would appear mid-text* (keeps leading spaces
    for both byte-level BPE and SentencePiece)."""
    anchor = tokenizer("A", add_special_tokens=False)["input_ids"]
    base = tokenizer.decode(anchor)
    out = []
    for i in range(len(tokenizer)):
        s = tokenizer.decode(anchor + [i])
        out.append(s[len(base):] if s.startswith(base) else tokenizer.decode([i]))
    return out


class ConstrainedVocabProcessor:
    def __init__(self, tokenizer, vocab: FrozenSet[str], bigrams: Set[Tuple[str, str]],
                 prompt_len: int, level: str = "bigram", successor_fallback: bool = True):
        self.vocab = frozenset(vocab)
        self.level = level
        self.fallback = successor_fallback
        self.prompt_len = prompt_len
        self.eos = set(t for t in [tokenizer.eos_token_id, *(getattr(tokenizer, "additional_eos_ids", []) or [])]
                       if t is not None)
        self.succ: Dict[str, Set[str]] = {}
        for a, b in bigrams:
            self.succ.setdefault(a, set()).add(b)
        surf = _surfaces_cached(tokenizer)
        # candidate tokens: optional single leading space + upper-case letters / ' / -
        self.cands = [(i, s) for i, s in enumerate(surf) if s and _WORD_PIECE.match(s)]
        self.space_tokens = [i for i, s in enumerate(surf) if s == " "]
        self.surf = surf
        self.states: List[Tuple[Optional[str], str]] = []
        self._mask_cache: Dict[Tuple, torch.Tensor] = {}
        self._prefix_cache: Dict[Optional[str], Tuple[FrozenSet[str], FrozenSet[str]]] = {}

    # ── constraint sets ────────────────────────────────────────────────────
    def _next_words(self, prev: Optional[str]) -> FrozenSet[str]:
        if self.level != "bigram" or prev is None:
            return self.vocab
        nxt = self.succ.get(prev, set()) & self.vocab
        if not nxt:
            return self.vocab if self.fallback else frozenset()
        return frozenset(nxt)

    def _prefixes(self, prev: Optional[str]):
        if prev not in self._prefix_cache:
            words = self._next_words(prev)
            pref = frozenset(w[:k] for w in words for k in range(1, len(w) + 1))
            self._prefix_cache[prev] = (words, pref)
        return self._prefix_cache[prev]

    @staticmethod
    def advance(state, piece: str):
        prev, partial = state
        if piece.startswith(" "):
            if partial:
                prev = partial
            return prev, piece[1:]
        return prev, partial + piece

    def _ok(self, state, piece: str) -> bool:
        prev, partial = state
        if piece.startswith(" "):
            if partial:
                words, _ = self._prefixes(prev)
                if partial not in words:
                    return False
                prev = partial
            partial = piece[1:]
        else:
            partial = partial + piece
        if not partial:
            return True
        _, pref = self._prefixes(prev)
        return partial in pref

    def _mask(self, state, size: int, device) -> torch.Tensor:
        key = (state, size, device)
        if key not in self._mask_cache:
            allowed = [i for i, s in self.cands if i < size and self._ok(state, s)]
            prev, partial = state
            words, _ = self._prefixes(prev)
            if not partial or partial in words:
                allowed += [e for e in self.eos if e < size]
                allowed += [i for i in self.space_tokens if i < size and partial]
            if not allowed:                      # dead end: let the sequence terminate
                allowed = [e for e in self.eos if e < size]
            m = torch.ones(size, dtype=torch.bool)
            m[torch.tensor(allowed, dtype=torch.long)] = False   # True = blocked
            self._mask_cache[key] = m.to(device)
        return self._mask_cache[key]

    # ── HF interface ───────────────────────────────────────────────────────
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.size(0)
        if not self.states or len(self.states) != B or input_ids.size(1) == self.prompt_len:
            self.states = [(None, "")] * B
            self.done = [False] * B
        if input_ids.size(1) > self.prompt_len:
            for b in range(B):
                t = int(input_ids[b, -1])
                if t in self.eos:
                    self.done[b] = True
                if not self.done[b]:
                    self.states[b] = self.advance(self.states[b], self.surf[t] if t < len(self.surf) else "")
        for b in range(B):
            if self.done[b]:
                continue
            scores[b] = scores[b].masked_fill(self._mask(self.states[b], scores.size(-1), scores.device), float("-inf"))
        return scores


_SURF_CACHE: Dict[int, List[str]] = {}


def _surfaces_cached(tokenizer) -> List[str]:
    key = id(tokenizer)
    if key not in _SURF_CACHE:
        _SURF_CACHE[key] = token_surfaces(tokenizer)
    return _SURF_CACHE[key]


def make_gcd_processor(tokenizer, artefacts, prompt_len: int, level: str, fallback: bool):
    bigrams = {tuple(g) for g in artefacts.ngrams.get(2, set())}
    return ConstrainedVocabProcessor(tokenizer, artefacts.vocab, bigrams, prompt_len, level, fallback)
