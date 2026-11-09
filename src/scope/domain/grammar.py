"""Role-gated CFG compliance (C_cfg) with crash- and hang-safe parsing.

The Earley parser can take pathologically long on some inputs, so every parse
runs in a pool of *spawned* worker processes (safe alongside CUDA/torch
threads).  Each worker compiles the grammar once in its initializer.  A task
that exceeds ``timeout`` scores 0 and the pool is recycled afterwards.

This module deliberately imports nothing heavy (no torch) so that spawned
workers start fast.
"""
from __future__ import annotations

import multiprocessing as mp
import re
from typing import Dict, List, Optional, Sequence

ROLE_START = {"controller": "controller_utterance", "pilot": "pilot_utterance"}
CATCHALL_RULES = ("general_instr", "pilot_general", "word")


def strict_grammar_source(src: str) -> str:
    """Drop the catch-all productions (§5.5, §6) so only specification rules remain."""
    out, skip = [], False
    for line in src.splitlines():
        is_alt = re.match(r"^\s*\|\s*(general_instr|pilot_general)\b", line)
        is_def = re.match(r"^(general_instr|pilot_general|word)\s*:", line)
        if is_alt:
            continue
        if is_def:
            skip = True
            continue
        if skip:
            cont = re.match(r"^\s*\|", line) or (
                line.strip() and not re.match(r"^\s*\w+\s*:", line)
                and not line.strip().startswith(("//", "%")))
            if cont:
                continue
            skip = False
        out.append(line)
    return "\n".join(out)


def start_symbol(role: str) -> str:
    return ROLE_START.get(role, "start")


# ── worker side ────────────────────────────────────────────────────────────
_PARSERS: Dict[str, object] = {}


def _init_worker(grammar_src: str) -> None:
    from lark import Lark
    for role in ("controller", "pilot", "unified"):
        _PARSERS[role] = Lark(grammar_src, parser="earley",
                              start=start_symbol(role), ambiguity="resolve")


def _parses(parser, text: str) -> bool:
    try:
        parser.parse(text)
        return True
    except Exception:
        return False


def _worker_score(args) -> float:
    """C_cfg (Def. 3): 1 if the utterance parses, else max k/T over parsable prefixes."""
    text, role, partial = args
    parser = _PARSERS.get(role) or _PARSERS["unified"]
    words = " ".join(text.upper().split()).split()
    if not words:
        return 0.0
    if _parses(parser, " ".join(words)):
        return 1.0
    if not partial:
        return 0.0
    for k in range(len(words) - 1, 0, -1):
        if _parses(parser, " ".join(words[:k])):
            return k / len(words)
    return 0.0


def _norm_role(role: Optional[str]) -> str:
    return role if role in ROLE_START else "unified"


# ── main-process API ──────────────────────────────────────────────────────
class GrammarScorer:
    """Parallel, timeout-protected C_cfg scorer."""

    def __init__(self, grammar_path: str, workers: int = 4, timeout: float = 3.0):
        with open(grammar_path) as f:
            self.src = strict_grammar_source(f.read())
        self.workers = max(1, workers)
        self.timeout = timeout
        self._pool = None
        # compile once in-process to fail fast on grammar errors
        from lark import Lark
        Lark(self.src, parser="earley", start="start", ambiguity="resolve")

    def _get_pool(self):
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=(self.src,))
        return self._pool

    def score(self, texts: Sequence[str], roles: Sequence[str], partial: bool = True) -> List[float]:
        if not texts:
            return []
        pool = self._get_pool()
        jobs = [pool.apply_async(_worker_score, ((t or "", _norm_role(r), partial),))
                for t, r in zip(texts, roles)]
        out, hung = [], False
        for j in jobs:
            try:
                out.append(float(j.get(timeout=self.timeout)))
            except mp.TimeoutError:
                out.append(0.0)
                hung = True
            except Exception:
                out.append(0.0)
        if hung:  # a worker may still be stuck inside Earley: recycle the pool
            self.close()
        return out

    def parses(self, texts: Sequence[str], roles: Sequence[str]) -> List[bool]:
        return [s >= 1.0 for s in self.score(texts, roles, partial=False)]

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
