"""Wall-clock profiling, token accounting and FLOP estimates.

* ``Profiler.section(name)`` — CUDA-synchronised timer that accumulates seconds per phase.
* token counters per phase, used for FLOP estimates:
      training forward+backward  ≈ 6 · N · tokens      (Kaplan et al., 2020)
      forward only (generation,  ≈ 2 · N · tokens
      reference / reward passes)
  N = number of model parameters; attention FLOPs are ignored (sequences here are short,
  ≤ 1k tokens, so the dense term dominates).  Reported as estimates.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict

import torch

FWD_BWD = 6.0
FWD = 2.0


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def hardware() -> Dict:
    if not torch.cuda.is_available():
        return {"n_gpus": 0, "gpu": "cpu"}
    return {"n_gpus": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0)}


def reset_peak_memory():
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)


def peak_memory_gb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    per = [torch.cuda.max_memory_allocated(d) / 1e9 for d in range(torch.cuda.device_count())]
    return {"peak_mem_gb_total": round(sum(per), 2), "peak_mem_gb_max_device": round(max(per), 2),
            "peak_mem_gb_per_device": [round(x, 2) for x in per]}


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


class Profiler:
    def __init__(self):
        self.seconds: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)
        self.tokens: Dict[str, int] = defaultdict(int)     # by phase
        self.flop_coef: Dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        _sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.seconds[name] += time.perf_counter() - t
            self.calls[name] += 1

    def count(self, phase: str, n_tokens: int, coef: float) -> None:
        self.tokens[phase] += int(n_tokens)
        self.flop_coef[phase] = coef

    def flops(self, params: int) -> Dict[str, float]:
        per = {k: self.flop_coef[k] * params * v for k, v in self.tokens.items()}
        return {"flops_total": float(sum(per.values())), **{f"flops_{k}": float(v) for k, v in per.items()}}

    def summary(self, params: int) -> Dict:
        return {"seconds": dict(self.seconds), "calls": dict(self.calls), "tokens": dict(self.tokens),
                **self.flops(params)}
