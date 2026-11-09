"""Small shared utilities: seeding, timing, console tee, JSON(L) I/O, depot mirroring."""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def fmt_elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class Tee:
    """Mirror stdout and stderr to a log file for the lifetime of the object."""

    def __init__(self, path: Path):
        self._f = open(path, "a", buffering=1)
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(self._out, self._f)
        sys.stderr = _TeeStream(self._err, self._f)

    def close(self) -> None:
        sys.stdout, sys.stderr = self._out, self._err
        self._f.close()


class _TeeStream:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, msg):
        self.a.write(msg)
        self.b.write(msg)

    def flush(self):
        self.a.flush()
        self.b.flush()

    def isatty(self):
        return False


def read_jsonl(path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_json(path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def mirror_to_depot(run_dir: Path, depot_root: str, rel: str) -> None:
    """Copy small result files and the best checkpoint to persistent storage."""
    if not depot_root:
        return
    try:
        dest = Path(depot_root) / rel
        dest.mkdir(parents=True, exist_ok=True)
        for pat in ("*.log", "*.csv", "*.json", "*.jsonl", "*.yaml"):
            for f in run_dir.glob(pat):
                shutil.copy2(f, dest / f.name)
        best = run_dir / "checkpoint-best"
        if best.exists():
            shutil.rmtree(dest / best.name, ignore_errors=True)
            shutil.copytree(best, dest / best.name)
        log(f"Mirrored run to depot: {dest}")
    except Exception as e:  # never kill a run over a mirror failure
        log(f"Depot mirror failed: {e}")
