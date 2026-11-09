"""One logging front-end for TensorBoard, Weights & Biases and CSV.

Trainers call ``tracker.log_step(...)`` / ``tracker.log_epoch(...)`` once; the
tracker fans out to every enabled backend.  Step metrics are keyed by
``global_step``; epoch/validation metrics by ``epoch`` (W&B ``define_metric``),
so W&B charts line up with the CSV files used by ``scope plot``.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig
from .utils.common import log


class _CSV:
    def __init__(self, path: Path):
        self.path = path
        self._f = None
        self._w = None
        self._fields: List[str] = []

    def write(self, row: Dict[str, Any]) -> None:
        new = [k for k in row if k not in self._fields]
        if self._w is None or new:
            # (re)open with the union of all columns seen so far; rewrite existing rows if needed
            old_rows = []
            if self.path.exists() and self.path.stat().st_size > 0:
                with open(self.path, newline="") as f:
                    old_rows = list(csv.DictReader(f))
                    self._fields = self._fields or (list(old_rows[0]) if old_rows else [])
            self._fields = self._fields + [k for k in row if k not in self._fields]
            if self._f:
                self._f.close()
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self._fields, extrasaction="ignore")
            self._w.writeheader()
            for r in old_rows:
                self._w.writerow(r)
        self._w.writerow({k: row.get(k, "") for k in self._fields})
        self._f.flush()

    def close(self):
        if self._f:
            self._f.close()


class Tracker:
    def __init__(self, cfg: ExperimentConfig, run_dir: Path, job_type: str):
        self.cfg = cfg
        self.run_dir = run_dir
        lc = cfg.logging
        self.tb = None
        self.wb = None
        self.steps_csv = _CSV(run_dir / "steps.csv") if lc.csv else None
        self.epochs_csv = _CSV(run_dir / "epochs.csv") if lc.csv else None

        if lc.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            except Exception as e:
                log(f"TensorBoard disabled ({e})")

        w = lc.wandb
        if w.enabled and w.mode != "disabled":
            try:
                import wandb
                r = cfg.run
                os.environ.setdefault("WANDB_DIR", str(run_dir))
                self.wb = wandb.init(
                    project=w.project,
                    entity=w.entity or None,
                    mode=w.mode,
                    name=f"{r.domain}-{r.model}-{r.method}-s{r.seed}",
                    group=f"{r.domain}/{r.model}/{r.method}",
                    job_type=job_type,
                    tags=[r.domain, r.model, r.method, f"seed{r.seed}", *w.tags],
                    config=cfg.to_dict(),
                    dir=str(run_dir),
                    resume="allow",
                    id=_run_id(run_dir),
                    reinit=True,
                )
                self.wb.define_metric("global_step")
                self.wb.define_metric("epoch")
                for prefix in ("train/*", "lambda/*", "contrib/*", "reward/*"):
                    self.wb.define_metric(prefix, step_metric="global_step")
                for prefix in ("val/*", "epoch/*"):
                    self.wb.define_metric(prefix, step_metric="epoch")
                log(f"W&B run: {self.wb.url or '(offline)'}")
            except Exception as e:
                log(f"W&B disabled ({e})")
                self.wb = None

    # ── API ────────────────────────────────────────────────────────────────
    def log_step(self, step: int, metrics: Dict[str, float], extra: Optional[Dict] = None):
        metrics = _numeric(metrics)
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)
        if self.wb:
            self.wb.log({**metrics, "global_step": step})
        if self.steps_csv:
            self.steps_csv.write({"global_step": step, **(extra or {}),
                                  **{_csv_key(k): round(float(v), 6) for k, v in metrics.items()}})

    def log_epoch(self, epoch: int, metrics: Dict[str, float], extra: Optional[Dict] = None):
        metrics = _numeric(metrics)
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)
        if self.wb:
            self.wb.log({**metrics, "epoch": epoch})
        if self.epochs_csv:
            self.epochs_csv.write({"epoch": epoch, **(extra or {}),
                                   **{_csv_key(k): round(float(v), 6) for k, v in metrics.items()}})

    def log_examples(self, key: str, rows: List[Dict[str, Any]], step: Optional[int] = None):
        if not rows:
            return
        if self.wb:
            import wandb
            cols = list(rows[0])
            table = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])
            self.wb.log({key: table, **({"epoch": step} if step is not None else {})})
        if self.tb:
            md = "\n".join(f"- **gen**: {r.get('generated')}  \n  **ref**: {r.get('reference')}"
                           for r in rows)
            self.tb.add_text(key, md, step or 0)

    def summary(self, metrics: Dict[str, float]):
        if self.wb:
            for k, v in metrics.items():
                self.wb.summary[k] = v

    def close(self):
        if self.tb:
            self.tb.close()
        for c in (self.steps_csv, self.epochs_csv):
            if c:
                c.close()
        if self.wb:
            self.wb.finish()


def _numeric(d: Dict) -> Dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _csv_key(k: str) -> str:
    return k.replace("/", "_")


def _run_id(run_dir: Path) -> str:
    """W&B id stored in the run directory: a resumed job continues the same W&B run, while a
    fresh attempt (new directory after --force) gets a new one."""
    import uuid
    f = run_dir / "wandb_run_id.txt"
    if f.exists():
        return f.read_text().strip()
    rid = uuid.uuid4().hex[:16]
    f.write_text(rid)
    return rid
