"""Training loop shared by every trained method.

Subclasses implement ``build_train_loader(epoch)`` and ``micro_step(batch, ctx)``;
the base class owns optimisation (gradient accumulation, clipping, cosine
schedule), logging, validation, checkpoint selection, early stopping and resume.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from ..config import ExperimentConfig
from ..data import Splits
from ..domain import Artefacts, GrammarScorer
from ..evaluation import evaluate
from ..models import n_gpus
from ..tracking import Tracker
from ..utils.common import fmt_elapsed, log, mirror_to_depot, write_jsonl
from ..utils.profiling import Profiler


def build_optimizer(model, cfg: ExperimentConfig):
    t = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]
    use8 = t.optimizer == "adamw8bit" or (t.optimizer == "auto" and n_gpus() >= 2)
    if use8:
        try:
            import bitsandbytes as bnb
            log("Optimizer: AdamW8bit")
            return bnb.optim.AdamW8bit(params, lr=t.lr, weight_decay=t.weight_decay)
        except ImportError:
            log("bitsandbytes missing — falling back to AdamW")
    log("Optimizer: AdamW")
    return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)


class BaseTrainer:
    job_type = "train"

    def __init__(self, cfg: ExperimentConfig, model, tokenizer, splits: Splits,
                 artefacts: Artefacts, grammar: GrammarScorer, tracker: Tracker):
        self.cfg, self.model, self.tok = cfg, model, tokenizer
        self.splits, self.art, self.grammar, self.tracker = splits, artefacts, grammar, tracker
        self.run_dir = Path(cfg.run.run_dir)
        self.opt_step = 0
        self.micro = 0
        self.best = -math.inf
        self.best_epoch = 0
        self.prof = Profiler()
        self.train_seconds = 0.0
        self.samples_seen = 0

    # ── to implement ────────────────────────────────────────────────────────
    def build_train_loader(self, epoch: int):
        raise NotImplementedError

    def micro_step(self, batch: Dict, scale: float) -> Dict[str, float]:
        """Forward + backward of one micro-batch (losses already multiplied by ``scale``)."""
        raise NotImplementedError

    def on_epoch_start(self, epoch: int) -> Dict[str, str]:
        return {}

    def before_optimizer_step(self) -> None:
        pass

    def extra_step_metrics(self) -> Dict[str, float]:
        return {}

    # ── loop ────────────────────────────────────────────────────────────────
    def _steps_per_epoch(self, n_batches: int) -> int:
        return math.ceil(n_batches / self.cfg.train.grad_accum)

    def train(self) -> Optional[Path]:
        from transformers import get_cosine_schedule_with_warmup
        t = self.cfg.train
        self.optimizer = build_optimizer(self.model, self.cfg)
        n_batches = len(self.build_train_loader(0))
        total = self._steps_per_epoch(n_batches) * t.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, max(1, int(t.warmup_ratio * total)), total)
        start_epoch = self._maybe_resume()
        no_improve = 0
        t_start = time.time()
        log(f"Training {t.epochs} epochs | ~{total} optimizer steps | "
            f"micro-batch {t.batch_size} × accum {t.grad_accum}")

        for epoch in range(start_epoch, t.epochs):
            t_ep = time.time()
            info = self.on_epoch_start(epoch)
            loader = self.build_train_loader(epoch)
            self.model.train()
            sums: Dict[str, float] = {}
            n = 0
            self.optimizer.zero_grad(set_to_none=True)
            full = (len(loader) // t.grad_accum) * t.grad_accum
            win: Dict[str, list] = {}
            for i, batch in enumerate(loader):
                last_in_window = (i + 1) % t.grad_accum == 0 or i + 1 == len(loader)
                window = t.grad_accum if i < full else len(loader) - full
                m = self.micro_step(batch, 1.0 / window)
                self.micro += 1
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + v
                    win.setdefault(k, []).append(v)
                n += 1
                self.samples_seen += len(batch["input_ids"]) if "ref_chosen" not in batch else len(batch["ref_chosen"])
                if last_in_window:
                    self.before_optimizer_step()
                    with self.prof.section("optimizer_step"):
                        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip).item()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.opt_step += 1
                    if self.opt_step % t.log_every == 0:
                        # average over the micro-batches of this accumulation window
                        metrics = {f"train/{k}": sum(v) / len(v) for k, v in win.items()}
                        metrics.update({"train/lr": self.scheduler.get_last_lr()[0], "train/grad_norm": gn})
                        metrics.update(self.extra_step_metrics())
                        self.tracker.log_step(self.opt_step, metrics, {"epoch": epoch + 1, **info})
                        log(f"E{epoch + 1} step {self.opt_step}/{total} | " +
                            " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()
                                     if k.startswith(("train/", "lambda/"))))
                    win = {}
                    if t.checkpoint_every_steps and self.opt_step % t.checkpoint_every_steps == 0:
                        self._save_step_checkpoint(epoch)
            avg = {f"epoch/train_{k}": v / max(1, n) for k, v in sums.items()}
            self.train_seconds += time.time() - t_ep

            # ── validation & selection ─────────────────────────────────────
            if t.checkpoint_every_steps:
                self._save_step_checkpoint(epoch + 1)
            with self.prof.section("validation"):
                rows, val = evaluate(self.model, self.tok, self.splits.val, self.cfg, self.art, self.grammar,
                                     use_gcd=False, with_bertscore=False, max_pairs=t.val_max_pairs)
            avg["epoch/wall_seconds"] = time.time() - t_ep
            avg["epoch/samples_per_sec"] = self.samples_seen / max(1e-9, self.train_seconds)
            write_jsonl(self.run_dir / f"val_predictions_e{epoch + 1}.jsonl", rows)
            vinf = val.pop("inference", {})
            vm = {f"val/{k}": v for k, v in val.items() if k != "n"}
            vm["val/ms_per_example"] = vinf.get("ms_per_example_amortised", 0.0)
            self.tracker.log_epoch(epoch + 1, {**avg, **vm}, info)
            self.tracker.log_examples("val/examples",
                                      [{k: r[k] for k in ("request", "reference", "generated", "c_bar")}
                                       for r in rows[: self.cfg.logging.log_examples]], epoch + 1)
            score = val.get(t.checkpoint_metric, 0.0)
            ok_hall = val.get("halluc", 0.0) <= t.hallucination_threshold
            log(f"Epoch {epoch + 1}/{t.epochs} ({fmt_elapsed(t_ep)}) | " +
                " ".join(f"{k}={val[k]:.4f}" for k in ("c_tok", "c_phr", "c_cfg", "c_bar", "slot_f1", "da_f1", "halluc")
                         if k in val))
            if score > self.best and ok_hall:
                self.best, self.best_epoch, no_improve = score, epoch + 1, 0
                self.save(self.run_dir / "checkpoint-best")
                log(f"  ✓ new best {t.checkpoint_metric}={score:.4f} (epoch {epoch + 1})")
                mirror_to_depot(self.run_dir, self.cfg.run.depot_path, self._rel())
            else:
                no_improve += 1
                why = "" if ok_hall else f" (halluc {val.get('halluc', 0):.2%} > threshold)"
                log(f"  no improvement {no_improve}/{t.early_stop_patience or '∞'}{why}")
            if t.early_stop_patience and no_improve >= t.early_stop_patience:
                log(f"Early stopping after epoch {epoch + 1}")
                break

        best = self.run_dir / "checkpoint-best"
        if not best.exists():  # nothing passed the hallucination gate: keep the last model
            log("No epoch passed the checkpoint gate — saving final model as checkpoint-best")
            self.save(best)
            self.best_epoch = epoch + 1
        for d in self.run_dir.glob("step_*"):
            shutil.rmtree(d, ignore_errors=True)
        self.tracker.summary({"best_epoch": self.best_epoch, f"best_val_{t.checkpoint_metric}": self.best})
        log(f"Training done in {fmt_elapsed(t_start)}; best epoch {self.best_epoch}")
        return best

    # ── checkpoints ─────────────────────────────────────────────────────────
    def _rel(self) -> str:
        r = self.cfg.run
        return f"{r.domain}/{r.model}/{r.method}/s{r.seed}"

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path), safe_serialization=True)
        self.tok.save_pretrained(str(path))
        with open(path / "training_state.json", "w") as f:
            json.dump({"opt_step": self.opt_step, "best": self.best, "best_epoch": self.best_epoch}, f)

    def _save_step_checkpoint(self, epoch: int) -> None:
        path = self.run_dir / f"step_{self.opt_step:07d}"
        self.save(path)
        with open(path / "training_state.json", "w") as f:
            json.dump({"epoch": epoch, "opt_step": self.opt_step, "best": self.best,
                       "best_epoch": self.best_epoch}, f)
        for d in self.run_dir.glob("step_*"):
            if d != path:
                shutil.rmtree(d, ignore_errors=True)

    def _maybe_resume(self) -> int:
        src = self.cfg.run.resume_from
        if not src:
            return 0
        state = json.load(open(Path(src) / "training_state.json"))
        from safetensors.torch import load_file
        sd = {}
        for shard in sorted(Path(src).glob("*.safetensors")):
            sd.update(load_file(str(shard)))
        self.model.load_state_dict(sd, strict=False)
        self.opt_step = state.get("opt_step", 0)
        self.best, self.best_epoch = state.get("best", -math.inf), state.get("best_epoch", 0)
        for _ in range(self.opt_step):
            self.scheduler.step()
        log(f"Resumed from {src}: epoch {state.get('epoch', 0)}, step {self.opt_step}")
        return int(state.get("epoch", 0))
