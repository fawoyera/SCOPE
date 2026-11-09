"""Loss curriculum and GradNorm re-weighting of the λ vector."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..config import ObjectiveConfig

TASKS = ("ce", "tok", "phr", "cfg")


@dataclass
class Phase:
    name: str
    active: Dict[str, bool]


class LossSchedule:
    """Phase 1: CE → Phase 2: CE+tok → Phase 3: all (only losses with λ>0 ever activate).

    New λ's are ramped linearly over ``curriculum_ramp_steps`` optimizer steps after a
    phase transition.  ``lambdas()`` returns base_λ × ramp × GradNorm weight.
    """

    def __init__(self, cfg: ObjectiveConfig, total_epochs: int):
        self.cfg = cfg
        self.base = {"ce": cfg.lambda_ce, "tok": cfg.lambda_tok, "phr": cfg.lambda_phr, "cfg": cfg.lambda_cfg}
        self.enabled = {k: v > 0 for k, v in self.base.items()}
        self.total = total_epochs
        self.phase = None
        self._ramp_start: Dict[str, int] = {}

    def phase_for(self, epoch: int) -> Phase:
        en = self.enabled
        if not self.cfg.loss_curriculum:
            return Phase("full", dict(en))
        p1 = max(1, math.ceil(self.cfg.curriculum_phase1 * self.total))
        p2 = max(p1 + 1, math.ceil(self.cfg.curriculum_phase2 * self.total))
        if epoch < p1:
            return Phase("P1:CE", {"ce": True, "tok": False, "phr": False, "cfg": False})
        if epoch < p2:
            return Phase("P2:CE+tok", {"ce": True, "tok": en["tok"], "phr": False, "cfg": False})
        return Phase("P3:all", dict(en))

    def start_epoch(self, epoch: int, opt_step: int) -> Phase:
        new = self.phase_for(epoch)
        if self.phase is not None and self.cfg.curriculum_ramp_steps > 0:
            for k in TASKS:
                if new.active[k] and not self.phase.active[k]:
                    self._ramp_start[k] = opt_step
        self.phase = new
        return new

    def ramp(self, task: str, opt_step: int) -> float:
        if task not in self._ramp_start:
            return 1.0
        return min(1.0, (opt_step - self._ramp_start[task]) / max(1, self.cfg.curriculum_ramp_steps))

    def lambdas(self, opt_step: int, weights: Dict[str, float]) -> Dict[str, float]:
        return {k: (self.base[k] * self.ramp(k, opt_step) * weights.get(k, 1.0)
                    if self.phase.active[k] else 0.0) for k in TASKS}


class GradNorm:
    """GradNorm (Chen et al., 2018) over the active SCOPE terms.

    Task "loss levels" L_i(t) used for the relative inverse training rate are
    L_CE, L_tok and (1 − mean reward) for the two GRPO terms.  Gradient norms
    G_i = ‖∇_W λ_i L_i‖ are measured on the last decoder block W.  Weights are
    renormalised to sum to the number of active tasks and floored at 0.01.
    """

    def __init__(self, alpha: float, lr: float):
        self.alpha = alpha
        self.w = torch.nn.Parameter(torch.ones(len(TASKS)))
        self.opt = torch.optim.Adam([self.w], lr=lr)
        self.L0: Dict[str, float] = {}

    def weights(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(TASKS, self.w.detach())}

    def update(self, levels: Dict[str, float], gnorms: Dict[str, float]) -> None:
        keys = [k for k in TASKS if k in gnorms and k in levels]
        if len(keys) < 2:
            return
        for k in keys:
            self.L0.setdefault(k, max(levels[k], 1e-8))
        idx = torch.tensor([TASKS.index(k) for k in keys])
        w = self.w[idx]
        G = w * torch.tensor([gnorms[k] for k in keys])           # ‖∇ w_i λ_i L_i‖ = w_i ‖∇ λ_i L_i‖
        r = torch.tensor([max(levels[k], 1e-8) / self.L0[k] for k in keys])
        target = (G.mean() * (r / r.mean()) ** self.alpha).detach()
        loss = F.l1_loss(G, target, reduction="sum")
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.w.clamp_(min=0.01)
            self.w[idx] *= len(keys) / self.w[idx].sum()

    def state_dict(self):
        return {"w": self.w.detach().tolist(), "L0": self.L0}
