"""Gradient-path tests for the SCOPE objective on the tiny debug model (CPU)."""
from pathlib import Path

import pytest
import torch

TINY = Path(__file__).resolve().parents[1] / "models" / "tiny-debug"
DATA = Path(__file__).resolve().parents[1] / "data" / "debug_subset.jsonl"
pytestmark = pytest.mark.skipif(not (TINY.exists() and DATA.exists()),
                                reason="needs models/tiny-debug and data/debug_subset.jsonl")


class _FakeGrammar:
    def score(self, texts, roles, partial=True):
        g = torch.Generator().manual_seed(len(texts))
        return torch.rand(len(texts), generator=g).tolist()


def _trainer(tmp_path, monkeypatch, **over):
    from scope.config import load_config
    from scope.data import load_splits
    from scope.domain import Artefacts
    from scope.models import load_model, load_tokenizer
    from scope.training import ObjectiveTrainer
    import scope.training.objective as obj

    ov = [f"data.path={DATA}", f"run.run_dir={tmp_path}", "logging.wandb.enabled=false",
          "objective.loss_curriculum=false", "objective.gradnorm_every=1"] + [f"{k}={v}" for k, v in over.items()]
    cfg = load_config("tiny_debug", "scope", overrides=ov)
    tok, model = load_tokenizer(cfg.model), load_model(cfg.model)
    # random compliance rewards so that group advantages are non-zero
    monkeypatch.setattr(obj, "c_phr", lambda w, ng: float(torch.rand(1)))

    class _T:
        def log_step(self, *a, **k): pass
    t = ObjectiveTrainer(cfg, model, tok, load_splits(cfg.data), Artefacts.load(cfg.artefacts),
                         _FakeGrammar(), _T())
    t.schedule.start_epoch(0, 0)
    return t, model


def test_grpo_branch_produces_gradients_and_gradnorm(tmp_path, monkeypatch):
    t, model = _trainer(tmp_path, monkeypatch)
    batch = next(iter(t.build_train_loader(0)))
    t.micro = 0                                  # RL branch runs on micro % every_n == 0
    m = t.micro_step(batch, 1.0)
    assert m["grpo_adv_abs"] > 0
    assert {"ce", "tok", "phr", "cfg"} <= set(t._gnorms)
    g = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert g > 0
    w0 = t.gradnorm.weights()
    t.before_optimizer_step()
    assert t.gradnorm.weights() != w0            # GradNorm moved the weights
    assert abs(sum(t.gradnorm.weights().values()) - 4) < 1e-4


def test_rl_only_gradient_differs_from_ce(tmp_path, monkeypatch):
    t, model = _trainer(tmp_path, monkeypatch, **{"objective.lambda_ce": 0.0, "objective.lambda_tok": 0.0,
                                                  "objective.gradnorm": "false", "objective.grpo.kl_beta": 0.1})
    batch = next(iter(t.build_train_loader(0)))
    t.micro = 0
    m = t.micro_step(batch, 1.0)
    assert m["loss_ce"] > 0 and m["grpo_kl"] >= 0
    g = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert g > 0                                 # gradient comes from the policy-gradient term alone
