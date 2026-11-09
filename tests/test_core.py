"""Fast unit tests (CPU, no downloads).  Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
import torch

from scope.config import available, load_config, parse_overrides
from scope.domain import Artefacts
from scope.losses import LexicalPenalty, group_advantages, masked_ce
from scope.metrics import c_phr, c_tok, semantic_scores, text_to_digits
from scope.analysis.stats import holm, paired_bootstrap_ci, sign_flip_pvalue


def test_every_model_method_combination_resolves():
    for model in available("models"):
        for method in available("methods"):
            cfg = load_config(model, method, "atc", 123)
            assert cfg.run.run_dir.endswith(f"atc/{model}/{method}/s123")
            assert cfg.model.hf_id


def test_layering_and_overrides():
    cfg = load_config("gpt2_large", "sft", overrides=["train.epochs=1", "logging.wandb.mode=offline"])
    assert cfg.train.lr == 5e-5                     # model-specific method override
    assert cfg.train.epochs == 1 and cfg.logging.wandb.mode == "offline"
    assert load_config("llama31_8b", "sft").train.lr == 2e-5
    assert load_config("llama31_8b", "dpo").train.trainer == "dpo"
    with pytest.raises(KeyError):
        load_config("llama31_8b", "sft", overrides=["train.lrr=1"])
    assert parse_overrides(["a.b=3", "a.c=x"]) == {"a": {"b": 3, "c": "x"}}


def test_sft_is_scope_with_zero_regularisers():
    o = load_config("llama31_8b", "sft").objective
    assert (o.lambda_ce, o.lambda_tok, o.lambda_phr, o.lambda_cfg) == (1.0, 0.0, 0.0, 0.0)


def test_compliance_metrics():
    art = Artefacts.load(load_config("llama31_8b", "scope").artefacts)
    w = "DELTA SIX TURN LEFT HEADING".split()
    assert c_tok(w, art.vocab) == 1.0
    assert c_tok("PLEASE TURN".split(), art.vocab) == 0.5
    assert 0 < c_phr(w, art.ngrams) <= 1
    assert c_phr([], art.ngrams) == 0.0


def test_semantic_metrics():
    assert text_to_digits("HEADING ONE EIGHT ZERO") == "HEADING 180"
    s = semantic_scores("DELTA SIX TURN LEFT HEADING ONE EIGHT ZERO",
                        "DELTA SIX TURN LEFT HEADING ONE EIGHT ZERO", "")
    assert s["slot_f1"] == 1.0 and s["da_f1"] == 1.0 and s["halluc"] == 0.0
    assert semantic_scores("SQUAWK FOUR FIVE SIX SEVEN", "ROGER", "")["halluc"] == 1.0


def test_group_advantages():
    a = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(a[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.all(a[1] == 0)          # zero-variance group → no signal


def test_masked_ce_and_lexical_penalty():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, 1], [-100, 2, 2, -100, -100]])
    ce = masked_ce(logits, labels)
    ce_w = masked_ce(logits, labels, torch.tensor([1.0, 1.0]))
    assert ce.item() > 0 and ce_w.item() > 0
    allowed = torch.zeros(7, dtype=torch.bool); allowed[:3] = True
    lt = LexicalPenalty(allowed)(logits, labels)
    assert 0 < lt.item() < 1
    lt.backward()
    big = torch.full((1, 3, 7), -20.0); big[..., :3] = 20.0
    assert LexicalPenalty(allowed)(big, torch.tensor([[-100, 1, 1]])).item() < 1e-6


def test_stats_helpers():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, 400)
    lo, hi = paired_bootstrap_ci(d, rng)
    assert lo < d.mean() < hi
    assert sign_flip_pvalue(d, rng) < 0.01
    assert sign_flip_pvalue(np.zeros(10), rng) == 1.0
    adj = holm(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert adj.iloc[0] == pytest.approx(0.03) and np.isnan(adj.iloc[3])
    assert (adj.dropna() >= pd.Series([0.01, 0.04, 0.03])).all()


def test_every_ablation_variant_resolves_to_a_valid_config():
    from scope.config import parse_overrides
    from scope.runner import load_ablation_spec, plan_ablation
    spec = load_ablation_spec("scope")
    jobs = plan_ablation(spec, "atc", ["llama31_8b"], [42], "results")
    labels = {j.tag for j in jobs}
    assert {"scope", "sft", "scope__no_lcfg", "scope__full_gcd_cfg", "scope__sft_6ep"} <= labels
    for j in jobs:
        ov = [o for o in j.overrides]
        if j.label and j.label != j.method:
            ov.append(f"run.method={j.label}")
        cfg = load_config(j.model, j.method, "atc", 42, ov)       # raises on unknown keys
        assert cfg.run.run_dir.endswith(f"/{j.tag}/s42")
    by = {j.tag: j for j in jobs}
    assert by["scope__full_gcd_cfg"].dep.tag == "scope"
    assert by["scope__sft_gcd_cfg_beam3"].dep.tag == "sft"
    assert load_config("llama31_8b", "scope", overrides=by["scope__ce_only"].overrides).objective.lambda_cfg == 0


def test_prompt_styles_are_nested_and_default_is_rules():
    from scope.data.prompts import PROMPT_STYLES, SYSTEM_PROMPTS, build_prompt
    pair = {"req_text": "DELTA SIX TURN LEFT HEADING ONE EIGHT ZERO", "req_role": "controller",
            "resp_role": "pilot", "exchange_type": "instruction_readback", "resp_text": "LEFT ONE EIGHT ZERO DELTA SIX",
            "quality_tier": "bronze"}
    demo = dict(pair, req_text="UNITED TWO CLIMB AND MAINTAIN FIVE THOUSAND",
                resp_text="CLIMBING TO FIVE THOUSAND UNITED TWO", pair_id="d1")
    p = {s: build_prompt(pair, style=s, demonstrations=[demo])["prompt"] for s in PROMPT_STYLES}
    rule = SYSTEM_PROMPTS["atc"].splitlines()[0]
    assert build_prompt(pair)["prompt"] == p["rules"]                 # default unchanged
    assert rule not in p["minimal"] and "Context" not in p["minimal"]
    assert rule not in p["task"] and "Context" in p["task"]
    assert rule in p["rules"] and "Examples" not in p["rules"]
    assert rule in p["fewshot"] and "CLIMBING TO FIVE THOUSAND UNITED TWO" in p["fewshot"]
    assert len(p["minimal"]) < len(p["task"]) < len(p["rules"]) < len(p["fewshot"])
    for s in PROMPT_STYLES:
        assert pair["req_text"] in p[s] and p[s].rstrip().endswith(":")
    with pytest.raises(ValueError):
        build_prompt(pair, style="verbose")
