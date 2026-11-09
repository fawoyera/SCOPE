"""Generation and scoring shared by validation (during training) and the final test pass.

Every evaluated example is written with its per-example metrics, so the
statistics module can run *paired* tests across methods and seeds.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch

from ..config import ExperimentConfig
from ..data import render, tokenize_prompts_left
from ..data.datasets import prompt_adds_special_tokens
from ..decoding import make_gcd_processor
from ..domain import Artefacts, GrammarScorer
from ..metrics import c_phr, c_phr_macro, c_tok, semantic_scores, words
from ..metrics.bertscore import bertscore_both
from ..models import input_device
from ..utils.common import log
from ..utils.profiling import _sync


class _TimedProcessor:
    """Wraps a logits processor and accumulates the time spent computing masks."""

    def __init__(self, inner, sink: Dict[str, float]):
        self.inner, self.sink = inner, sink

    def __call__(self, input_ids, scores):
        t = time.perf_counter()
        out = self.inner(input_ids, scores)
        self.sink["mask_seconds"] = self.sink.get("mask_seconds", 0.0) + time.perf_counter() - t
        return out

METRICS = ["c_tok", "c_phr", "c_cfg", "c_bar", "slot_f1", "da_f1", "halluc", "bertscore_d", "bertscore_g"]


@torch.no_grad()
def generate(model, tokenizer, prompts: List[str], cfg: ExperimentConfig,
             artefacts: Optional[Artefacts] = None, use_gcd: bool = False,
             progress_every: int = 0, roles: Optional[List[str]] = None,
             timing: Optional[Dict] = None) -> List[str]:
    """Greedy (or beam, under GCD) decoding.  With ``gcd.level == 'cfg'`` prompts are grouped
    by response role so each batch is constrained by its role's grammar."""
    from transformers import LogitsProcessorList
    model.eval()
    ev = cfg.eval
    add = prompt_adds_special_tokens(cfg)
    max_prompt = cfg.model.max_length - ev.max_new_tokens
    dev = input_device(model)
    cfg_mode = use_gcd and cfg.gcd.level == "cfg"
    if cfg_mode:
        from ..decoding.cfg_gcd import get_cfg_decoder
        decoder = get_cfg_decoder(artefacts.grammar_path, tokenizer)
    roles = roles or ["unified"] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: roles[i]) if cfg_mode else list(range(len(prompts)))
    outs: List[Optional[str]] = [None] * len(prompts)
    timing = timing if timing is not None else {}
    timing.setdefault("per_example_ms", [None] * len(prompts))
    timing.setdefault("per_example_tokens", [None] * len(prompts))
    done = 0
    i = 0
    while i < len(order):
        idx = order[i:i + ev.batch_size]
        if cfg_mode:                                   # one role per batch
            idx = [j for j in idx if roles[j] == roles[idx[0]]]
        i += len(idx)
        enc = tokenize_prompts_left(tokenizer, [prompts[j] for j in idx], add, max_prompt).to(dev)
        P = enc["input_ids"].size(1)
        procs, beams = None, 1
        if use_gcd:
            beams = max(1, cfg.gcd.num_beams)
            proc = (decoder.processor(roles[idx[0]], P) if cfg_mode else
                    make_gcd_processor(tokenizer, artefacts, P, cfg.gcd.level, cfg.gcd.successor_fallback))
            procs = LogitsProcessorList([_TimedProcessor(proc, timing)])
        _sync()
        t0 = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=ev.max_new_tokens, do_sample=False, num_beams=beams,
                             repetition_penalty=ev.repetition_penalty, logits_processor=procs,
                             pad_token_id=tokenizer.pad_token_id, temperature=None, top_p=None, top_k=None)
        _sync()
        dt = time.perf_counter() - t0
        timing["generate_seconds"] = timing.get("generate_seconds", 0.0) + dt
        timing["prompt_tokens"] = timing.get("prompt_tokens", 0) + int(enc["attention_mask"].sum())
        gen = out[:, P:]
        stop = (gen == tokenizer.eos_token_id) | (gen == tokenizer.pad_token_id)
        n_tok = torch.where(stop.any(-1), stop.float().argmax(-1) + 1, torch.full_like(stop[:, 0], gen.size(1), dtype=torch.long))
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, t, k in zip(idx, texts, n_tok.tolist()):
            outs[j] = t.strip()
            timing["per_example_ms"][j] = 1000 * dt / len(idx)        # amortised over the batch
            timing["per_example_tokens"][j] = int(k)
        done += len(idx)
        if progress_every and done // progress_every != (done - len(idx)) // progress_every:
            log(f"  generated {done}/{len(prompts)}")
    return outs


def score_examples(rows: List[Dict], artefacts: Artefacts, grammar: GrammarScorer,
                   cfg: ExperimentConfig, with_bertscore: bool = True) -> List[Dict]:
    """rows: dicts with generated/reference/request/role.  Adds per-example metrics in place."""
    gens = [r["generated"] for r in rows]
    cfgs = grammar.score(gens, [r["role"] for r in rows], partial=True)
    for r, cc in zip(rows, cfgs):
        w = words(r["generated"])
        r["c_tok"] = c_tok(w, artefacts.vocab)
        r["c_phr"] = c_phr(w, artefacts.ngrams)
        r["c_phr_macro"] = c_phr_macro(w, artefacts.ngrams)
        r["c_cfg"] = cc
        r["c_bar"] = (r["c_tok"] + r["c_phr"] + r["c_cfg"]) / 3
        r.update(semantic_scores(r["generated"], r["reference"], r["request"]))
    if with_bertscore:
        d, g = bertscore_both(gens, [r["reference"] for r in rows], cfg.eval.bertscore)
        for k, vals in (("bertscore_d", d), ("bertscore_g", g)):
            if vals is not None:
                for r, v in zip(rows, vals):
                    r[k] = v
    return rows


def aggregate(rows: Sequence[Dict]) -> Dict[str, float]:
    out = {"n": len(rows)}
    for k in METRICS + ["c_phr_macro"]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    if "halluc" in out:
        out["halluc_pct"] = 100 * out["halluc"]
    return out


def evaluate(model, tokenizer, pairs: List[dict], cfg: ExperimentConfig, artefacts: Artefacts,
             grammar: GrammarScorer, use_gcd: bool = False, with_bertscore: bool = True,
             max_pairs: int = 0):
    """Generate for ``pairs`` (first ``max_pairs`` in file order) and score.  Returns (rows, summary)."""
    pairs = pairs[:max_pairs] if max_pairs else pairs
    rendered = [render(p, tokenizer, cfg) for p in pairs]
    timing: Dict = {}
    gens = generate(model, tokenizer, [r["prompt"] for r in rendered], cfg, artefacts, use_gcd,
                    progress_every=200, roles=[r["role"] for r in rendered], timing=timing)
    rows = [{"pair_id": p.get("pair_id", str(i)), "role": r["role"], "request": r["request"],
             "reference": r["target"], "generated": g, "tier": p.get("quality_tier"),
             "exchange_type": p.get("exchange_type"),
             "latency_ms": timing["per_example_ms"][i], "gen_tokens": timing["per_example_tokens"][i]}
            for i, (p, r, g) in enumerate(zip(pairs, rendered, gens))]
    score_examples(rows, artefacts, grammar, cfg, with_bertscore)
    summary = aggregate(rows)
    toks = sum(r["gen_tokens"] for r in rows)
    gs = timing.get("generate_seconds", 0.0)
    summary["inference"] = {
        "batch_size": cfg.eval.batch_size, "num_beams": cfg.gcd.num_beams if use_gcd else 1,
        "decoding": (f"gcd-{cfg.gcd.level}" if use_gcd else "unconstrained"),
        "generate_seconds": gs, "examples": len(rows), "generated_tokens": toks,
        "prompt_tokens": timing.get("prompt_tokens", 0),
        "ms_per_example_amortised": 1000 * gs / max(1, len(rows)),
        "ms_per_token_amortised": 1000 * gs / max(1, toks),
        "tokens_per_example": toks / max(1, len(rows)),
        "mask_seconds": timing.get("mask_seconds", 0.0),
        "mask_fraction": timing.get("mask_seconds", 0.0) / max(1e-9, gs),
    }
    return rows, summary
