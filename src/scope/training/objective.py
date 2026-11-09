"""SCOPE trainer (Algorithm 1).  SFT is the same trainer with λ = (1, 0, 0, 0).

One micro-step:
  Phase 1  forward on (x, y*)                → L_CE (+ tier weights), L_tok      [differentiable]
  Phase 2  sample M completions per prompt   → R_phr = C_phr, R_cfg = C_cfg      [every k steps]
  Phase 3  group-relative advantages A_phr, A_cfg
  Phase 4  ∇[λ_CE L_CE + λ_tok L_tok] + Σ_m ∇[−(λ_phr A_phr + λ_cfg A_cfg) · log p_θ(ŷ^(m)|x)]
GradNorm (optional) rescales the λ's from per-task gradient norms on the last block.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..data import SFTDataset, collate_sft, curriculum_pairs, tokenize_prompts_left
from ..data.datasets import prompt_adds_special_tokens
from ..losses import (TASKS, GradNorm, LexicalPenalty, LossSchedule, group_advantages, length_penalty,
                      masked_ce, sequence_logprobs, vocab_token_mask)
from ..metrics import c_phr, words
from ..models import input_device, shared_parameters
from ..utils.common import log
from ..utils.profiling import FWD, FWD_BWD
from .base import BaseTrainer


def _grad_norm(loss: torch.Tensor, params) -> float:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = [g.float().pow(2).sum() for g in gs if g is not None]
    return float(torch.stack(sq).sum().sqrt()) if sq else 0.0


class ObjectiveTrainer(BaseTrainer):
    job_type = "train"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        o = self.cfg.objective
        self.schedule = LossSchedule(o, self.cfg.train.epochs)
        self.gradnorm = GradNorm(o.gradnorm_alpha, o.gradnorm_lr) if o.gradnorm else None
        self.shared = shared_parameters(self.model) if self.gradnorm else None
        vsize = self.model.get_output_embeddings().weight.size(0)
        self.lexical = LexicalPenalty(vocab_token_mask(self.tok, self.art.vocab, vsize)) \
            if o.lambda_tok > 0 else None
        self.ref_model = None
        if o.grpo.kl_beta > 0 and (o.lambda_phr > 0 or o.lambda_cfg > 0):
            log("Loading frozen reference policy for the GRPO KL term")
            self.ref_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self._levels: Dict[str, float] = {}
        self._gnorms: Dict[str, float] = {}
        self._lam = {k: 0.0 for k in TASKS}
        self._last_rewards = {"phr": 0.0, "cfg": 0.0}

    # ── data ────────────────────────────────────────────────────────────────
    def build_train_loader(self, epoch: int):
        pairs = curriculum_pairs(self.splits.train, self.cfg.data, epoch, self.cfg.train.epochs)
        ds = SFTDataset(pairs, self.tok, self.cfg)
        g = torch.Generator().manual_seed(self.cfg.run.seed + epoch)
        return DataLoader(ds, batch_size=self.cfg.train.batch_size, shuffle=True, generator=g,
                          collate_fn=collate_sft(self.tok.pad_token_id),
                          num_workers=self.cfg.train.num_workers)

    def on_epoch_start(self, epoch: int) -> Dict[str, str]:
        ph = self.schedule.start_epoch(epoch, self.opt_step)
        active = "+".join(k for k in TASKS if ph.active[k])
        log(f"── epoch {epoch + 1}: loss phase {ph.name} [{active}] | data curriculum {self.cfg.data.curriculum}")
        return {"phase": ph.name}

    # ── one micro-batch ────────────────────────────────────────────────────
    def micro_step(self, batch: Dict, scale: float) -> Dict[str, float]:
        o = self.cfg.objective
        weights = self.gradnorm.weights() if self.gradnorm else {}
        lam = self.schedule.lambdas(self.opt_step, weights)
        self._lam = lam
        active = self.schedule.phase.active
        dev = input_device(self.model)
        gn_step = (self.gradnorm is not None and self.opt_step % o.gradnorm_every == 0
                   and sum(active.values()) >= 2)
        self._gnorms = {}

        # Phase 1: differentiable terms
        prof = self.prof
        with prof.section("ce_tok_fwd_bwd"):
            out = self.model(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
                             use_cache=False)
            w = batch["weight"] if self.cfg.data.curriculum == "B" else None
            L_ce = masked_ce(out.logits, batch["labels"], w)
            L_tok = self.lexical(out.logits, batch["labels"]) if (self.lexical and active["tok"]) else None
            terms = {"ce": lam["ce"] * L_ce}
            if L_tok is not None:
                terms["tok"] = lam["tok"] * L_tok
        if gn_step:
            with prof.section("gradnorm"):
                for k, v in terms.items():
                    self._gnorms[k] = _grad_norm(v, self.shared)
        with prof.section("ce_tok_fwd_bwd"):
            (sum(terms.values()) * scale).backward()
        prof.count("ce_tok_fwd_bwd", int(batch["attention_mask"].sum()), FWD_BWD)
        metrics = {"loss_ce": L_ce.item(), "loss_tok": L_tok.item() if L_tok is not None else 0.0}
        self._levels = {"ce": metrics["loss_ce"], "tok": metrics["loss_tok"]}
        del out

        # Phases 2–4: GRPO on the compliance rewards
        rl = (active["phr"] or active["cfg"]) and self.micro % o.grpo.every_n_steps == 0
        if rl:
            metrics.update(self._grpo(batch, lam, scale, gn_step))
        metrics["reward_phr"] = self._last_rewards["phr"]
        metrics["reward_cfg"] = self._last_rewards["cfg"]
        metrics["loss_total"] = (lam["ce"] * metrics["loss_ce"] + lam["tok"] * metrics["loss_tok"]
                                 + lam["phr"] * (1 - metrics["reward_phr"]) + lam["cfg"] * (1 - metrics["reward_cfg"]))
        return metrics

    def _grpo(self, batch: Dict, lam: Dict[str, float], scale: float, gn_step: bool) -> Dict[str, float]:
        g = self.cfg.objective.grpo
        M, B = g.num_samples, len(batch["prompt"])
        dev = input_device(self.model)
        enc = tokenize_prompts_left(self.tok, batch["prompt"], prompt_adds_special_tokens(self.cfg),
                                    self.cfg.model.max_length - g.max_new_tokens).to(dev)
        P = enc["input_ids"].size(1)

        # Phase 2: sample a group of M completions per prompt (rows ordered b*M + m)
        prof = self.prof
        self.model.eval()
        with prof.section("rollout_generation"), torch.no_grad():
            seq = self.model.generate(**enc, do_sample=True, temperature=g.temperature, top_p=g.top_p,
                                      max_new_tokens=g.max_new_tokens, num_return_sequences=M,
                                      pad_token_id=self.tok.pad_token_id)
        self.model.train()
        gen = seq[:, P:]
        eos = self.tok.eos_token_id
        is_eos = gen == eos
        after_eos = (is_eos.long().cumsum(-1) - is_eos.long()) > 0           # tokens after the first EOS
        gen_mask = (~after_eos) & (gen != self.tok.pad_token_id) | (is_eos & ~after_eos)
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        roles = [r for r in batch["role"] for _ in range(M)]
        refs = [t for t in batch["target"] for _ in range(M)]

        # generation cost: prompt prefill + one forward per generated token (KV cache)
        prof.count("rollout_generation", int(enc["attention_mask"].sum()) * M + int(gen_mask.sum()), FWD)
        with prof.section("reward_scoring"):
            r_phr = [c_phr(words(t), self.art.ngrams) for t in texts]
            r_cfg = (self.grammar.score(texts, roles, partial=g.partial_credit) if lam["cfg"] > 0
                     else [0.0] * len(texts))
        if g.length_penalty:
            r_phr = [length_penalty(r, len(t.split()), len(ref.split())) for r, t, ref in zip(r_phr, texts, refs)]
            r_cfg = [length_penalty(r, len(t.split()), len(ref.split())) for r, t, ref in zip(r_cfg, texts, refs)]
        R_phr = torch.tensor(r_phr).view(B, M)
        R_cfg = torch.tensor(r_cfg).view(B, M)
        self._last_rewards = {"phr": R_phr.mean().item(), "cfg": R_cfg.mean().item()}
        self._levels.update({"phr": 1 - self._last_rewards["phr"], "cfg": 1 - self._last_rewards["cfg"]})

        # Phase 3: group-relative advantages, combined into one per-sample coefficient
        A_phr, A_cfg = group_advantages(R_phr), group_advantages(R_cfg)
        coef = (lam["phr"] * A_phr + lam["cfg"] * A_cfg).view(-1)          # (B*M,)

        # Phase 4: policy-gradient backward, one group slice at a time to bound memory
        prompt_ids = enc["input_ids"].repeat_interleave(M, 0)
        prompt_att = enc["attention_mask"].repeat_interleave(M, 0)
        full_ids = torch.cat([prompt_ids, gen], 1)
        full_att = torch.cat([prompt_att, gen_mask.long()], 1)
        labels = torch.cat([torch.full_like(prompt_ids, -100), gen.masked_fill(~gen_mask, -100)], 1)
        kl_sum, n_tok_total = 0.0, 0
        for m in range(M):
            rows = torch.arange(B, device=full_ids.device) * M + m
            if gen_mask[rows].sum() == 0:
                continue
            sec = prof.section("policy_gradient_fwd_bwd")
            sec.__enter__()
            lp_sum, n_tok, tok_lp = sequence_logprobs(self.model, full_ids[rows], full_att[rows], labels[rows],
                                                      tail_only=True, return_token_logps=g.kl_beta > 0)
            prof.count("policy_gradient_fwd_bwd", int(full_att[rows].sum()), FWD_BWD)
            lp = lp_sum / n_tok.clamp(min=1) if g.token_reduction == "mean" else lp_sum
            c = coef[rows.cpu()].to(lp.device)
            pg = -(c * lp).sum() / (B * M)
            if self.ref_model is not None:
                with torch.no_grad():
                    _, _, ref_lp = sequence_logprobs(self.ref_model, full_ids[rows], full_att[rows], labels[rows],
                                                     tail_only=True, return_token_logps=True)
                d = (ref_lp.to(tok_lp.device) - tok_lp)
                k3 = (d.exp() - d - 1) * (labels[rows][:, -tok_lp.size(1):].to(tok_lp.device) != -100)
                kl = k3.sum() / n_tok.sum().clamp(min=1)
                pg = pg + g.kl_beta * (lam["phr"] + lam["cfg"]) * kl / M
                kl_sum += kl.item()
                prof.count("reference_fwd", int(full_att[rows].sum()), FWD)
            if gn_step and m == 0:
                # gradient norm of each RL term on the shared block (first group slice)
                for k in ("phr", "cfg"):
                    if lam[k] > 0:
                        ck = (lam[k] * (A_phr if k == "phr" else A_cfg)).view(-1)[rows.cpu()].to(lp.device)
                        self._gnorms[k] = _grad_norm(-(ck * lp).sum() / B, self.shared)
            (pg * scale).backward()
            sec.__exit__(None, None, None)
            n_tok_total += int(n_tok.sum())
        return {"grpo_gen_len": n_tok_total / max(1, B * M), "grpo_kl": kl_sum / M,
                "grpo_adv_abs": coef.abs().mean().item()}

    # ── hooks ───────────────────────────────────────────────────────────────
    def before_optimizer_step(self) -> None:
        if self.gradnorm is not None and len(self._gnorms) >= 2:
            self.gradnorm.update(self._levels, self._gnorms)

    def extra_step_metrics(self) -> Dict[str, float]:
        out = {f"lambda/{k}": v for k, v in self._lam.items()}
        lv = {"ce": self._levels.get("ce", 0), "tok": self._levels.get("tok", 0),
              "phr": 1 - self._last_rewards["phr"], "cfg": 1 - self._last_rewards["cfg"]}
        out.update({f"contrib/{k}": self._lam[k] * lv[k] for k in TASKS})
        out.update({f"reward/{k}": v for k, v in self._last_rewards.items()})
        return out
