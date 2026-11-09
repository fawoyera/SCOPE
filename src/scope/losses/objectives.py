"""SCOPE loss components.

    L_SCOPE = λ_CE·L_CE + λ_tok·L_tok + λ_phr·L_phr + λ_cfg·L_cfg          (Eq. combined)

* ``masked_ce``          — L_CE over response tokens (Eq. lce_masked), optional tier weights.
* ``LexicalPenalty``     — L_tok: expected probability mass outside V (Eq. ltok), differentiable.
* ``group_advantages``   — A^(m) = (R^(m) − μ) / (σ + ε) within each prompt's group of M samples.
* ``sequence_logprobs``  — Σ/mean_t log p_θ(ŷ_t | ŷ_<t, x) over response tokens, for GRPO and DPO.
* ``dpo_loss``           — Rafailov et al. (2023).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..models import supports_logits_to_keep


# ── L_CE ────────────────────────────────────────────────────────────────────
def masked_ce(logits: torch.Tensor, labels: torch.Tensor,
              weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Response-masked CE.  With ``weights`` (data curriculum B) each sequence's mean
    token loss is scaled by its tier weight, then averaged over the batch."""
    lg = logits[:, :-1].float()
    tg = labels[:, 1:].to(lg.device)
    tok = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1),
                          ignore_index=-100, reduction="none").view(tg.shape)
    mask = (tg != -100).float()
    if weights is None:
        return (tok * mask).sum() / mask.sum().clamp(min=1)
    per_seq = (tok * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return (per_seq * weights.to(per_seq.device)).mean()


# ── L_tok ───────────────────────────────────────────────────────────────────
_SUBWORD_PREFIX = re.compile(r"^(Ġ|Ċ|▁|##)+")


def vocab_token_mask(tokenizer, vocab, size: int) -> torch.Tensor:
    """Boolean mask over the LM vocabulary: True for sub-word tokens compatible with V.

    Same rule as the original implementation: a token is allowed if its surface
    form (subword markers stripped, upper-cased) is a word of V, is empty, is
    numeric, or is whitespace/punctuation; special tokens are always allowed.
    """
    mask = torch.zeros(size, dtype=torch.bool)
    upper = {w.upper() for w in vocab}
    for tok, idx in tokenizer.get_vocab().items():
        if idx >= size:
            continue
        clean = _SUBWORD_PREFIX.sub("", tok).upper()
        if clean in upper or not clean or clean.isdigit() or re.fullmatch(r"[ ,.\-/]+", tok):
            mask[idx] = True
    for t in tokenizer.all_special_ids:
        if t is not None and t < size:
            mask[t] = True
    return mask


class LexicalPenalty:
    """L_tok = mean_{t∈resp} Σ_{v∉V} p_t(v) = mean_t (1 − Σ_{v∈V} p_t(v))."""

    def __init__(self, allowed: torch.Tensor):
        self.allowed = allowed
        self._idx = {}

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # position t predicts token t+1 → response positions are labels[:, 1:] != -100
        sel = (labels[:, 1:] != -100).to(logits.device)
        h = logits[:, :-1][sel].float()                          # (N_resp, |vocab|)
        if h.numel() == 0:
            return logits.sum() * 0.0
        dev = h.device
        if dev not in self._idx:
            self._idx[dev] = self.allowed[: h.size(-1)].nonzero().squeeze(-1).to(dev)
        log_in = torch.logsumexp(h.index_select(-1, self._idx[dev]), -1) - torch.logsumexp(h, -1)
        return (1.0 - log_in.exp()).mean()


# ── GRPO ────────────────────────────────────────────────────────────────────
def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """rewards: (B, M).  Per-prompt standardisation; a zero-variance group gives zero advantage."""
    mu = rewards.mean(-1, keepdim=True)
    sd = rewards.std(-1, keepdim=True, unbiased=True) if rewards.size(-1) > 1 else torch.zeros_like(mu)
    adv = (rewards - mu) / (sd + eps)
    return torch.where(sd < eps, torch.zeros_like(adv), adv)


def length_penalty(reward: float, n_gen: int, n_ref: int) -> float:
    """Damp rewards of degenerate short samples (e.g. a bare callsign), as in the original code."""
    if n_gen <= 2 and n_ref > 4:
        return reward * 0.05
    if n_gen > 0 and n_ref > 0 and n_gen / n_ref < 0.4:
        return reward * 0.2
    return reward


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      labels: torch.Tensor, tail_only: bool = False,
                      return_token_logps: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Per-sequence (sum log p, #tokens) over positions where labels != -100.

    ``tail_only``: all labelled tokens sit at the end of the (left-padded) sequence,
    so only the last K logits are materialised (``logits_to_keep``) — this is what
    makes the GRPO branch affordable with 128k–152k vocabularies.
    Position ids are derived from the attention mask so left padding is handled.
    """
    pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    kw = {}
    tgt = labels[:, 1:]
    if tail_only and supports_logits_to_keep(model):
        first = int((labels != -100).float().argmax(-1).min().item())
        keep = input_ids.size(1) - first + 1
        kw["logits_to_keep"] = keep
        tgt = labels[:, -(keep - 1):]
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos,
                use_cache=False, **kw)
    logits = out.logits[:, :-1].float()
    tgt = tgt.to(logits.device)
    mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    lp = lp * mask
    return lp.sum(-1), mask.sum(-1), (lp if return_token_logps else None)


# ── DPO ─────────────────────────────────────────────────────────────────────
def dpo_loss(pol_c: torch.Tensor, pol_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor,
             beta: float):
    """−log σ(β[(log π_c − log π_ref,c) − (log π_r − log π_ref,r)])."""
    margin = beta * ((pol_c - ref_c.to(pol_c.device)) - (pol_r - ref_r.to(pol_r.device)))
    loss = -F.logsigmoid(margin).mean()
    return loss, {"dpo/margin": margin.mean().item(),
                  "dpo/accuracy": (margin > 0).float().mean().item(),
                  "dpo/reward_chosen": (beta * (pol_c - ref_c.to(pol_c.device))).mean().item(),
                  "dpo/reward_rejected": (beta * (pol_r - ref_r.to(pol_r.device))).mean().item()}
