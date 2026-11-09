"""DPO baseline.

1. ``build_preference_pairs``: for each training prompt x, the *chosen* response is the
   ground-truth target when it fully parses under G (C_cfg = 1); the *rejected* response
   is a sample from the SFT model with C_cfg < 1.  Reference log-probs log π_SFT(y|x) are
   computed once here, so DPO training needs only one model in GPU memory.
2. ``DPOTrainer``: optimises the DPO loss starting from the SFT weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data import DPODataset, collate_dpo, render, tokenize_prompts_left
from ..data.datasets import encode_prompt_response, pad_right, prompt_adds_special_tokens
from ..domain import Artefacts, GrammarScorer
from ..losses import dpo_loss, sequence_logprobs
from ..models import input_device
from ..utils.common import log, read_jsonl, write_jsonl
from ..utils.profiling import FWD, FWD_BWD, Profiler
from .base import BaseTrainer


@torch.no_grad()
def _ref_logps(model, tok, cfg: ExperimentConfig, prompts, seps, responses, bs: int) -> List[float]:
    add = prompt_adds_special_tokens(cfg)
    out: List[float] = []
    dev = input_device(model)
    for i in range(0, len(prompts), bs):
        encs = [encode_prompt_response(tok, p, s + r, cfg.model.max_length, add)
                for p, s, r in zip(prompts[i:i + bs], seps[i:i + bs], responses[i:i + bs])]
        ids = pad_right([e["input_ids"] for e in encs], tok.pad_token_id).to(dev)
        lab = pad_right([e["labels"] for e in encs], -100).to(dev)
        att = pad_right([[1] * len(e["input_ids"]) for e in encs], 0).to(dev)
        s, _, _ = sequence_logprobs(model, ids, att, lab)
        out += s.float().cpu().tolist()
    return out


@torch.no_grad()
def build_preference_pairs(model, tok, cfg: ExperimentConfig, train_pairs: List[dict],
                           grammar: GrammarScorer, out_path: Path, prof: "Profiler" = None) -> List[dict]:
    prof = prof or Profiler()
    if out_path.exists():
        rows = read_jsonl(out_path)
        log(f"Reusing {len(rows)} preference pairs from {out_path}")
        return rows
    d = cfg.dpo
    pairs = train_pairs[: d.max_prompts] if d.max_prompts else train_pairs
    rendered = [render(p, tok, cfg) for p in pairs]
    chosen_ok = grammar.parses([r["target"] for r in rendered], [r["role"] for r in rendered])
    keep = [(p, r) for p, r, ok in zip(pairs, rendered, chosen_ok) if ok and r["target"]]
    log(f"DPO: {len(keep)}/{len(pairs)} targets fully parse under G (eligible as chosen)")

    model.eval()
    add = prompt_adds_special_tokens(cfg)
    dev = input_device(model)
    K = d.samples_per_prompt
    rows: List[dict] = []
    for i in range(0, len(keep), d.gen_batch_size):
        chunk = keep[i:i + d.gen_batch_size]
        enc = tokenize_prompts_left(tok, [r["prompt"] for _, r in chunk], add,
                                    cfg.model.max_length - cfg.eval.max_new_tokens).to(dev)
        with prof.section("dpo_pair_sampling"):
            seq = model.generate(**enc, do_sample=True, temperature=d.sample_temperature, top_p=d.sample_top_p,
                                 max_new_tokens=cfg.eval.max_new_tokens, num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id)
        gen = seq[:, enc["input_ids"].size(1):]
        prof.count("dpo_pair_sampling", int(enc["attention_mask"].sum()) * K
                   + int((gen != tok.pad_token_id).sum()), FWD)
        texts = [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
        roles = [r["role"] for _, r in chunk for _ in range(K)]
        with prof.section("dpo_pair_scoring"):
            scores = grammar.score(texts, roles, partial=True)
        for j, (p, r) in enumerate(chunk):
            cands = [(scores[j * K + k], texts[j * K + k]) for k in range(K)]
            bad = [(s, t) for s, t in cands if s < 1.0 and t and t.upper() != r["target"].upper()]
            if not bad:
                continue
            s, t = min(bad, key=lambda x: x[0])         # most clearly non-compliant sample
            rows.append({"pair_id": p.get("pair_id"), "prompt": r["prompt"], "sep": r["sep"],
                         "chosen": r["target"], "rejected": t, "rejected_c_cfg": s})
        if (i // d.gen_batch_size) % 50 == 0:
            log(f"  sampled {i + len(chunk)}/{len(keep)} prompts → {len(rows)} pairs")

    bs = max(1, d.gen_batch_size)
    sec = prof.section("dpo_reference_logps")
    sec.__enter__()
    rc = _ref_logps(model, tok, cfg, [r["prompt"] for r in rows], [r["sep"] for r in rows],
                    [r["chosen"] for r in rows], bs)
    rr = _ref_logps(model, tok, cfg, [r["prompt"] for r in rows], [r["sep"] for r in rows],
                    [r["rejected"] for r in rows], bs)
    sec.__exit__(None, None, None)
    prof.count("dpo_reference_logps", sum(len(tok(r["prompt"] + r["sep"] + x)["input_ids"])
                                          for r in rows for x in (r["chosen"], r["rejected"])), FWD)
    for r, a, b in zip(rows, rc, rr):
        r["ref_logp_chosen"], r["ref_logp_rejected"] = a, b
    write_jsonl(out_path, rows)
    log(f"DPO: wrote {len(rows)} preference pairs to {out_path}")
    return rows


class DPOTrainer(BaseTrainer):
    job_type = "train"

    def __init__(self, *a, pref_rows: List[dict], **kw):
        super().__init__(*a, **kw)
        self.ds = DPODataset(pref_rows, self.tok, self.cfg)

    def build_train_loader(self, epoch: int):
        g = torch.Generator().manual_seed(self.cfg.run.seed + epoch)
        return DataLoader(self.ds, batch_size=self.cfg.train.batch_size, shuffle=True, generator=g,
                          collate_fn=collate_dpo(self.tok.pad_token_id))

    def micro_step(self, batch: Dict, scale: float) -> Dict[str, float]:
        dev = input_device(self.model)
        with self.prof.section("dpo_fwd_bwd"):
            lp, _, _ = sequence_logprobs(self.model, batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                                         batch["labels"].to(dev))
            B = batch["ref_chosen"].size(0)
            loss, stats = dpo_loss(lp[:B], lp[B:], batch["ref_chosen"], batch["ref_rejected"], self.cfg.dpo.beta)
            (loss * scale).backward()
        self.prof.count("dpo_fwd_bwd", int(batch["attention_mask"].sum()), FWD_BWD)
        return {"loss_dpo": loss.item(), **{k.split("/")[1]: v for k, v in stats.items()}}
