"""Splits, data curricula, tokenised datasets and collators."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..config import DataConfig, ExperimentConfig, resolve_path
from ..utils.common import read_jsonl
from .prompts import build_prompt


# ════════════════════════════════════════════════════════════════════════════
# Splits and curricula
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Splits:
    train: List[dict]
    val: List[dict]
    test: List[dict]

    def describe(self) -> str:
        tiers = Counter(p.get("quality_tier", "bronze") for p in self.train)
        return (f"train={len(self.train)} (gold={tiers['gold']}, silver={tiers['silver']}, "
                f"bronze={tiers['bronze']}) | val={len(self.val)} | test={len(self.test)}")


def load_splits(cfg: DataConfig) -> Splits:
    """Read the JSONL and partition on its ``split`` field.

    File order is preserved (no shuffling), so a capped evaluation set
    (``val_max_pairs`` / ``test_max_pairs``) contains the *same* examples for
    every method and seed — a prerequisite for the paired statistical tests.
    """
    rows = read_jsonl(resolve_path(cfg.path))
    if cfg.drop_unintelligible:
        rows = [r for r in rows if not r.get("resp_unintelligible")]
    get = lambda s: [r for r in rows if r.get(cfg.split_field) == s]
    return Splits(get(cfg.train_split), get(cfg.val_split), get(cfg.test_split))


def curriculum_pairs(train: List[dict], cfg: DataConfig, epoch: int, total_epochs: int) -> List[dict]:
    """A: gold+silver in phase 1, all later | B: all (tier-weighted CE) | C: gold+silver only |
    none: all pairs, unweighted."""
    if cfg.curriculum not in ("A", "B", "C", "none"):
        raise ValueError(f"data.curriculum must be A, B, C or none, got {cfg.curriculum!r}")
    gs = [p for p in train if p.get("quality_tier", "bronze") in ("gold", "silver")]
    if cfg.curriculum == "C":
        return gs
    if cfg.curriculum == "A" and epoch < int(cfg.phase1_frac * total_epochs):
        return gs
    return train


def tier_weight(pair: dict, cfg: DataConfig) -> float:
    if cfg.curriculum != "B":
        return 1.0
    return float(cfg.tier_weights.get(pair.get("quality_tier", "bronze"), 1.0))


# ════════════════════════════════════════════════════════════════════════════
# Tokenisation helpers
# ════════════════════════════════════════════════════════════════════════════
def prompt_adds_special_tokens(cfg: ExperimentConfig) -> bool:
    # A rendered chat template already contains BOS/role tokens.
    return cfg.model.prompt_format != "chat"


_DEMOS: Dict[tuple, Dict[str, List[dict]]] = {}


_STOP = {"THE", "AND", "TO", "AT", "FOR", "OF", "IS", "ON", "A", "ROGER", "WILCO", "AFFIRM", "THANK", "YOU", "AH", "UH"}


def _content(text: str) -> set:
    from ..metrics.semantic import text_to_digits
    return {w for w in text_to_digits(text).upper().split() if w not in _STOP}


def demonstrations(cfg: ExperimentConfig) -> Dict[str, List[dict]]:
    """Fixed few-shot demonstrations per response role, drawn from the training split only and
    identical for every test example, method and seed.  Candidates are gold pairs whose target
    (i) fully parses under G, (ii) is unique among gold targets (duplicated expert targets are
    annotation artefacts), (iii) has ≥ 3 words and (iv) shares ≥ 2 content tokens with its request
    (digits normalised), so the demonstration is a genuine response.  Distinct exchange types are
    preferred; ties broken by file order."""
    d = cfg.data
    key = (resolve_path(d.path), d.fewshot_k, cfg.artefacts.grammar)
    if key not in _DEMOS:
        from collections import Counter
        from lark import Lark
        from ..domain.grammar import ROLE_START, strict_grammar_source
        from .prompts import select_target
        src = strict_grammar_source(open(resolve_path(cfg.artefacts.grammar)).read())
        train = load_splits(d).train
        gold = [p for p in train if p.get("quality_tier") == "gold" and p.get("req_text")]
        dup = Counter(select_target(p) for p in gold)
        out: Dict[str, List[dict]] = {}
        for role in ("controller", "pilot"):
            parser = Lark(src, parser="earley", start=ROLE_START[role], ambiguity="resolve")
            cands = []
            for p in gold:
                t = select_target(p)
                if p.get("resp_role") != role or dup[t] > 1 or len(t.split()) < 3:
                    continue
                if len(_content(t) & _content(p["req_text"])) < 2:
                    continue
                try:
                    parser.parse(" ".join(t.upper().split()))
                except Exception:
                    continue
                cands.append(p)
            picked, seen = [], set()
            for p in cands:                      # first pass: one per exchange type
                if len(picked) < d.fewshot_k and p.get("exchange_type") not in seen:
                    picked.append(p)
                    seen.add(p.get("exchange_type"))
            for p in cands:                      # fill up
                if len(picked) < d.fewshot_k and p not in picked:
                    picked.append(p)
            out[role] = picked
        _DEMOS[key] = out
    return _DEMOS[key]


def render(pair: dict, tokenizer, cfg: ExperimentConfig) -> Dict[str, str]:
    m, d = cfg.model, cfg.data
    demos = None
    if d.prompt_style == "fewshot":
        demos = demonstrations(cfg).get(pair.get("resp_role"), [])
        demos = [x for x in demos if x.get("pair_id") != pair.get("pair_id")]   # never show the item itself
    return build_prompt(pair, tokenizer, m.prompt_format, cfg.artefacts.system_prompt,
                        m.chat_template_kwargs, m.merge_system_into_user, d.prompt_style, demos)


def encode_prompt_response(tokenizer, prompt: str, response: str, max_length: int,
                           add_special_tokens: bool) -> Dict[str, List[int]]:
    """prompt ids + response ids + EOS; labels mask the prompt.  Prompt is left-truncated
    if needed so the response (and EOS) are never cut."""
    p_ids = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    r_ids = r_ids[: max_length // 2]
    budget = max_length - len(r_ids)
    if len(p_ids) > budget:
        p_ids = p_ids[len(p_ids) - budget:]
    ids = p_ids + r_ids
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + r_ids, "prompt_len": len(p_ids)}


class SFTDataset(Dataset):
    """Prompt/target pairs for CE-based objectives (SFT and SCOPE)."""

    def __init__(self, pairs: Sequence[dict], tokenizer, cfg: ExperimentConfig):
        self.items = []
        add = prompt_adds_special_tokens(cfg)
        for p in pairs:
            r = render(p, tokenizer, cfg)
            enc = encode_prompt_response(tokenizer, r["prompt"], r["sep"] + r["target"],
                                         cfg.model.max_length, add)
            self.items.append({**enc, "prompt": r["prompt"], "target": r["target"],
                               "role": r["role"], "request": r["request"],
                               "weight": tier_weight(p, cfg.data)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class DPODataset(Dataset):
    """Rows from ``dpo_pairs.jsonl`` (prompt, chosen, rejected, ref log-probs)."""

    def __init__(self, rows: Sequence[dict], tokenizer, cfg: ExperimentConfig):
        self.items = []
        add = prompt_adds_special_tokens(cfg)
        for r in rows:
            c = encode_prompt_response(tokenizer, r["prompt"], r["sep"] + r["chosen"], cfg.model.max_length, add)
            j = encode_prompt_response(tokenizer, r["prompt"], r["sep"] + r["rejected"], cfg.model.max_length, add)
            self.items.append({"chosen": c, "rejected": j,
                               "ref_chosen": r["ref_logp_chosen"], "ref_rejected": r["ref_logp_rejected"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def pad_right(seqs: List[List[int]], value: int) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs], dtype=torch.long)


def collate_sft(pad_id: int):
    def fn(batch: List[dict]) -> Dict:
        ids = pad_right([b["input_ids"] for b in batch], pad_id)
        labels = pad_right([b["labels"] for b in batch], -100)
        attn = pad_right([[1] * len(b["input_ids"]) for b in batch], 0)
        return {"input_ids": ids, "attention_mask": attn, "labels": labels,
                "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
                "prompt": [b["prompt"] for b in batch], "target": [b["target"] for b in batch],
                "role": [b["role"] for b in batch], "request": [b["request"] for b in batch]}
    return fn


def collate_dpo(pad_id: int):
    def fn(batch: List[dict]) -> Dict:
        seqs = [b["chosen"] for b in batch] + [b["rejected"] for b in batch]
        return {"input_ids": pad_right([s["input_ids"] for s in seqs], pad_id),
                "labels": pad_right([s["labels"] for s in seqs], -100),
                "attention_mask": pad_right([[1] * len(s["input_ids"]) for s in seqs], 0),
                "ref_chosen": torch.tensor([b["ref_chosen"] for b in batch], dtype=torch.float32),
                "ref_rejected": torch.tensor([b["ref_rejected"] for b in batch], dtype=torch.float32)}
    return fn


def tokenize_prompts_left(tokenizer, prompts: List[str], add_special_tokens: bool,
                          max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Left-padded prompt batch for generation."""
    old = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens,
                        truncation=max_length is not None, max_length=max_length)
    finally:
        tokenizer.padding_side = old
    return enc
