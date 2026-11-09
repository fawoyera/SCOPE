"""Domain-adapt BERT with masked-LM on training utterances (encoder for BERTSc_D).

Run once per domain *before* the experiment grid:  ``scope bert-mlm --domain atc``.
Only the training split is used, so the test set never leaks into the evaluator.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import load_config
from ..data import load_splits
from ..utils.common import log, set_seed


def finetune_bert_mlm(domain: str, base: str, out_dir: str, epochs: int = 3, batch_size: int = 16,
                      lr: float = 2e-5, seed: int = 42) -> str:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
    if (Path(out_dir) / "config.json").exists():
        log(f"{out_dir} already exists — skipping")
        return out_dir
    set_seed(seed)
    cfg = load_config("llama31_8b", "vanilla", domain)
    splits = load_splits(cfg.data)
    texts = [t for p in splits.train for t in (p.get("req_text"), p.get("resp_text")) if t and len(t.strip()) > 3]
    log(f"BERT MLM on {len(texts)} {domain} training utterances ({base} → {out_dir})")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForMaskedLM.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()
    enc = tok(texts, truncation=True, max_length=128)
    feats = [{"input_ids": i, "attention_mask": a} for i, a in zip(enc["input_ids"], enc["attention_mask"])]
    coll = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)
    dl = torch.utils.data.DataLoader(feats, batch_size=batch_size, shuffle=True, collate_fn=coll)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        tot = 0.0
        for b in dl:
            loss = model(**{k: v.to(dev) for k, v in b.items()}).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
        log(f"  epoch {ep + 1}/{epochs} loss={tot / len(dl):.4f}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
