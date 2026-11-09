#!/usr/bin/env python3
"""Build a tiny randomly-initialised GPT-2 (+ byte-level BPE tokenizer trained on the
corpus) under models/tiny-debug.  Used by `--model tiny_debug` to exercise the whole
pipeline on a laptop/CPU in minutes; its scores are meaningless."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/atc_pairs_enriched.jsonl")
ap.add_argument("--out", default="models/tiny-debug")
ap.add_argument("--vocab-size", type=int, default=2000)
a = ap.parse_args()

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

texts = []
for line in open(a.data):
    r = json.loads(line)
    texts += [r.get("req_text") or "", r.get("resp_text") or "", r.get("judge_corrected") or ""]
import scope.data.prompts as P
texts += list(P.SYSTEM_PROMPTS.values()) * 50 + list(P.PREAMBLE.values()) * 50

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
bpe = ByteLevelBPETokenizer()
bpe.train_from_iterator(texts, vocab_size=a.vocab_size, special_tokens=["<|endoftext|>"])
tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<|endoftext|>",
                              bos_token="<|endoftext|>", unk_token="<|endoftext|>")
tok.save_pretrained(out)
cfg = GPT2Config(vocab_size=len(tok), n_positions=1024, n_embd=64, n_layer=2, n_head=2,
                 bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
GPT2LMHeadModel(cfg).save_pretrained(out)
print(f"tiny model + tokenizer ({len(tok)} tokens) → {out}")
