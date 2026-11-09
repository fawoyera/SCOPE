"""GCD† processor: every generated word must be in V (and follow a whitelisted bigram)."""
from pathlib import Path

import pytest
import torch

from scope.config import load_config
from scope.decoding import make_gcd_processor
from scope.domain import Artefacts

TINY = Path(__file__).resolve().parents[1] / "models" / "tiny-debug"


@pytest.mark.skipif(not TINY.exists(), reason="run scripts/make_tiny_model.py first")
@pytest.mark.parametrize("level", ["vocab", "bigram"])
def test_gcd_constrains_random_model(level):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY)
    art = Artefacts.load(load_config("tiny_debug", "gcd").artefacts)
    enc = tok(["### Response (UAV Pilot):"] * 3, return_tensors="pt")
    torch.manual_seed(0)
    proc = make_gcd_processor(tok, art, enc["input_ids"].size(1), level, True)
    out = model.generate(**enc, do_sample=True, max_new_tokens=25, logits_processor=LogitsProcessorList([proc]),
                         pad_token_id=tok.eos_token_id)
    succ = {}
    for a, b in art.ngrams[2]:
        succ.setdefault(a, set()).add(b)
    for row in out[:, enc["input_ids"].size(1):]:
        text = tok.decode(row, skip_special_tokens=True)
        ws = text.split()
        complete = ws if (row == tok.eos_token_id).any() else ws[:-1]   # last word may be truncated
        assert all(w in art.vocab for w in complete), text
        if level == "bigram":
            for u, v in zip(ws, ws[1:]):
                assert v in succ.get(u, art.vocab) or not succ.get(u), text
