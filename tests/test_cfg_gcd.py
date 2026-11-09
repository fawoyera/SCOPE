"""GCD after Geng et al. (2023): Lark→EBNF conversion and grammar-constrained generation."""
import random
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers_cfg")

from scope.config import load_config, resolve_path  # noqa: E402
from scope.decoding.cfg_gcd import lark_to_ebnf, regex_to_ebnf  # noqa: E402
from scope.domain.grammar import ROLE_START, strict_grammar_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TINY, DATA = ROOT / "models" / "tiny-debug", ROOT / "data" / "debug_subset.jsonl"
GRAMMAR = ROOT / "artefacts" / "atc" / "grammar.lark"
needs_tiny = pytest.mark.skipif(not (TINY.exists() and DATA.exists()), reason="needs tiny model + debug data")


def test_regex_conversion():
    assert regex_to_ebnf("[A-Z]{2,3}") == "[A-Z] [A-Z] ([A-Z])?"
    assert regex_to_ebnf(r"1[0-9]{2}\.[0-9]") == '"1" [0-9] [0-9] "." [0-9]'
    assert regex_to_ebnf("[ ]O'CLOCK").startswith("[ ]")
    assert regex_to_ebnf("(?:AB|C)+") in ('("A" "B" | "C")+', '(("A" "B" | "C"))+')


def test_no_left_recursion_and_all_rules_defined():
    src = strict_grammar_source(GRAMMAR.read_text())
    for role in ("controller", "pilot", "unified"):
        ebnf = lark_to_ebnf(src, ROLE_START.get(role, "start"))
        defined = {l.split(" ::= ")[0] for l in ebnf.splitlines()}
        assert "general-instr" not in defined and "pilot-general" not in defined   # strict grammar


@needs_tiny
def test_ebnf_accepts_exactly_what_lark_accepts():
    from lark import Lark
    from transformers import AutoTokenizer
    from scope.data import load_splits, select_target
    from scope.decoding.cfg_gcd import CFGDecoder
    tok = AutoTokenizer.from_pretrained(TINY)
    dec = CFGDecoder(str(GRAMMAR), tok)
    src = strict_grammar_source(GRAMMAR.read_text())
    cfg = load_config("tiny_debug", "gcd_cfg", overrides=[f"data.path={DATA}"])
    sp = load_splits(cfg.data)
    random.seed(0)
    for role in ("controller", "pilot"):
        lark = Lark(src, parser="earley", start=ROLE_START[role], ambiguity="resolve")
        for p in sp.train + sp.val + sp.test:
            if p["resp_role"] != role:
                continue
            t = " ".join(select_target(p).upper().split())
            w = t.split()
            cases = [t] + ([" ".join(w[:-1]), " ".join(w[1:])] if len(w) > 1 else [])
            for s in cases:
                try:
                    lark.parse(s)
                    ok = True
                except Exception:
                    ok = False
                assert dec.accepts(s, role) == ok, (role, s)


@needs_tiny
@pytest.mark.parametrize("mode", ["greedy", "beam", "sample"])
def test_constrained_outputs_are_grammatical(mode):
    from lark import Lark
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    from scope.decoding.cfg_gcd import get_cfg_decoder
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY).eval()
    dec = get_cfg_decoder(str(GRAMMAR), tok)
    src = strict_grammar_source(GRAMMAR.read_text())
    torch.manual_seed(0)
    for role in ("controller", "pilot"):
        lark = Lark(src, parser="earley", start=ROLE_START[role], ambiguity="resolve")
        tok.padding_side = "left"
        enc = tok(["### Response (ATC):", "### Input:\nDELTA SIX\n\n### Response (Pilot):"], return_tensors="pt",
                  padding=True)
        P = enc["input_ids"].size(1)
        kw = dict(max_new_tokens=40, pad_token_id=tok.eos_token_id,
                  logits_processor=LogitsProcessorList([dec.processor(role, P)]))
        if mode == "beam":
            kw.update(num_beams=3, do_sample=False)
        elif mode == "sample":
            kw.update(do_sample=True, temperature=1.5)
        else:
            kw.update(do_sample=False)
        out = model.generate(**enc, **kw)
        for row in out[:, P:]:
            text = " ".join(tok.decode(row, skip_special_tokens=True).split())
            finished = (row == tok.eos_token_id).any().item()
            rec = dec.recognizers[role].string_recognizer
            assert rec._accept_prefix(text) or rec._accept_prefix(" " + text), text   # always a viable prefix
            if finished and text:
                lark.parse(text)                                                    # complete ⇒ in L(G)
