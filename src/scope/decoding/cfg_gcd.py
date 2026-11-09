"""Grammar-Constrained Decoding (GCD) after Geng et al., EMNLP 2023
("Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning").

Their recipe, reproduced here:
  * the output space is a formal grammar; at every decoding step an *incremental parser*
    returns the set of next tokens that keep the prefix inside the grammar's prefix
    language, and every other token's logit is set to −∞;
  * EOS is allowed only when the prefix is a complete sentence of the grammar;
  * grammars may be *input-dependent* (they pick a grammar per data point) — here the
    start symbol is chosen per example from the response role (controller/pilot),
    exactly the role gating used by C_cfg;
  * decoding is greedy or beam search (``gcd.num_beams``).

The original code queries a Grammatical Framework (GF) server; the same authors'
HF-native implementation is ``transformers-cfg`` (EBNF grammars, incremental pushdown
recogniser over a byte-level token trie), which we use as the parsing backend.  The
Lark grammar G_ATC is converted mechanically to that EBNF dialect — same productions,
with Lark's left-recursive list rules rewritten as EBNF repetition (a pushdown
recogniser cannot handle left recursion) and alternatives left-factored for speed.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

from ..domain.grammar import ROLE_START, strict_grammar_source

# ════════════════════════════════════════════════════════════════════════════
# Lark → EBNF (transformers-cfg / GBNF dialect)
# ════════════════════════════════════════════════════════════════════════════


def _name(n: str) -> str:
    s = re.sub(r"[^A-Za-z0-9-]", "-", n).strip("-") or "r"
    return s if s[0].isalpha() else "r-" + s


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def regex_to_ebnf(rx: str) -> str:
    """Convert the regex subset used in Lark terminals to EBNF: char classes, escapes,
    literals, (?:...) / (...) groups with |, and quantifiers ? * + {m} {m,} {m,n}."""
    pos = 0

    def parse_alt() -> str:
        nonlocal pos
        alts = [parse_seq()]
        while pos < len(rx) and rx[pos] == "|":
            pos += 1
            alts.append(parse_seq())
        return alts[0] if len(alts) == 1 else "(" + " | ".join(alts) + ")"

    def parse_seq() -> str:
        nonlocal pos
        items = []
        while pos < len(rx) and rx[pos] not in "|)":
            atom = parse_atom()
            items.append(parse_quant(atom))
        return " ".join(items) if items else '""'

    def parse_atom() -> str:
        nonlocal pos
        c = rx[pos]
        if c == "[":
            end = pos + 1
            if end < len(rx) and rx[end] == "^":
                end += 1
            if end < len(rx) and rx[end] == "]":
                end += 1
            while rx[end] != "]":
                end += 2 if rx[end] == "\\" else 1
            cls = rx[pos:end + 1]
            pos = end + 1
            return cls
        if c == "(":
            pos += 1
            if rx.startswith("?:", pos):
                pos += 2
            inner = parse_alt()
            assert rx[pos] == ")", rx
            pos += 1
            return "(" + inner + ")"
        if c == "\\":
            e = rx[pos + 1]
            pos += 2
            table = {"d": "[0-9]", "s": "[ \\t\\n]", "w": "[A-Za-z0-9_]", "n": '"\\n"', "t": '"\\t"'}
            return table.get(e, _lit(e))
        if c == ".":
            pos += 1
            return "[^\\n]"
        pos += 1
        return _lit(c)

    def parse_quant(atom: str) -> str:
        nonlocal pos
        if pos >= len(rx):
            return atom
        c = rx[pos]
        if c in "?*+":
            pos += 1
            return atom + c
        if c == "{":
            end = rx.index("}", pos)
            spec = rx[pos + 1:end]
            pos = end + 1
            lo, _, hi = spec.partition(",")
            lo = int(lo or 0)
            if "," not in spec:
                hi_n = lo
            elif hi == "":
                return " ".join([atom] * lo + [atom + "*"])
            else:
                hi_n = int(hi)
            req = [atom] * lo
            opt = ""
            for _ in range(hi_n - lo):          # nested optionals: unambiguous, no stack blow-up
                opt = f"({atom} {opt})?" if opt else f"({atom})?"
            return " ".join(req + ([opt] if opt else []))
        return atom

    out = parse_alt()
    if pos != len(rx):
        raise ValueError(f"Unsupported regex {rx!r} (stopped at {pos})")
    return out


def left_factor(alts: List[List[str]]) -> str:
    """Render alternatives with shared prefixes factored out: [[a,b],[a,b,c],[a,d],[e]] →
    (a (b (c)? | d) | e).  Language-preserving; it removes the redundant parallel parser
    stacks that Lark's expansion of optionals into separate alternatives would create
    (≈400× faster masking on the pilot grammar)."""
    END = object()
    trie: dict = {}
    for a in alts:
        node = trie
        for sym in a:
            node = node.setdefault(sym, {})
        node[END] = True

    def render(node):
        kids = [(s, n) for s, n in node.items() if s is not END]
        if not kids:
            return None
        parts = []
        for sym, child in kids:
            seq = [sym]
            while END not in child and len(child) == 1:      # follow unary chains
                (s2, c2), = child.items()
                seq.append(s2)
                child = c2
            rest = render(child)
            parts.append(" ".join(seq) + (f" {rest}" if rest else ""))
        body = parts[0] if len(parts) == 1 else "(" + " | ".join(parts) + ")"
        return f"({body})?" if END in node else body

    out = render(trie)
    return out if out is not None else '""'


def lark_to_ebnf(lark_src: str, start: str, leading_space: bool = True) -> str:
    """Compile a Lark grammar to EBNF rooted at ``start``.

    Lark is used to *parse* the grammar (so EBNF sugar, templates and anonymous
    terminals are resolved exactly as Lark resolves them); its BNF rules are then
    printed, eliminating direct left recursion  A → A β | α   ⇒   A ::= α β*.
    """
    from lark import Lark
    from lark.grammar import NonTerminal, Terminal
    from lark.lexer import PatternRE, PatternStr

    parser = Lark(lark_src, parser="earley", start=start, keep_all_tokens=True)
    term_ebnf: Dict[str, str] = {}
    for t in parser.terminals:
        p = t.pattern
        if isinstance(p, PatternStr):
            term_ebnf[t.name] = _lit(p.value)
        elif isinstance(p, PatternRE):
            if p.flags:
                raise ValueError(f"Regex flags unsupported: {t.name} {p.flags}")
            term_ebnf[t.name] = regex_to_ebnf(p.value)
    ignore = set(getattr(parser, "ignore_tokens", []) or [])

    alts: Dict[str, List[List[str]]] = defaultdict(list)
    order: List[str] = []
    for r in parser.rules:
        o = r.origin.name
        if o not in alts:
            order.append(o)
        seq = []
        for s in r.expansion:
            if isinstance(s, Terminal):
                if s.name in ignore:
                    continue
                seq.append(term_ebnf[s.name])
            else:
                seq.append(_name(s.name))
        alts[o].append(seq)

    lines = []
    for o in order:
        me = _name(o)
        rec = [a[1:] for a in alts[o] if a and a[0] == me]
        base = [a for a in alts[o] if not (a and a[0] == me)]
        if rec:
            if not base:
                raise ValueError(f"Rule {o} is only left-recursive")
            body = f"({left_factor(base)}) ({left_factor(rec)})*"
        else:
            body = left_factor(base)
        lines.append(f"{me} ::= {body}")
    space = '" "? ' if leading_space else ""
    root = f"root ::= {space}{_name(start)}"
    return "\n".join([root] + lines) + "\n"


def check_no_left_recursion(ebnf: str) -> None:
    """Fail fast on indirect left recursion (unsupported by a pushdown recogniser)."""
    rules = {}
    for line in ebnf.strip().splitlines():
        k, v = line.split(" ::= ", 1)
        rules[k] = v
    first = {}
    for k, v in rules.items():
        first[k] = {re.match(r"\(?\s*([A-Za-z][A-Za-z0-9-]*)", a.strip()).group(1)
                    for a in v.split(" | ") if re.match(r"\(?\s*[A-Za-z]", a.strip())}
        first[k] &= set(rules)

    def reach(a, seen):
        for b in first.get(a, ()):
            if b == start_rule:
                return True
            if b not in seen:
                seen.add(b)
                if reach(b, seen):
                    return True
        return False

    for start_rule in rules:
        if reach(start_rule, set()):
            raise ValueError(f"Left recursion through rule {start_rule}")


# ════════════════════════════════════════════════════════════════════════════
# Decoding
# ════════════════════════════════════════════════════════════════════════════
class _SurfaceBytes:
    """Model-agnostic token → bytes map (works for byte-level BPE and SentencePiece,
    incl. Llama-3/3.1, Qwen, Gemma, Mistral, GPT-2) built from mid-text surface strings.
    Replaces transformers-cfg's per-tokenizer-class mappings."""

    def __init__(self, tokenizer):
        from .gcd import _surfaces_cached
        self._len = len(tokenizer)
        special = set(tokenizer.all_special_ids)
        self.bytes = [b"\xff" if i in special else s.encode("utf-8")
                      for i, s in enumerate(_surfaces_cached(tokenizer))]

    def __len__(self):
        return self._len

    def map(self, token_id: int, verbose: bool = False) -> bytes:
        return self.bytes[int(token_id)]


def _byte_trie(mapping: _SurfaceBytes, special: set):
    from transformers_cfg.tokenization.byte_trie import ByteTrie
    trie = ByteTrie()
    for i, b in enumerate(mapping.bytes):
        if i not in special and b and b != b"\xff":
            trie.insert(b, i)
    trie.vocab_size = len(mapping)
    return trie


class PrefixCFGProcessor:
    """HF LogitsProcessor equivalent to Geng et al.'s ``prefix_allowed_tokens_fn``:
    the allowed set is a function of each row's *generated prefix* (so it is correct under
    beam search, where rows are re-ordered every step).  Parse states are cached by prefix."""

    def __init__(self, recognizer, prompt_len: int, stop_ids):
        self.rec = recognizer
        self.prompt_len = prompt_len
        self.stop = set(stop_ids)
        self.cache = {(): recognizer.string_recognizer.get_initial_parsing_state()}
        self.mask_cache = {}

    def _state(self, key):
        if key in self.cache:
            return self.cache[key]
        parent = self._state(key[:-1])
        st = self.rec._update_state_with_token_id(key[-1], parent) if parent.stacks else parent
        self.cache[key] = st
        return st

    def __call__(self, input_ids, scores):
        import torch
        for b in range(input_ids.size(0)):
            key = tuple(input_ids[b, self.prompt_len:].tolist())
            if key and key[-1] in self.stop:          # finished row (EOS/pad): leave untouched
                continue
            if key not in self.mask_cache:
                st = self._state(key)
                acc = self.rec.filter_vocab(st, scores.device)
                acc = acc[: scores.size(-1)]
                if acc.size(0) < scores.size(-1):
                    acc = torch.cat([acc, acc.new_zeros(scores.size(-1) - acc.size(0))])
                if not acc.any():                        # dead end: allow termination
                    acc = acc.clone()
                    acc[list(self.stop & set(range(scores.size(-1))))] = True
                self.mask_cache[key] = acc
            scores[b] = scores[b].masked_fill(~self.mask_cache[key], float("-inf"))
        return scores


class CFGDecoder:
    """One incremental recogniser per role (input-dependent grammar, cf. Geng et al.'s
    per-datapoint grammars), sharing a single token byte-trie."""

    def __init__(self, grammar_path: str, tokenizer):
        try:
            from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer
        except ImportError as e:
            raise ImportError("GCD (level=cfg) needs `pip install transformers-cfg`") from e
        src = strict_grammar_source(open(grammar_path).read())
        self.tokenizer = tokenizer
        mapping = _SurfaceBytes(tokenizer)
        trie = _byte_trie(mapping, set(tokenizer.all_special_ids))
        self.ebnf: Dict[str, str] = {}
        self.recognizers = {}
        for role in ("controller", "pilot", "unified"):
            ebnf = lark_to_ebnf(src, ROLE_START.get(role, "start"))
            check_no_left_recursion(ebnf)
            self.ebnf[role] = ebnf
            self.recognizers[role] = IncrementalTokenRecognizer(ebnf, "root", tokenizer, trie=trie,
                                                                homomorphism=mapping)

    def processor(self, role: str, prompt_len: int):
        rec = self.recognizers[role if role in ROLE_START else "unified"]
        stop = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id} - {None}
        return PrefixCFGProcessor(rec, prompt_len, stop)

    def accepts(self, text: str, role: str) -> bool:
        rec = self.recognizers[role if role in ROLE_START else "unified"]
        return bool(rec.string_recognizer._accept_string(text))


_DECODERS: Dict[Tuple[str, int], CFGDecoder] = {}


def get_cfg_decoder(grammar_path: str, tokenizer) -> CFGDecoder:
    key = (grammar_path, id(tokenizer))
    if key not in _DECODERS:
        _DECODERS[key] = CFGDecoder(grammar_path, tokenizer)
    return _DECODERS[key]
