"""Ground-truth consistency metrics: Slot-F1, dialogue-act F1 and hallucination.

Ported unchanged from the original training script so numbers stay comparable.
All three operate on text normalised by ``text_to_digits``.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

ATC_SLOT_PATTERNS = {
    "callsign": re.compile(
        r"\b([A-Z]{2,}(?:\s+[A-Z0-9]+){1,6})"
        r"(?=\s+(?:TURN|DESCEND|CLIMB|MAINTAIN|CONTACT|CLEARED|SQUAWK|RADAR|"
        r"TRAFFIC|SURFACE|REPORT|CONFIRM|SAY|RESUME|HOLD|EXPECT|CROSS|LINE|"
        r"VACATE|TAXI|CANCEL|TAKEOFF|GO|CONTINUE|IDENT|STOP|DIRECT|FLY|"
        r"HEADING|FREQUENCY|INFORMATION|ROGER|WILCO|AFFIRM|NEGATIVE|UNABLE|"
        r"STANDBY|CORRECTION|GOOD|THANK|READBACK))"),
    "altitude": re.compile(r"\b(\d{1,2}\s*(?:THOUSAND|HUNDRED)|FLIGHT\s+LEVEL\s+\d+)\b"),
    "heading": re.compile(r"\bHEADING\s+(\d+)\b"),
    "frequency": re.compile(r"\b(\d{3}[\s\.]+\d+)\b"),
    "squawk": re.compile(r"\bSQUAWK\s+(\d+)\b"),
    "runway": re.compile(r"\bRUNWAY\s+(\d+(?:\s+(?:LEFT|RIGHT|CENTER))?)\b"),
}

DA_PATTERNS = [
    ("clearance", re.compile(r"\b(CLEARED|CLEARANCE|LINE\s+UP|TAKEOFF|TAKE-OFF)\b")),
    ("readback", re.compile(r"\b(WILCO|AFFIRM|ROGER|READBACK\s+CORRECT)\b")),
    ("hold", re.compile(r"\b(HOLD|STANDBY|UNABLE)\b")),
    ("advisory", re.compile(r"\b(TRAFFIC|SURFACE\s+WIND|INFORMATION)\b")),
    ("handoff", re.compile(r"\b(CONTACT|FREQUENCY\s+CHANGE|GOOD\s+DAY)\b")),
    ("correction", re.compile(r"\b(CORRECTION|SAY\s+AGAIN)\b")),
]

ENTITY_PATTERNS = [
    re.compile(r"\b[A-Z]{2,3}\s+\d+\b"),
    re.compile(r"\b(?:FL|FLIGHT\s+LEVEL)\s+\d+\b"),
    re.compile(r"\b\d{3}[\s\.]+\d+\b"),
    re.compile(r"\b\d{4}\b"),
    re.compile(r"\b\d{1,2}(?:\s+(?:LEFT|RIGHT|CENTER))?\b"),
]

_NUM_MAP = {
    "ZERO": "0", "ONE": "1", "WUN": "1", "TWO": "2", "TOO": "2", "THREE": "3", "TREE": "3",
    "FOUR": "4", "FOWER": "4", "FIVE": "5", "FIFE": "5", "SIX": "6", "SEVEN": "7",
    "EIGHT": "8", "NINE": "9", "NINER": "9", "TEN": "10", "POINT": ".", "DECIMAL": ".",
    "HUNDRED": "00", "THOUSAND": "000", "R": "RIGHT", "L": "LEFT", "C": "CENTER",
}


def text_to_digits(text: str) -> str:
    if not text:
        return ""
    t = text.upper().replace("-", " ").replace(".", " POINT ")
    t = re.sub(r"(\d+)([LRC])\b", r"\1 \2", t)
    combined = " ".join(_NUM_MAP.get(w, w) for w in t.split())
    combined = re.sub(r"(?<=\d)\s+(?=\d)|(?<=\d)\s+(?=\.)|(?<=\.)\s+(?=\d)", "", combined)
    return " ".join(str(int(tok)) if tok.isdigit() else tok for tok in combined.split())


def extract_slots(text: str) -> Dict[str, Optional[str]]:
    t = text.upper()
    out = {}
    for name, pat in ATC_SLOT_PATTERNS.items():
        m = pat.search(t)
        out[name] = m.group(1).strip() if m else None
    return out


def predict_da(text: str) -> str:
    t = text.upper()
    for label, pat in DA_PATTERNS:
        if pat.search(t):
            return label
    return "other"


def slot_f1(ref: Dict[str, Optional[str]], gen: Dict[str, Optional[str]]) -> float:
    tp = fp = fn = 0
    for k in set(ref) | set(gen):
        r = ref.get(k).split() if ref.get(k) else []
        g = gen.get(k).split() if gen.get(k) else []
        if (not r and not g) or r == g:
            tp += 1
        elif r and not g:
            fn += 1
        elif g and not r:
            fp += 1
        else:
            fp += 1
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * rc / (p + rc) if p + rc else 0.0


def hallucination(gen: str, ref: str, req: str) -> float:
    """1 if the generation contains an entity found in neither the reference nor the request."""
    context = f"{ref} {req}".upper()
    g = gen.upper()
    for pat in ENTITY_PATTERNS:
        for m in pat.finditer(g):
            if m.group(0).strip() not in context:
                return 1.0
    return 0.0


def semantic_scores(generated: str, reference: str, request: str) -> Dict[str, float]:
    g, r, q = (text_to_digits(x).upper() for x in (generated, reference, request))
    return {
        "slot_f1": slot_f1(extract_slots(r), extract_slots(g)),
        "da_f1": 1.0 if predict_da(r) == predict_da(g) else 0.0,
        "halluc": hallucination(g, r, q),
    }
