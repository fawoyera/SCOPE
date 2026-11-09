#!/usr/bin/env python3
"""Write data/debug_subset.jsonl: the first N train / valid / test rows of the corpus
(file order), used by the tests and the CPU smoke run."""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/atc_pairs_enriched.jsonl")
ap.add_argument("--out", default="data/debug_subset.jsonl")
ap.add_argument("--train", type=int, default=96)
ap.add_argument("--valid", type=int, default=16)
ap.add_argument("--test", type=int, default=24)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data) if l.strip()]
sub = []
for split, n in (("train", a.train), ("valid", a.valid), ("test", a.test)):
    sub += [r for r in rows if r.get("split") == split][:n]
with open(a.out, "w") as f:
    f.write("\n".join(json.dumps(r) for r in sub) + "\n")
print(f"wrote {len(sub)} rows → {a.out}")
