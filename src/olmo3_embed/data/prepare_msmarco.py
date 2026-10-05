"""Build the Phase 2 proxy training set: MS MARCO queries with dense-mined hard negatives.

    uv run python -m olmo3_embed.data.prepare_msmarco --n 50000

Source: sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1, config triplet-50
        (query, positive, negative_1..negative_50; negatives = coCondenser top-50 retrievals, rank order).
Output: data/msmarco_cocondenser_<n>.parquet with columns query, positive, negatives (list[str]).

False-negative control (simple version): keep only negatives from ranks NEG_LO..NEG_HI. The very top ranks of a
dense retriever often hold unlabeled relevant passages (MS MARCO labels ~1 positive per query). Positive-aware
filtering with a cross-encoder (NV-Retriever) replaces this in Phase 3.

Track note: MS MARCO is a BEIR training split, so this is Track B data, and NanoMSMARCO is in-domain for any model
trained on it. Phase 2 comparisons therefore report NanoBEIR avg both with and without MSMARCO.
"""
import argparse
import os

import pandas as pd
from datasets import load_dataset

SOURCE = "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1"
NEG_LO, NEG_HI = 10, 50  # inclusive ranks kept as hard-negative candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    ds = load_dataset(SOURCE, "triplet-50", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=50_000).take(args.n)
    rows = []
    for r in ds:
        negs = [r[f"negative_{i}"] for i in range(NEG_LO, NEG_HI + 1)]
        negs = [n for n in negs if n and n != r["positive"]]
        rows.append({"query": r["query"], "positive": r["positive"], "negatives": negs})
        if len(rows) % 10_000 == 0:
            print(f"{len(rows)} rows", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"msmarco_cocondenser_{args.n // 1000}k.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"wrote {path}: {len(rows)} rows, {sum(len(r['negatives']) for r in rows) / len(rows):.1f} negatives/row")


if __name__ == "__main__":
    main()
