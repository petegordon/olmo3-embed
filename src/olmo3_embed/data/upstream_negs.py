"""B4 data (docs/phase3-plan.md §6): the B1 rows with their *upstream* negatives instead of the rescored ones.

    python -m olmo3_embed.data.upstream_negs --rescored rescored.parquet --decontam decontam.parquet --out out.parquet

Same qids, queries and positives as rescored.parquet. Negatives = the source's own candidates (RLHN / bge order), first
--max-cands, without the Qwen3 rescore or the 95% guard. SciFact/NFCorpus have no upstream negatives (rescore.py mined
them), so they keep the mined list: B4 vs B1 then isolates "rescore + guard" on the other six tasks.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescored", required=True)
    ap.add_argument("--decontam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cands", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_parquet(args.rescored, columns=["qid", "task", "query", "positives", "negatives"])
    up = pd.read_parquet(args.decontam, columns=["qid", "negatives"]).set_index("qid")["negatives"]
    upstream = df.qid.map(up)
    keep_mined = upstream.map(lambda n: not hasattr(n, "__len__") or len(n) == 0)  # missing (NaN) or empty
    df["negatives"] = [list(r)[: args.max_cands] if not m else list(n)
                       for r, n, m in zip(upstream, df.negatives, keep_mined)]
    df.to_parquet(args.out, index=False, row_group_size=20_000)
    print(f"{len(df)} rows; upstream negatives for {int((~keep_mined).sum())}, mined kept for {int(keep_mined.sum())}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
