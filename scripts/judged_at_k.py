"""Judged@k and nDCG@k-on-judged-only for saved MTEB rankings (Phase 6 TREC-COVID diagnostic).

TREC-COVID's qrels come from shallow 2020 pools: a document no 2020 system retrieved was never judged, and MTEB scores
it as non-relevant. A model that ranks different, unjudged CORD-19 papers highly is penalized for it. This reports, per
ranking file, the fraction of top-k documents that were judged at all, and nDCG@k computed both normally and with
unjudged documents removed from the ranking ("condensed list", Sakai 2007). A big gap between the two means the score
is held down by unjudged documents rather than by judged-irrelevant ones.

    python scripts/judged_at_k.py QRELS.jsonl PREDICTIONS.json [PREDICTIONS.json ...] [--k 10]
"""
import argparse
import json
import math


def rankings(obj):
    """Find the {query_id: {doc_id: score}} mapping inside an MTEB predictions file, whatever its nesting."""
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
        inner = next(iter(obj.values()))
        if inner and all(isinstance(x, (int, float)) for x in inner.values()):
            return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = rankings(v)
            if r:
                return r
    return None


def ndcg(ranked, rel, k):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qrels")
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    qrels = {}
    for line in open(a.qrels, encoding="utf-8"):
        r = json.loads(line)
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = max(int(r["score"]), 0)
    print(f"{'file':60} {'judged@'+str(a.k):>9} {'nDCG@'+str(a.k):>8} {'nDCG@'+str(a.k)+' judged-only':>20} {'rel@'+str(a.k):>7}")
    for p in a.preds:
        rk = rankings(json.load(open(p, encoding="utf-8")))
        if not rk:
            print(f"{p}: no rankings found")
            continue
        jd, nd, ndj, rl, n = 0.0, 0.0, 0.0, 0.0, 0
        for q, rel in qrels.items():
            if q not in rk:
                continue
            ranked = [d for d, _ in sorted(rk[q].items(), key=lambda x: -x[1])]
            top = ranked[: a.k]
            jd += sum(d in rel for d in top) / a.k
            rl += sum(rel.get(d, 0) > 0 for d in top) / a.k
            nd += ndcg(ranked, rel, a.k)
            ndj += ndcg([d for d in ranked if d in rel], rel, a.k)
            n += 1
        print(f"{p[-60:]:60} {jd / n:9.3f} {100 * nd / n:8.2f} {100 * ndj / n:20.2f} {rl / n:7.3f}   ({n} queries)")


if __name__ == "__main__":
    main()
