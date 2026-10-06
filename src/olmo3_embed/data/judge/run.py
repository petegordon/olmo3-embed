"""Select judge candidates, run a judge backend over them, and report agreement (docs/phase3-plan.md §4b).

  select   python -m olmo3_embed.data.judge.run select --rescored <rescored.parquet> --out pairs.jsonl
                [--pilot 500]        500 Quora queries, every band candidate (the 4-judge pilot)
  judge    python -m olmo3_embed.data.judge.run judge --pairs pairs.jsonl --backend vllm|claude|codex|gemini
                [--model M] [--only-flagged verdicts_by_other_judge.jsonl] --out verdicts_<backend>.jsonl
  report   python -m olmo3_embed.data.judge.run report verdicts_*.jsonl

Pairs: candidates whose rescore score is in [band, guard] x the positive's score (the band rescore.py counts),
top --per-query per query, for the tasks nobody has LLM-judged (Quora, SciFact, NFCorpus; RLHN rows were judged
upstream). Verdict lines record judge, model, prompt hash and the raw response, so every label is traceable.
Resumable: judge skips pair ids already in --out.
"""
import argparse
import collections
import itertools
import json
import os
import random

import pandas as pd

from olmo3_embed.data.judge.backends import BACKENDS
from olmo3_embed.data.judge.prompt import PROMPT_SHA, build, parse

TASKS = ("QuoraRetrieval", "SciFact", "NFCorpus")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(args):
    df = pd.read_parquet(args.rescored, columns=["qid", "task", "query", "positives", "pos_score", "negatives",
                                                 "neg_scores"])
    df = df[df.task.isin(TASKS)]
    if args.pilot:
        q = df[df.task == "QuoraRetrieval"]
        df = q.sample(n=min(args.pilot, len(q)), random_state=args.seed)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            band = [(c, s) for c, s in zip(r.negatives, r.neg_scores) if s >= args.band * r.pos_score]
            for rank, (c, s) in enumerate(band if args.pilot else band[: args.per_query]):
                f.write(json.dumps({"id": f"{r.qid}#{rank}", "qid": r.qid, "task": r.task, "query": r.query,
                                    "positive": r.positives[0], "candidate": c, "cand_score": s,
                                    "pos_score": r.pos_score}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} pairs from {len(df)} queries -> {args.out}")


def judge(args):
    pairs = read_jsonl(args.pairs)
    if args.only_flagged:  # cascade: verify only what another judge flagged (+ its calibration sample)
        keep = {v["id"] for v in read_jsonl(args.only_flagged) if v["relevant"] or v.get("calibration")}
        pairs = [p for p in pairs if p["id"] in keep]
    done = {v["id"] for v in read_jsonl(args.out)}
    todo = [p for p in pairs if p["id"] not in done]
    print(f"{len(pairs)} pairs, {len(done)} already judged, {len(todo)} to go ({args.backend})", flush=True)
    if not todo:
        return
    backend = BACKENDS[args.backend](**({"model": args.model} if args.model else {}))
    model = getattr(backend, "model", None) or args.backend
    # one prompt = one query's candidates, in groups of --batch (shares the query/positive context)
    groups = []
    for _, items in itertools.groupby(sorted(todo, key=lambda p: (p["qid"], p["id"])), key=lambda p: p["qid"]):
        items = list(items)
        groups += [items[i : i + args.batch] for i in range(0, len(items), args.batch)]
    rng = random.Random(0)
    with open(args.out, "a", encoding="utf-8") as f:
        for s in range(0, len(groups), args.chunk):
            chunk = groups[s : s + args.chunk]
            prompts = [build(g[0]["task"], g[0]["query"], g[0]["positive"], g) for g in chunk]
            for g, raw in zip(chunk, backend(prompts)):
                verdicts = parse(raw, [p["id"] for p in g])
                for p in g:
                    if p["id"] not in verdicts:
                        continue  # unparsed: retried on the next run
                    rel = verdicts[p["id"]]
                    f.write(json.dumps({"id": p["id"], "task": p["task"], "judge": args.backend, "model": model,
                                        "prompt": PROMPT_SHA, "relevant": rel,
                                        "calibration": (not rel) and rng.random() < args.calibration,
                                        "raw": raw if p is g[0] else None}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(s + args.chunk, len(groups))}/{len(groups)} prompts", flush=True)


def report(args):
    by_judge = {}
    for path in args.verdicts:
        vs = read_jsonl(path)
        if vs:
            by_judge[vs[0]["judge"] + ":" + vs[0]["model"]] = {v["id"]: v["relevant"] for v in vs}
    for j, v in by_judge.items():
        n = len(v)
        print(f"{j:<45} judged {n:>7}  relevant (= false negative) {sum(v.values()):>6} ({100 * sum(v.values()) / max(n, 1):.1f}%)")
    for (a, va), (b, vb) in itertools.combinations(by_judge.items(), 2):
        common = va.keys() & vb.keys()
        if common:
            agree = sum(va[i] == vb[i] for i in common)
            both = sum(va[i] and vb[i] for i in common)
            print(f"{a} vs {b}: {len(common)} common, agreement {agree / len(common):.3f}, "
                  f"both-relevant {both}, kappa {kappa([va[i] for i in common], [vb[i] for i in common]):.3f}")


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rescored", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--band", type=float, default=0.80)
    s.add_argument("--per-query", type=int, default=3)
    s.add_argument("--pilot", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    j = sub.add_parser("judge")
    j.add_argument("--pairs", required=True)
    j.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    j.add_argument("--model")
    j.add_argument("--out", required=True)
    j.add_argument("--only-flagged")
    j.add_argument("--batch", type=int, default=25, help="candidates per prompt")
    j.add_argument("--chunk", type=int, default=256, help="prompts per backend call (vllm batches them)")
    j.add_argument("--calibration", type=float, default=0.02, help="share of 'not relevant' calls sent to verify")
    r = sub.add_parser("report")
    r.add_argument("verdicts", nargs="+")
    args = ap.parse_args()
    {"select": select, "judge": judge, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
