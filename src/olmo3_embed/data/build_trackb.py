"""Phase 3 step 1: ingest the Track B v1 sources into one unified parquet (docs/phase3-plan.md §2, §8).

    uv run python -m olmo3_embed.data.build_trackb [--msmarco-cap 150000] [--out data/trackb_v1]

Sources (downloaded to data/raw/ on first use):
  rlhn/rlhn-680K                   msmarco_passage (capped), nq, hotpotqa, fever, fiqa. arguana and scidocsrr are
                                   EXCLUDED: BEIR has no train split for them, so they overlap the test sets.
  cfli/bge-full-data (quora split) Quora duplicates. BEIR has no Quora train split: decontam.py must check it strictly.
  mteb/scifact, mteb/nfcorpus      official BEIR train qrels; negatives are mined later (rescore.py), so empty here.

Output: <out>/ingested.parquet, one row per training query:
  qid        "<task>:<source id>" (stable across rebuilds)
  task       MTEB task name (MSMARCO, NQ, HotpotQA, FEVER, FiQA2018, QuoraRetrieval, SciFact, NFCorpus); train.py
             resolves the per-task instruction from it (instructions: per_task)
  source     dataset the row came from
  query      raw query text (no instruction)
  positives  list[str], documents as "title text" (MTEB's corpus format)
  negatives  list[str], in source order (RLHN/bge: as mined upstream; SciFact/NFCorpus: empty until rescore.py)
Plus <out>/ingest_stats.json: rows in/out per source, for the data card.
"""
import argparse
import collections
import glob
import json
import os
import random

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

RAW = "data/raw"
RLHN_TASKS = {"msmarco_passage": "MSMARCO", "nq": "NQ", "hotpotqa": "HotpotQA", "fever": "FEVER", "fiqa": "FiQA2018"}
RLHN_EXCLUDED = {"arguana", "scidocsrr"}
BEIR_TRAIN = {"SciFact": "mteb/scifact", "NFCorpus": "mteb/nfcorpus"}


def doc(title: str | None, text: str | None) -> str:
    return f"{title or ''} {text or ''}".strip()


def ingest_rlhn(msmarco_cap: int, seed: int, stats: dict) -> list[dict]:
    path = snapshot_download("rlhn/rlhn-680K", repo_type="dataset", local_dir=f"{RAW}/rlhn-680K",
                             allow_patterns=["data/*.parquet", "README.md"])
    rows, seen = [], set()
    for f in sorted(glob.glob(f"{path}/data/*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20_000):
            for r in batch.to_pylist():
                sub = r["subset"]
                stats[f"rlhn:{sub}"]["in"] += 1
                if sub in RLHN_EXCLUDED:
                    stats[f"rlhn:{sub}"]["excluded_no_beir_train_split"] += 1
                    continue
                key = (sub, r["query_id"])
                if key in seen:
                    stats[f"rlhn:{sub}"]["dup_query_id"] += 1
                    continue
                seen.add(key)
                pos = [d for d in (doc(p["title"], p["text"]) for p in r["positive_passages"]) if d]  # FiQA has empty docs
                if not pos or not r["query"].strip():
                    stats[f"rlhn:{sub}"]["empty"] += 1
                    continue
                negs = [d for d in (doc(n["title"], n["text"]) for n in r["negative_passages"]) if d and d not in pos]
                rows.append({"qid": f"{RLHN_TASKS[sub]}:{r['query_id']}", "task": RLHN_TASKS[sub], "source": "rlhn-680K",
                             "query": r["query"], "positives": pos, "negatives": negs})
    # Cap MS MARCO with a seeded sample so it doesn't dominate the mix (plan §2)
    ms = [r for r in rows if r["task"] == "MSMARCO"]
    if len(ms) > msmarco_cap:
        keep = set(r["qid"] for r in random.Random(seed).sample(ms, msmarco_cap))
        stats["rlhn:msmarco_passage"]["capped"] = len(ms) - msmarco_cap
        rows = [r for r in rows if r["task"] != "MSMARCO" or r["qid"] in keep]
    return rows


def ingest_bge_quora(stats: dict) -> list[dict]:
    p = hf_hub_download("cfli/bge-full-data", "data/quora-00000-of-00001.parquet", repo_type="dataset",
                        local_dir=f"{RAW}/bge-full-data")
    rows, seen = [], set()
    for i, r in enumerate(pq.read_table(p, columns=["query", "pos", "neg"]).to_pylist()):
        stats["bge:quora"]["in"] += 1
        q = r["query"].strip()
        if not q or not r["pos"]:
            stats["bge:quora"]["empty"] += 1
            continue
        if q in seen:
            stats["bge:quora"]["dup_query"] += 1
            continue
        seen.add(q)
        negs = [n for n in r["neg"] if n and n not in r["pos"]]
        rows.append({"qid": f"QuoraRetrieval:bge{i}", "task": "QuoraRetrieval", "source": "bge-full-data",
                     "query": q, "positives": list(r["pos"]), "negatives": negs})
    return rows


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ingest_beir_train(task: str, repo: str, stats: dict) -> list[dict]:
    """Official BEIR train qrels. Positives = the query's highest-relevance docs (NFCorpus grades 1-2)."""
    d = snapshot_download(repo, repo_type="dataset", local_dir=f"{RAW}/{repo.split('/')[1]}",
                          allow_patterns=["corpus.jsonl", "queries.jsonl", "qrels/train.jsonl"])
    corpus = {r["_id"]: doc(r.get("title"), r.get("text")) for r in read_jsonl(f"{d}/corpus.jsonl")}
    queries = {r["_id"]: r["text"] for r in read_jsonl(f"{d}/queries.jsonl")}
    rel = collections.defaultdict(dict)
    for r in read_jsonl(f"{d}/qrels/train.jsonl"):
        score = int(r["score"])  # mteb's qrels jsonl stores scores as strings
        if score > 0:
            rel[str(r["query-id"])][str(r["corpus-id"])] = score
    rows = []
    for qid, docs in rel.items():
        stats[f"beir:{task}"]["in"] += 1
        top = max(docs.values())
        pos = [corpus[c] for c, s in docs.items() if s == top and c in corpus]
        if qid not in queries or not pos:
            stats[f"beir:{task}"]["missing_query_or_doc"] += 1
            continue
        rows.append({"qid": f"{task}:{qid}", "task": task, "source": repo, "query": queries[qid],
                     "positives": pos, "negatives": []})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msmarco-cap", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/trackb_v1")
    args = ap.parse_args()

    stats = collections.defaultdict(collections.Counter)
    rows = ingest_rlhn(args.msmarco_cap, args.seed, stats)
    rows += ingest_bge_quora(stats)
    for task, repo in BEIR_TRAIN.items():
        rows += ingest_beir_train(task, repo, stats)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    # Small row groups: one huge group makes the nested list columns unreadable into pandas (>2 GB chunks)
    df.to_parquet(f"{args.out}/ingested.parquet", index=False, row_group_size=20_000)
    by_task = df.groupby("task").agg(rows=("qid", "size"), avg_pos=("positives", lambda s: s.map(len).mean()),
                                     avg_neg=("negatives", lambda s: s.map(len).mean()))
    print(by_task.round(1).to_string())
    print(f"total {len(df)} rows -> {args.out}/ingested.parquet")
    with open(f"{args.out}/ingest_stats.json", "w", encoding="utf-8") as f:
        json.dump({"sources": {k: dict(v) for k, v in stats.items()},
                   "by_task": by_task.round(2).reset_index().to_dict(orient="records"),
                   "args": vars(args)}, f, indent=1)


if __name__ == "__main__":
    main()
