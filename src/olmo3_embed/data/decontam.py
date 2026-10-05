"""Phase 3 step 2: decontaminate the Track B mix against every BEIR / NanoBEIR / CQADupStack test set (plan §3).

    uv run python -m olmo3_embed.data.decontam [--data data/trackb_v1] [--docs]

Test sets come from the same mteb task metadata (dataset path + revision) the eval harness scores on:
the 14 BEIR tasks besides CQADupStack, the 12 CQADupStack subforums, and the 13 NanoBEIR tasks. For each task the
"test" qrels are its eval_splits plus every other non-train split (MSMARCO: dev + test). Nano tasks keep their eval
split even though it's named "train".

Rules (applied across ALL tasks, not just the matching one, e.g. FEVER claims vs ClimateFEVER):
  1. query exact     normalized train query == a test query                                       -> drop
  2. query near-dup  MinHash LSH over word 3-grams, confirmed Jaccard >= 0.8                      -> drop
     (queries are short, so 3-grams instead of the plan's 5-grams; queries under 3 words get exact match only)
  3. Quora           a QuoraRetrieval row whose query or positive equals a Quora/NanoQuora test
                     query or test-positive document is itself a test duplicate pair              -> drop
  4. positive docs   (--docs) train positives that are also test-qrel positive documents          -> REPORT only
                     (Track B: shared corpora are standard practice; Track A will drop these)
Rule 3 and 4 need the test corpora (--docs streams them once and caches the positive-doc hashes; MSMARCO and
ClimateFEVER are ~3.4 GB each). Without --docs, rule 3 checks Quora queries/positives against test queries only.

Outputs in --data:
  decontam.parquet          rows kept
  decontam_dropped.parquet  one row per drop: qid, task, rule, test_task, test_qid, train_text, test_text
  decontam_report.json      counts per rule x train task x test task, for the data card
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import unicodedata

import mteb
import pandas as pd
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from huggingface_hub import HfApi, hf_hub_download

CACHE = os.path.join(os.environ.get("OLMO3_RAW", "data/raw"), "beir_test")
BEIR = ["MSMARCO", "NQ", "HotpotQA", "FEVER", "FiQA2018", "QuoraRetrieval", "SciFact", "NFCorpus", "ArguAna", "SCIDOCS",
        "ClimateFEVER", "DBPedia", "TRECCOVID", "Touche2020"]
QUORA_TEST = {"QuoraRetrieval", "NanoQuoraRetrieval"}
NUM_PERM, THRESHOLD, NGRAM = 128, 0.8, 3


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


def shingles(n: str) -> set[str]:
    w = n.split()
    return {" ".join(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def h(s: str) -> str:
    return hashlib.sha1(norm(s).encode()).hexdigest()


def test_tasks() -> list:
    ts = list(mteb.get_tasks(tasks=BEIR))
    ts += [t for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
           if t.metadata.name.startswith("CQADupstack") and t.metadata.name != "CQADupstackRetrieval"]
    ts += [t for t in mteb.get_tasks(task_types=["Retrieval"])
           if t.metadata.name.startswith("Nano") and t.metadata.dataset["path"].startswith("mteb/Nano")]
    return ts


def iter_records(repo: str, rev: str, files: list[str]):
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=f"{CACHE}/{repo.split('/')[1]}")
        if f.endswith(".jsonl"):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
        else:
            for b in pq.ParquetFile(p).iter_batches(batch_size=50_000):
                yield from b.to_pylist()


def rid(r: dict) -> str:
    return str(r["_id"] if "_id" in r else r["id"])  # CQADupstack parquet uses "id"


def load_test(task, docs: bool) -> dict:
    """{'queries': {qid: text}, 'pos_docs': {hash: (docid, text)}} for the task's test splits (cached as json)."""
    name, repo, rev = task.metadata.name, task.metadata.dataset["path"], task.metadata.dataset.get("revision")
    cache = f"{CACHE}/{name}.json"
    if os.path.exists(cache):
        c = json.load(open(cache, encoding="utf-8"))
        if c["pos_docs"] is not None or not docs:
            return c
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    qrel_splits = {re.split(r"[/.-]", f)[1] for f in files if f.startswith("qrels/")}
    splits = set(task.metadata.eval_splits) | (qrel_splits - {"train"})
    qrel_files = [f for f in files if f.startswith("qrels/") and re.split(r"[/.-]", f)[1] in splits
                  and (f.endswith(".jsonl") or f.endswith(".parquet"))]
    pos = collections.defaultdict(set)
    for r in iter_records(repo, rev, qrel_files):
        if int(r["score"]) > 0:
            pos[str(r["query-id"])].add(str(r["corpus-id"]))
    qfiles = [f for f in files if f == "queries.jsonl" or (f.startswith("queries/") and f.endswith(".parquet"))]
    queries = {rid(r): r["text"] for r in iter_records(repo, rev, qfiles) if rid(r) in pos}
    pos_docs = None
    if docs:
        want = set().union(*pos.values()) if pos else set()
        cfiles = [f for f in files if f == "corpus.jsonl" or (f.startswith("corpus/") and f.endswith(".parquet"))]
        pos_docs = {}
        for r in iter_records(repo, rev, cfiles):
            if rid(r) in want:
                t = f"{r.get('title') or ''} {r.get('text') or ''}".strip()
                pos_docs[h(t)] = [rid(r), t[:300]]
    c = {"task": name, "repo": repo, "revision": rev, "splits": sorted(splits), "queries": queries, "pos_docs": pos_docs}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(c, open(cache, "w", encoding="utf-8"))
    print(f"  {name}: {len(queries)} test queries" + (f", {len(pos_docs)} positive docs" if docs else ""), flush=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trackb_v1")
    ap.add_argument("--docs", action="store_true", help="also stream test corpora: Quora doc rule + positive-doc report")
    ap.add_argument("--manifest", default="manifests/trackb_v1_dropped_qids.txt",
                    help="committed list of dropped qids (written by a full run, read by --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="skip the checks: drop the manifest's qids from ingested.parquet (Colab: no test corpora needed)")
    args = ap.parse_args()

    df = pd.read_parquet(f"{args.data}/ingested.parquet")
    print(f"train rows: {len(df)}", flush=True)
    if args.apply:
        drop = set(open(args.manifest, encoding="utf-8").read().split())
        kept = df[~df.qid.isin(drop)].reset_index(drop=True)
        missing = len(drop) - (len(df) - len(kept))
        if missing:  # the rebuild must be identical to the one the manifest came from
            raise SystemExit(f"{missing} manifest qids not in ingested.parquet: rebuild differs from the manifest's")
        kept.to_parquet(f"{args.data}/decontam.parquet", index=False, row_group_size=20_000)
        print(f"applied {args.manifest}: kept {len(kept)} / {len(df)}")
        return
    tests = [load_test(t, args.docs) for t in test_tasks()]

    # Test-query indexes: exact (normalized) and MinHash LSH
    exact = collections.defaultdict(list)
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    test_sh = {}
    for c in tests:
        for qid, q in c["queries"].items():
            n = norm(q)
            exact[n].append((c["task"], qid, q))
            sh = shingles(n)
            if sh:
                key = f"{c['task']}\t{qid}"
                test_sh[key] = (sh, q)
                m = MinHash(num_perm=NUM_PERM)
                m.update_batch([s.encode() for s in sh])
                lsh.insert(key, m)
    print(f"test queries: {sum(len(c['queries']) for c in tests)} across {len(tests)} tasks", flush=True)

    quora_q = {norm(q) for c in tests if c["task"] in QUORA_TEST for q in c["queries"].values()}
    quora_d = {k for c in tests if c["task"] in QUORA_TEST and c["pos_docs"] for k in c["pos_docs"]}

    dropped, counts = [], collections.Counter()
    keep = []
    for i, r in enumerate(df.itertuples(index=False)):
        n = norm(r.query)
        hit = None
        if n in exact:
            tt, tq, tx = exact[n][0]
            hit = ("query_exact", tt, tq, tx)
        else:
            sh = shingles(n)
            if sh:
                m = MinHash(num_perm=NUM_PERM)
                m.update_batch([s.encode() for s in sh])
                for key in lsh.query(m):
                    tsh, tx = test_sh[key]
                    if len(sh & tsh) / len(sh | tsh) >= THRESHOLD:
                        tt, tq = key.split("\t")
                        hit = ("query_near", tt, tq, tx)
                        break
        if hit is None and r.task == "QuoraRetrieval":
            for p in r.positives:
                if norm(p) in quora_q or h(p) in quora_d:
                    hit = ("quora_pair", "QuoraRetrieval", "", p)
                    break
        if hit:
            counts[(hit[0], r.task, hit[1])] += 1
            dropped.append({"qid": r.qid, "task": r.task, "rule": hit[0], "test_task": hit[1], "test_qid": hit[2],
                            "train_text": r.query, "test_text": hit[3]})
        else:
            keep.append(i)
        if (i + 1) % 100_000 == 0:
            print(f"  checked {i + 1} rows, dropped {len(dropped)}", flush=True)

    kept = df.iloc[keep].reset_index(drop=True)
    report = {"train_rows_in": len(df), "train_rows_out": len(kept), "dropped": len(dropped),
              "drops": [{"rule": a, "train_task": b, "test_task": c, "n": n} for (a, b, c), n in sorted(counts.items())],
              "test_sets": [{"task": c["task"], "repo": c["repo"], "revision": c["revision"], "splits": c["splits"],
                             "queries": len(c["queries"])} for c in tests],
              "params": {"ngram": NGRAM, "jaccard": THRESHOLD, "num_perm": NUM_PERM, "docs": args.docs}}

    if args.docs:  # rule 4: report train positives that are test-qrel positive docs, per (train task, test task)
        doc_index = collections.defaultdict(list)
        for c in tests:
            for k in c["pos_docs"] or {}:
                doc_index[k].append(c["task"])
        overlap = collections.Counter()
        for r in kept.itertuples(index=False):
            for t in {tt for p in r.positives if p.strip() for tt in doc_index.get(h(p), [])}:
                overlap[(r.task, t)] += 1
        report["positive_doc_overlap_rows"] = [{"train_task": a, "test_task": b, "rows": n}
                                               for (a, b), n in sorted(overlap.items())]

    kept.to_parquet(f"{args.data}/decontam.parquet", index=False, row_group_size=20_000)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(d["qid"] for d in dropped)) + "\n")
    pd.DataFrame(dropped).to_parquet(f"{args.data}/decontam_dropped.parquet", index=False)
    json.dump(report, open(f"{args.data}/decontam_report.json", "w", encoding="utf-8"), indent=1)
    print(f"kept {len(kept)} / {len(df)}; dropped {len(dropped)}")
    for d in report["drops"]:
        print(f"  {d['rule']:<12} {d['train_task']:<15} vs {d['test_task']:<32} {d['n']}")


if __name__ == "__main__":
    main()
