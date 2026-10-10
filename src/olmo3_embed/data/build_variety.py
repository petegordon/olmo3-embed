"""Task-variety V1: public non-BEIR pairs (docs/task-variety-plan.md, sources A; vetted 2026-10-07).

    python -m olmo3_embed.data.build_variety [--per-site 1000] [--s2orc 60000] [--out data/variety_v1]
    python -m olmo3_embed.data.build_variety --sources pubmedqa --pubmedqa 30000 --out data/variety_bio
    python -m olmo3_embed.data.build_variety --sources s2search --s2search 30000 --out data/variety_s2search

  StackExchange  flax-sentence-embeddings/stackexchange_title_body_jsonl (CC BY-SA 4.0): question title -> title+body.
                 Every site EXCEPT the 12 CQADupStack test forums and money (FiQA-2018's source: decontam found 44
                 exact FiQA test queries there); up to --per-site rows per site (seeded sample).
  S2ORCCitation  sentence-transformers/s2orc title-citation-pair (ODC-By 1.0): paper title -> title it cites.
                 Pairs where either title is a SCIDOCS corpus title are dropped (SCIDOCS is built from the same S2
                 graph); --s2orc rows sampled from the stream.
  PubMedQA       qiaojin/PubMedQA pqa_artificial (MIT; Phase 6, biomedical gap / TREC-COVID shape): research question
                 (generated from the paper's title) -> that paper's PubMed abstract (context paragraphs + conclusion).
                 One row per PubMed id, --pubmedqa rows sampled. Zero-shot protocol: no TREC-COVID queries, qrels or
                 CORD-19 documents; doc_decontam.py drops any abstract matching TREC-COVID/NFCorpus/SciFact corpus docs.
  S2Search       allenai/scirepeval search train (ODC-BY; Phase 6 fallback source): real Semantic Scholar search queries
                 -> the clicked paper (title + abstract). Pasted-title queries are dropped (query words mostly inside
                 the clicked title), keeping short keyword searches, TREC-COVID's query shape. One row per query.

Output <out>/ingested.parquet in Track B's schema (qid, task, source, query, positives, negatives=[]). Then, as for
Track B: decontam.py --docs (queries), doc_decontam.py --check (passages), rescore.py (negatives mined within each
task's own positives pool).
"""
import argparse
import collections

import json
import os
import random

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from olmo3_embed.data.decontam import CACHE, iter_records, norm, test_tasks

RAW = os.environ.get("OLMO3_RAW", "data/raw")
SE_REPO = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
CQADUP_SITES = {"android", "english", "gaming", "gis", "mathematica", "physics", "programmers", "softwareengineering",
                "stats", "tex", "unix", "webmasters", "wordpress"}  # CQADupStack's 12 test forums (programmers renamed)
EXCLUDED_SITES = CQADUP_SITES | {"money"}  # money.stackexchange (personal finance) is FiQA-2018's question source


def site_of(path: str) -> str:
    return path.split(".")[0]


def ingest_stackexchange(per_site: int, seed: int, stats: dict, max_scan: int = 200_000) -> list[dict]:
    """Streams each site file from the Hub (no full download: stackoverflow alone is 7.8 GB) and reservoir-samples
    per_site posts from its first max_scan lines. Memory stays O(per_site)."""
    files = [f for f in HfApi().list_repo_files(SE_REPO, repo_type="dataset") if f.endswith(".jsonl.gz")]
    rows = []
    for f in sorted(files):
        site = site_of(f)
        if site in EXCLUDED_SITES:
            stats["stackexchange"]["excluded_test_sites"] += 1
            continue
        ds = load_dataset("json", data_files=f"hf://datasets/{SE_REPO}/{f}", split="train", streaming=True)
        rng, keep, seen = random.Random(f"{seed}:{site}"), [], 0
        for r in ds.take(max_scan):
            title, body = (r.get("texts") or [None, None])[:2]
            if not title or not body or len(body) < 40:
                continue
            seen += 1
            item = (seen, title.strip(), body.strip())
            if len(keep) < per_site:
                keep.append(item)
            elif (j := rng.randrange(seen)) < per_site:
                keep[j] = item
        for i, title, body in keep:
            rows.append({"qid": f"StackExchange:{site}:{i}", "task": "StackExchange", "source": f"{SE_REPO}/{site}",
                         "query": title, "positives": [f"{title} {body}"], "negatives": []})
        stats["stackexchange"]["sites"] += 1
        stats["stackexchange"]["rows"] += len(keep)
        print(f"  {site}: {len(keep)} of {seen} scanned", flush=True)
    return rows


def scidocs_titles() -> set[str]:
    """Normalized titles of every SCIDOCS (and NanoSCIDOCS) corpus document and query."""
    titles = set()
    for task in test_tasks():
        name = task.metadata.name
        if "SCIDOCS" not in name:
            continue
        repo, rev = task.metadata.dataset["path"], task.metadata.dataset.get("revision")
        files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
        for kind in ("corpus", "queries"):
            fs = [f for f in files if f == f"{kind}.jsonl" or (f.startswith(f"{kind}/") and f.endswith(".parquet"))]
            for r in iter_records(repo, rev, fs):
                t = r.get("title") if kind == "corpus" else r.get("text")
                if t and norm(t):
                    titles.add(norm(t))
    return titles


def ingest_s2orc(n: int, seed: int, stats: dict) -> list[dict]:
    bad = scidocs_titles()
    stats["s2orc"]["scidocs_titles"] = len(bad)
    ds = load_dataset("sentence-transformers/s2orc", "title-citation-pair", split="train", streaming=True)
    rows, seen = [], set()
    for i, r in enumerate(ds.shuffle(seed=seed, buffer_size=200_000)):
        a, b = (r.get("title") or "").strip(), (r.get("citation") or "").strip()
        if not a or not b:
            continue
        na, nb = norm(a), norm(b)
        if na in bad or nb in bad:
            stats["s2orc"]["dropped_scidocs_title"] += 1
            continue
        if na in seen:  # one row per citing paper keeps the mix diverse
            continue
        seen.add(na)
        rows.append({"qid": f"S2ORCCitation:{i}", "task": "S2ORCCitation", "source": "sentence-transformers/s2orc",
                     "query": a, "positives": [b], "negatives": []})
        if len(rows) >= n:
            break
    stats["s2orc"]["rows"] = len(rows)
    return rows


def ingest_pubmedqa(n: int, seed: int, stats: dict) -> list[dict]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    stats["pubmedqa"]["source_rows"] = len(ds)
    rows, seen = [], set()
    for i in random.Random(seed).sample(range(len(ds)), len(ds)):
        r = ds[i]
        q = (r.get("question") or "").strip()
        ctx = r.get("context") or {}
        parts = [c.strip() for c in (ctx.get("contexts") or []) if c and c.strip()]
        if (r.get("long_answer") or "").strip():
            parts.append(r["long_answer"].strip())  # full abstract: CORD-19 / NFCorpus / SciFact docs are full abstracts
        doc = " ".join(parts)
        pid = str(r.get("pubid"))
        if not q or len(doc) < 200 or pid in seen:
            stats["pubmedqa"]["skipped_empty_or_dup"] += 1
            continue
        seen.add(pid)
        rows.append({"qid": f"PubMedQA:{pid}", "task": "PubMedQA", "source": "qiaojin/PubMedQA/pqa_artificial",
                     "query": q, "positives": [doc], "negatives": []})
        if len(rows) >= n:
            break
    stats["pubmedqa"]["rows"] = len(rows)
    return rows


def ingest_s2search(n: int, seed: int, stats: dict, max_words: int = 12, max_title_overlap: float = 0.6) -> list[dict]:
    ds = load_dataset("allenai/scirepeval", "search", split="train", streaming=True)
    rows, seen = [], set()
    for i, r in enumerate(ds.shuffle(seed=seed, buffer_size=100_000)):
        st = stats["s2search"]
        st["scanned"] += 1
        q = (r.get("query") or "").strip()
        cands = r.get("candidates") or []
        cands = json.loads(cands) if isinstance(cands, str) else cands
        pos = [c for c in cands if (c.get("score") or 0) > 0 and (c.get("abstract") or "").strip()]
        nq = norm(q)
        if not nq or not pos or nq in seen:
            st["skipped_no_click_or_dup"] += 1
            continue
        qw = set(nq.split())
        if len(qw) > max_words:
            st["skipped_long_query"] += 1
            continue
        title_w = set(norm(pos[0].get("title") or "").split())
        if len(qw & title_w) / len(qw) > max_title_overlap:  # a pasted title, not a search
            st["skipped_title_query"] += 1
            continue
        docs = [f"{(c.get('title') or '').strip()} {(c.get('abstract') or '').strip()}".strip() for c in pos]
        docs = [d for d in docs if len(d) >= 200]
        if not docs:
            st["skipped_short_doc"] += 1
            continue
        seen.add(nq)
        rows.append({"qid": f"S2Search:{i}", "task": "S2Search", "source": "allenai/scirepeval/search",
                     "query": q, "positives": docs, "negatives": []})
        if len(rows) >= n:
            break
    stats["s2search"]["rows"] = len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-site", type=int, default=1000)
    ap.add_argument("--s2orc", type=int, default=60_000)
    ap.add_argument("--pubmedqa", type=int, default=30_000)
    ap.add_argument("--s2search", type=int, default=30_000)
    ap.add_argument("--sources", default="stackexchange,s2orc", help="comma list of stackexchange, s2orc, pubmedqa, s2search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/variety_v1")
    args = ap.parse_args()
    stats = collections.defaultdict(collections.Counter)
    src = {x.strip() for x in args.sources.split(",")}
    rows = []
    if "stackexchange" in src:
        rows += ingest_stackexchange(args.per_site, args.seed, stats)
    if "s2orc" in src:
        rows += ingest_s2orc(args.s2orc, args.seed, stats)
    if "pubmedqa" in src:
        rows += ingest_pubmedqa(args.pubmedqa, args.seed, stats)
    if "s2search" in src:
        rows += ingest_s2search(args.s2search, args.seed, stats)
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{args.out}/ingested.parquet", index=False, row_group_size=20_000)
    print(df.groupby("task").size().to_string(), f"\ntotal {len(df)} -> {args.out}/ingested.parquet")
    with open(f"{args.out}/ingest_stats.json", "w", encoding="utf-8") as f:
        json.dump({"sources": {k: dict(v) for k, v in stats.items()}, "args": vars(args)}, f, indent=1)


if __name__ == "__main__":
    main()
