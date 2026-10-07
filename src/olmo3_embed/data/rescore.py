"""Phase 3 step 3: rescore hard-negative candidates with an embedder, apply the TopK-PercPos guard, mine SciFact and
NFCorpus negatives (docs/phase3-plan.md §4a).

    python -m olmo3_embed.data.rescore --data data/trackb_v1 --work <scratch dir> [--out <dir>] [--limit N] [--bench N]

Teacher: Qwen/Qwen3-Embedding-0.6B (bf16), queries as "Instruct: {task instruction}\\nQuery:{query}" (Qwen3's
format), documents plain, cosine similarity.

1. Mine SciFact/NFCorpus: embed the whole (small) corpus, take each query's top 100, keep ranks 10-100 that aren't
   positives. Other tasks keep their upstream candidates (RLHN / bge), capped at --max-cands in source order.
2. Embed every unique query and candidate/positive text once (length-sorted shards in --work, resumable: a finished
   shard is never recomputed).
3. Score: pos_score = best positive. Guard: drop a candidate scoring > --guard x pos_score (NV-Retriever's
   TopK-PercPos, 0.95). Survivors are sorted hardest-first with their scores; train.py samples k from the top.
   Candidates in [--band x pos_score, guard x pos_score] are the LLM judge's input (plan §4b), counted in the stats.

Outputs in --out (default --data): rescored.parquet (qid, task, source, query, positives, pos_score, negatives,
neg_scores, n_cands, n_guarded, n_band) and rescore_stats.json.
--bench N: embed N sample docs, print docs/s and the projected time for the full job, then exit.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch

TEACHER = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCTIONS = {  # the instructions Qwen3-Embedding / E5 use for these MTEB tasks
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
}
MINE = {"SciFact": "scifact", "NFCorpus": "nfcorpus"}  # raw dirs written by build_trackb
RAW = os.environ.get("OLMO3_RAW", "data/raw")


INSTRUCTIONS.update({  # task-variety sources (teacher prompts only; training uses no instructions)
    "StackExchange": "Given a question title from a Q&A forum, retrieve the full question post",
    "S2ORCCitation": "Given the title of a scientific paper, retrieve the titles of papers it cites",
})
DEFAULT_INSTRUCTION = "Given a query, retrieve relevant documents"


def qtext(task: str, q: str) -> str:
    return f"Instruct: {INSTRUCTIONS.get(task, DEFAULT_INSTRUCTION)}\nQuery:{q}"


def load_model(max_len: int):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(TEACHER, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    m.max_seq_length = max_len
    return m


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float16)


def embed_all(model, texts: list[str], work: str, tag: str, batch_size: int, shard: int) -> np.ndarray:
    """Length-sorted shards saved to work/<tag>_<fingerprint>_<i>.npy; reruns load finished shards. The fingerprint
    covers the exact text list, shard size and teacher, so a changed input can never reuse stale shards."""
    fp = hashlib.sha1(f"{TEACHER}|{shard}|{len(texts)}".encode())
    for t in texts:
        fp.update(t.encode())
        fp.update(b"\0")
    tag = f"{tag}_{fp.hexdigest()[:12]}"
    order = np.argsort([-len(t) for t in texts], kind="stable")
    dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
    out = np.zeros((len(texts), dim), dtype=np.float16)
    n = (len(texts) + shard - 1) // shard
    t0 = time.time()
    for i in range(n):
        idx = order[i * shard : (i + 1) * shard]
        path = os.path.join(work, f"{tag}_{i:04d}.npy")
        if os.path.exists(path):
            out[idx] = np.load(path)
            continue
        e = encode(model, [texts[j] for j in idx], batch_size)
        np.save(path + ".tmp.npy", e)
        os.replace(path + ".tmp.npy", path)  # a shard exists only once fully written
        out[idx] = e
        done = (i + 1) * shard
        rate = min(done, len(texts)) / (time.time() - t0)
        print(f"  {tag}: shard {i + 1}/{n}, {rate:.0f} texts/s, ~{(len(texts) - done) / rate / 60:.0f} min left",
              flush=True)
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mine(df: pd.DataFrame, model, batch_size: int, top: int = 100, lo: int = 10,
         pool: dict[str, list[str]] | None = None) -> pd.DataFrame:
    for task, d in MINE.items():
        rows = df.index[df.task == task]
        if len(rows) == 0:
            continue
        corpus = [f"{r.get('title') or ''} {r.get('text') or ''}".strip() for r in read_jsonl(f"{RAW}/{d}/corpus.jsonl")]
        corpus = [c for c in dict.fromkeys(corpus) if c]
        ce = torch.from_numpy(encode(model, corpus, batch_size)).cuda()
        qe = torch.from_numpy(encode(model, [qtext(task, q) for q in df.loc[rows, "query"]], batch_size)).cuda()
        top_idx = (qe @ ce.T).topk(min(top, len(corpus)), dim=1).indices.cpu().numpy()
        for r, ids in zip(rows, top_idx):
            pos = set(df.at[r, "positives"])
            df.at[r, "negatives"] = [corpus[j] for j in ids[lo - 1 :] if corpus[j] not in pos]
        print(f"  mined {task}: {len(rows)} queries over {len(corpus)} docs, ranks {lo}-{top}", flush=True)
    # Tasks with no upstream negatives and no corpus file (task-variety sources: StackExchange, S2ORC citations,
    # synthetic): mine from the task's own pool of positives, i.e. other rows' documents of the same kind, plus the
    # task's --pool documents if given (a large same-domain corpus: own pools of a few thousand are too easy).
    for task in sorted(set(df.task) - set(MINE)):
        rows = df.index[df.task == task]
        if any(len(n) for n in df.loc[rows, "negatives"]):
            continue
        corpus = list(dict.fromkeys([p for P in df.loc[rows, "positives"] for p in P if p] + (pool or {}).get(task, [])))
        ce = torch.from_numpy(encode(model, corpus, batch_size)).cuda()
        qs = [qtext(task, q) for q in df.loc[rows, "query"]]
        for s in range(0, len(rows), 4096):  # chunked: a 60k x 60k score matrix wouldn't fit
            qe = torch.from_numpy(encode(model, qs[s : s + 4096], batch_size)).cuda()
            top_idx = (qe @ ce.T).topk(min(top, len(corpus)), dim=1).indices.cpu().numpy()
            for r, ids in zip(rows[s : s + 4096], top_idx):
                pos = set(df.at[r, "positives"])
                df.at[r, "negatives"] = [corpus[j] for j in ids[lo - 1 :] if corpus[j] not in pos]
        print(f"  mined {task} (own pool{' + --pool' if task in (pool or {}) else ''}): {len(rows)} queries over "
              f"{len(corpus)} docs, ranks {lo}-{top}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trackb_v1")
    ap.add_argument("--work", default="data/trackb_v1/emb")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-cands", type=int, default=30)
    ap.add_argument("--guard", type=float, default=0.95)
    ap.add_argument("--band", type=float, default=0.80)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--shard", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--pool", nargs="*", default=[],
                    help="parquet(s) with task, text: extra mining corpus for own-pool tasks (synth.py pool)")
    args = ap.parse_args()
    out_dir = args.out or args.data
    os.makedirs(args.work, exist_ok=True)

    df = pd.read_parquet(f"{args.data}/decontam.parquet")
    if args.limit:
        df = df.groupby("task", group_keys=False).head(max(1, args.limit // df.task.nunique())).reset_index(drop=True)
    df["negatives"] = df.negatives.map(lambda n: list(n)[: args.max_cands])
    df["positives"] = df.positives.map(list)
    print(f"rows {len(df)}; gpu {torch.cuda.get_device_name(0)}", flush=True)
    model = load_model(args.max_len)

    docs = list(dict.fromkeys(d for col in ("positives", "negatives") for L in df[col] for d in L))
    if args.bench:
        sample = [docs[i] for i in np.random.default_rng(0).choice(len(docs), min(args.bench, len(docs)), replace=False)]
        encode(model, sample[:512], args.batch_size)  # warm-up
        torch.cuda.synchronize()
        t = time.time()
        encode(model, sample, args.batch_size)
        torch.cuda.synchronize()
        rate = len(sample) / (time.time() - t)
        total = len(docs) + 5_000 + len(df)  # + mined corpora + queries, roughly
        print(f"BENCH {torch.cuda.get_device_name(0)}: {rate:.0f} docs/s -> ~{total / rate / 3600:.2f} h for "
              f"{total:,} texts; peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        return

    pool = {}
    for p in args.pool:
        for task, d in pd.read_parquet(p).groupby("task"):
            pool.setdefault(task, []).extend(d.text.tolist())
    df = mine(df, model, args.batch_size, pool=pool)
    df["negatives"] = df.negatives.map(lambda n: n[: args.max_cands])  # mined lists are rank-ordered: ranks 10-39
    docs = list(dict.fromkeys(d for col in ("positives", "negatives") for L in df[col] for d in L))
    doc_id = {d: i for i, d in enumerate(docs)}
    print(f"unique docs {len(docs):,}", flush=True)
    tag = f"docs_l{args.limit}" if args.limit else "docs"
    de = embed_all(model, docs, args.work, tag, args.batch_size, args.shard)
    qe = embed_all(model, [qtext(t, q) for t, q in zip(df.task, df["query"])], args.work, tag.replace("docs", "queries"),
                   args.batch_size, args.shard)

    pos_score, negs, neg_scores, n_guarded, n_band = [], [], [], [], []
    for i, (P, N) in enumerate(zip(df.positives, df.negatives)):
        q = qe[i].astype(np.float32)
        ps = float((de[[doc_id[p] for p in P]].astype(np.float32) @ q).max())
        if N:
            cs = de[[doc_id[n] for n in N]].astype(np.float32) @ q
            keep = cs <= args.guard * ps
            o = np.argsort(-cs[keep])
            negs.append([N[j] for j in np.flatnonzero(keep)[o]])
            neg_scores.append(cs[keep][o].round(4).tolist())
            n_guarded.append(int((~keep).sum()))
            n_band.append(int(((cs >= args.band * ps) & keep).sum()))
        else:
            negs.append([]), neg_scores.append([]), n_guarded.append(0), n_band.append(0)
        pos_score.append(round(ps, 4))
    df["n_cands"] = df.negatives.map(len)
    df["negatives"], df["neg_scores"], df["pos_score"] = negs, neg_scores, pos_score
    df["n_guarded"], df["n_band"] = n_guarded, n_band

    name = f"rescored_l{args.limit}" if args.limit else "rescored"
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(f"{out_dir}/{name}.parquet", index=False, row_group_size=20_000)
    g = df.groupby("task")
    stats = pd.DataFrame({
        "rows": g.size(), "cands/row": g.n_cands.mean(), "guarded/row": g.n_guarded.mean(),
        "guarded%": 100 * g.n_guarded.sum() / g.n_cands.sum().clip(lower=1), "judge_band/row": g.n_band.mean(),
        "rows_<3_negs": g.negatives.apply(lambda s: int((s.map(len) < 3).sum())), "pos_score": g.pos_score.mean(),
    }).round(2)
    print(stats.to_string())
    with open(f"{out_dir}/{name}_stats.json", "w", encoding="utf-8") as f:
        json.dump({"teacher": TEACHER, "args": vars(args), "gpu": torch.cuda.get_device_name(0),
                   "unique_docs": len(docs), "by_task": stats.reset_index().to_dict(orient="records")}, f, indent=1)


if __name__ == "__main__":
    main()
