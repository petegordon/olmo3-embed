"""Document-level decontamination for new training sources (docs/task-variety-plan.md, "Ground rules").

decontam.py checks training *queries* against test queries. New task-variety sources (argument text, Wikipedia
entities, scientific abstracts) also put test-corpus *documents* at risk, so passages are checked too:

  doc_exact / doc_prefix   passage == a document of a FULL-CORPUS test set (normalized text, or the same first 200
                           normalized chars). Full corpora are indexed for every non-Wikipedia test set: Touche-2020
                           (args.me), ArguAna (idebate), SCIDOCS, CQADupStack (12 forums), Quora, TREC-COVID, FiQA,
                           SciFact, NFCorpus. ~1.7M documents, hashed.
  pos_exact / pos_near     passage == or near-duplicate of (MinHash over word 3-grams of the first 300 chars,
                           Jaccard >= 0.8) a test-qrel POSITIVE document of any of the 39 test sets. This also covers
                           the Wikipedia-based tasks (ClimateFEVER, DBPedia, FEVER, HotpotQA, NQ), whose corpora are too
                           large to index whole and are shared with standard training data anyway.
  doc_near                 passage near-duplicates (same MinHash rule) ANY document of the biomedical full corpora:
                           TREC-COVID (CORD-19, 171k), NFCorpus, SciFact. Catches the same abstract with different
                           formatting or a truncated/extended version (Phase 6: PubMed sources vs CORD-19).
  doc_contain              >= CONTAIN (5%) of the passage's word 8-grams occur anywhere in a NEAR_FULL corpus. Needed
                           because CORD-19 stores "title + section label + abstract" while PubMed sources carry the bare
                           abstract: the same text, offset, so neither exact/prefix nor first-300-char MinHash matches.
                           Phase 6 (PubMedQA, 30k): the old rules caught 5 rows, this rule 186 more (139 at >=50%).

    python -m olmo3_embed.data.doc_decontam --build                      # once: index -> $OLMO3_RAW/beir_test/
    python -m olmo3_embed.data.doc_decontam --check data.parquet --columns positives negatives [--out kept.parquet]

Any flagged passage is reported. With --out, rows whose query-side positive is flagged are dropped, and flagged
negatives are removed from the row (a test document as a *negative* teaches the model to push it away, which is
also a leak).
"""
import argparse
import collections
import hashlib
import json
import os
import pickle

import numpy as np

from datasketch import MinHash, MinHashLSH
from huggingface_hub import HfApi

from olmo3_embed.data.decontam import CACHE, h, iter_records, load_test, norm, rid, shingles, test_tasks

FULL_CORPUS = {"Touche2020", "ArguAna", "SCIDOCS", "NFCorpus", "SciFact", "FiQA2018", "TRECCOVID", "QuoraRetrieval"}
NEAR_FULL = {"TRECCOVID", "NFCorpus", "SciFact"}  # every corpus doc gets the near-duplicate check, not just positives
PREFIX, NEAR_CHARS, NUM_PERM, JACCARD = 200, 300, 128, 0.8
CONTAIN_N, CONTAIN = 8, 0.05
INDEX = os.path.join(CACHE, "doc_index_v2.pkl")  # v2: + doc_near over NEAR_FULL corpora


def h64(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "big")


def grams8(n: str) -> np.ndarray:
    """64-bit hashes of the word CONTAIN_N-grams of normalized text n."""
    w = n.split()
    return np.array([h64(" ".join(w[i : i + CONTAIN_N])) for i in range(len(w) - CONTAIN_N + 1)], dtype=np.uint64)


def mh(sh: set[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch([s.encode() for s in sh])
    return m


class DocIndex:
    def __init__(self):
        self.full: dict[int, str] = {}  # hash of normalized text -> task
        self.prefix: dict[int, str] = {}  # hash of first PREFIX normalized chars (long docs only) -> task
        self.pos: dict[str, tuple[str, str]] = {}  # decontam.h(full positive text) -> (task, docid)
        self.pos_sh: dict[str, set[str]] = {}
        self.lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
        self.grams: dict[str, np.ndarray] = {}  # NEAR_FULL task -> sorted unique 8-gram hashes of its whole corpus

    def contain(self, n: str):
        g = grams8(n)
        if not len(g):
            return None
        for task, s in getattr(self, "grams", {}).items():
            i = np.searchsorted(s, g)
            i[i == len(s)] = 0
            frac = float((s[i] == g).mean())
            if frac >= CONTAIN:
                return ("doc_contain", task, f"{frac:.2f}")
        return None

    def add_full(self, task: str, text: str):
        n = norm(text)
        if not n:
            return
        self.full.setdefault(h64(n), task)
        if len(n) > PREFIX:
            self.prefix.setdefault(h64(n[:PREFIX]), task)

    def add_near(self, task: str, docid: str, text: str):
        """Near-duplicate entry for a full-corpus document (NEAR_FULL tasks); keyed 'task\tdoc:<id>'."""
        sh = shingles(norm(text)[:NEAR_CHARS])
        key = f"{task}\tdoc:{docid}"
        if sh and key not in self.pos_sh:
            self.pos_sh[key] = sh
            self.lsh.insert(key, mh(sh))

    def add_positive(self, task: str, docid: str, text: str, full_hash: str | None = None):
        """text may be a prefix (decontam.py caches the first 300 chars); full_hash = decontam.h(full text)."""
        n = norm(text)
        if not n:
            return
        self.pos.setdefault(full_hash or h(text), (task, docid))
        sh = shingles(n[:NEAR_CHARS])
        key = f"{task}\t{docid}"
        if sh and key not in self.pos_sh:
            self.pos_sh[key] = sh
            self.lsh.insert(key, mh(sh))

    def check(self, text: str):
        """None if clean, else (rule, task, detail)."""
        n = norm(text)
        if not n:
            return None
        k = h64(n)
        if k in self.full:
            return ("doc_exact", self.full[k], "")
        if len(n) > PREFIX and h64(n[:PREFIX]) in self.prefix:
            return ("doc_prefix", self.prefix[h64(n[:PREFIX])], "")
        hk = h(text)
        if hk in self.pos:
            return ("pos_exact",) + self.pos[hk]
        sh = shingles(n[:NEAR_CHARS])
        if sh:
            for key in self.lsh.query(mh(sh)):
                t = self.pos_sh[key]
                if len(sh & t) / len(sh | t) >= JACCARD:
                    task, ref = key.split("\t")
                    return ("doc_near", task, ref[4:]) if ref.startswith("doc:") else ("pos_near", task, ref)
        return self.contain(n)


def corpus_records(task):
    repo, rev = task.metadata.dataset["path"], task.metadata.dataset.get("revision")
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    cfiles = [f for f in files if f == "corpus.jsonl" or (f.startswith("corpus/") and f.endswith(".parquet"))]
    yield from iter_records(repo, rev, cfiles)


def build() -> DocIndex:
    idx = DocIndex()
    counts = collections.Counter()
    for task in test_tasks():
        name = task.metadata.name
        full = name in FULL_CORPUS or (name.startswith("CQADupstack") and not name.startswith("Nano"))
        if full:
            gs = []
            for r in corpus_records(task):
                text = f"{r.get('title') or ''} {r.get('text') or ''}"
                if name in NEAR_FULL:
                    gs.append(grams8(norm(text)))
                idx.add_full(name, text)
                counts[f"full:{name}"] += 1
                if name in NEAR_FULL:
                    idx.add_near(name, rid(r), text)
                    for part in (r.get("title") or "", r.get("text") or ""):  # abstract alone, without its title
                        if part and len(norm(part)) > 50:
                            idx.add_full(name, part)
            if gs:
                idx.grams[name] = np.unique(np.concatenate(gs))
        c = load_test(task, docs=True)  # cached by decontam.py --docs
        for k, (docid, text300) in (c["pos_docs"] or {}).items():
            idx.add_positive(name, docid, text300, full_hash=k)
            counts[f"pos:{name}"] += 1
        print(f"  {name}: " + (f"{counts[f'full:{name}']} corpus docs, " if full else "")
              + f"{counts[f'pos:{name}']} positives", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    with open(INDEX, "wb") as f:
        pickle.dump(idx, f)
    print(f"index: {len(idx.full)} full-corpus docs, {len(idx.pos)} positives -> {INDEX}")
    return idx


class _Unpickler(pickle.Unpickler):
    """An index built by `python -m ...doc_decontam --build` pickles its class as __main__.DocIndex; resolve it here so
    other modules (synth.py pool) can load it too."""

    def find_class(self, module, name):
        if name == "DocIndex":
            return DocIndex
        return super().find_class(module, name)


def load() -> DocIndex:
    with open(INDEX, "rb") as f:
        return _Unpickler(f).load()


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check")
    ap.add_argument("--columns", nargs="+", default=["positives", "negatives"])
    ap.add_argument("--out")
    args = ap.parse_args()
    idx = build() if args.build else None
    if not args.check:
        return
    idx = idx or load()
    df = pd.read_parquet(args.check)
    hits, keep, cleaned_negs = collections.Counter(), [], []
    for i, r in enumerate(df.itertuples(index=False)):
        bad_pos = False
        for col in args.columns:
            vals = getattr(r, col)
            vals = [vals] if isinstance(vals, str) else list(vals)
            ok = []
            for v in vals:
                hit = idx.check(v)
                if hit:
                    hits[(col, hit[0], hit[1])] += 1
                    bad_pos |= col.startswith("pos")
                else:
                    ok.append(v)
            if col == "negatives":
                cleaned_negs.append(ok)
        keep.append(not bad_pos)
    for (col, rule, task), n in sorted(hits.items()):
        print(f"  {col:<10} {rule:<11} {task:<32} {n}")
    print(f"rows {len(df)}; rows with a flagged positive {keep.count(False)}")
    if args.out:
        if "negatives" in args.columns:
            df = df.assign(negatives=cleaned_negs)
        df[keep].to_parquet(args.out, index=False, row_group_size=20_000)
        with open(args.out + ".doc_decontam.json", "w", encoding="utf-8") as f:
            json.dump({"rows_in": len(df), "rows_out": int(sum(keep)),
                       "hits": [{"column": c, "rule": ru, "task": t, "n": n} for (c, ru, t), n in hits.items()]}, f, indent=1)
        print(f"kept {sum(keep)} rows -> {args.out}")


if __name__ == "__main__":
    main()
