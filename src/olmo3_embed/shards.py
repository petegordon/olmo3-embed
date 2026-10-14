"""Corpus-sharded retrieval for corpora too big for one Colab runtime (MSMARCO: 8.8M passages, ~11.5 h on a G4,
and Colab ends runtimes at ~11.3 h).

Shard i of N embeds a contiguous 1/N slice of the corpus (in mteb's load order), runs the usual exact search over it
and saves each query's top-k (doc id -> score) to a gzipped JSON. MergedSearch then serves the union to mteb, which
keeps the best top-k per query: every document is scored exactly once against every query, so the global top-k (and
every metric) is the same as a single full-corpus search.
"""
from __future__ import annotations

import gzip
import heapq
import json

from mteb.models.model_meta import ModelMeta
from mteb.models.search_wrappers import SearchEncoderWrapper


class ShardSearch(SearchEncoderWrapper):
    def __init__(self, model, shard: int, n_shards: int, out_path: str):
        super().__init__(model)
        self.shard, self.n_shards, self.out_path = shard, n_shards, out_path
        # Its own results folder: the scores mteb computes over one shard mean nothing.
        self.mteb_model_meta = model.mteb_model_meta.model_copy(
            update={"name": f"{model.mteb_model_meta.name}-shard{shard}of{n_shards}"})

    def index(self, corpus, **kwargs):
        n = len(corpus)
        lo, hi = n * self.shard // self.n_shards, n * (self.shard + 1) // self.n_shards
        self.rows = hi - lo
        print(f"shard {self.shard}/{self.n_shards}: corpus rows {lo}:{hi} of {n}", flush=True)
        super().index(corpus.select(range(lo, hi)), **kwargs)

    def search(self, queries, **kwargs):
        results = super().search(queries, **kwargs)
        with gzip.open(self.out_path, "wt", encoding="utf-8") as f:
            json.dump({"shard": self.shard, "n_shards": self.n_shards, "rows": self.rows,
                       "top_k": kwargs["top_k"], "results": results}, f)
        print(f"saved {len(results)} queries -> {self.out_path}", flush=True)
        return results


class MergedSearch:
    """A SearchProtocol model that answers from saved shard results (no encoding)."""

    def __init__(self, meta: ModelMeta, paths: list[str]):
        self.mteb_model_meta = meta
        self.shards = [json.load(gzip.open(p, "rt", encoding="utf-8")) for p in paths]
        got = sorted(s["shard"] for s in self.shards)
        n = self.shards[0]["n_shards"]
        assert got == list(range(n)) and all(s["n_shards"] == n for s in self.shards), f"need shards 0..{n - 1}, got {got}"

    def index(self, corpus, **kwargs):
        rows = sum(s["rows"] for s in self.shards)
        assert rows == len(corpus), f"shards cover {rows} rows, corpus has {len(corpus)}"

    def search(self, queries, *, top_k: int, **kwargs):
        assert all(s["top_k"] >= top_k for s in self.shards)
        out = {}
        for qid in queries["id"]:
            pooled = [(score, did) for s in self.shards for did, score in s["results"].get(qid, {}).items()]
            out[qid] = {did: score for score, did in heapq.nlargest(top_k, pooled)}
        return out
