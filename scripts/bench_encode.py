"""Document-encoding throughput and correctness check for full-BEIR planning (Colab GPU).

    python scripts/bench_encode.py --run runs/p7-run4-7b [--n 20000]

Samples real corpora (MS MARCO, FEVER, Touche2020: short, medium, long), times the fixed batch-size path against
token-budget batching, checks the two give the same embeddings, checks the document cache and the GPU similarity.
"""
import argparse, itertools, json, time

import numpy as np
import torch
from datasets import load_dataset

from olmo3_embed.model import DocEmbeddingCache, OlmoEmbedder

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--budgets", default="32768,65536,131072")
ap.add_argument("--out", default="bench_encode.json")
args = ap.parse_args()


def sample(path, n):
    ds = load_dataset(path, "corpus", split="corpus", streaming=True)
    return [((r.get("title") or "") + " " + r["text"]).strip() for r in itertools.islice(ds, n)]


gpu = torch.cuda.get_device_name(0)
m = OlmoEmbedder.from_run(args.run, max_length=512, batch_size=64)
res = {"gpu": gpu, "rows": []}
for name, path, n in [("MSMARCO", "mteb/msmarco", args.n), ("FEVER", "mteb/fever", args.n), ("Touche2020", "mteb/touche2020", args.n // 4)]:
    texts = sample(path, n)
    chars = sum(map(len, texts))
    m._encode_texts(texts[:256])  # warm-up
    for mode in ["fixed64"] + [f"budget{b}" for b in args.budgets.split(",")]:
        m.token_budget = None if mode == "fixed64" else int(mode[6:])
        m.batch_size = 64
        torch.cuda.synchronize(); t = time.time()
        emb = m._encode_texts(texts)
        torch.cuda.synchronize(); dt = time.time() - t
        row = {"corpus": name, "mode": mode, "docs": len(texts), "chars": chars, "sec": round(dt, 1),
               "docs_per_s": round(len(texts) / dt), "chars_per_s": round(chars / dt)}
        if mode == "fixed64":
            ref = emb
        else:
            row["max_abs_diff_vs_fixed"] = float(np.abs(emb - ref).max())
            row["min_cos_vs_fixed"] = float((emb * ref).sum(1).min())
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

# document cache: second pass must be all hits and return the stored (float16) embeddings
m.token_budget = 65536
m.doc_cache = DocEmbeddingCache("/content/doc_cache_test.npy", m.model.config.hidden_size, 10000)
from mteb.types import PromptType
batch = [{"text": texts[:2000]}]
e1 = m.encode(batch, task_metadata=None, hf_split="test", hf_subset="default", prompt_type=PromptType.document)
e2 = m.encode(batch, task_metadata=None, hf_split="test", hf_subset="default", prompt_type=PromptType.document)
res["cache"] = {"hits": m.doc_cache.hits, "misses": m.doc_cache.misses, "max_abs_diff": float(np.abs(e1 - e2).max())}
print("cache", res["cache"], flush=True)

# GPU similarity vs mteb's CPU cos_sim
from mteb.similarity_functions import cos_sim
q = ref[:500]
d = ref[500:5000]
res["similarity_max_abs_diff"] = float((m.similarity(q, d) - cos_sim(q, d)).abs().max())
print("similarity diff", res["similarity_max_abs_diff"], flush=True)
print("free disk GB", round(__import__("shutil").disk_usage("/content").free / 1e9), "| RAM GB",
      round(__import__("os").sysconf("SC_PAGE_SIZE") * __import__("os").sysconf("SC_PHYS_PAGES") / 1e9))
json.dump(res, open(args.out, "w"), indent=1)
print("BENCH_DONE")
