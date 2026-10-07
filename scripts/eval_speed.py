"""Measure embedding throughput of a trained run (adapter merged, as in eval) to price larger BEIR evals.

    python scripts/eval_speed.py --run runs/p4-run1-7b [--n 10000] [--batch-sizes 16 64 128] [--max-len 512]

Samples real passages from two BEIR-15-lite corpora with different lengths (HotpotQA HardNegatives: Wikipedia
paragraphs; Touche-2020: long arguments), embeds them exactly like OlmoEmbedder.encode does (longest-first,
fixed batch size), and prints passages/s per batch size plus projected hours/units for BEIR-15 lite (~2.0M passages)
and full BEIR (~33M).
"""
import argparse
import os
import sys
import time

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from olmo3_embed.model import OlmoEmbedder  # noqa: E402

SOURCES = {"hotpotqa-hn": "mteb/HotpotQA_test_top_250_only_w_correct-v2", "touche2020": "mteb/touche2020"}
LITE_DOCS, FULL_DOCS = 2_025_000, 33_000_000  # docs/phase4-results.md


def sample(repo: str, n: int) -> list[str]:
    ds = load_dataset(repo, "corpus", split=None, streaming=True)
    split = next(iter(ds))
    out = []
    for r in ds[split].shuffle(seed=0, buffer_size=20_000):
        t = f"{r.get('title') or ''} {r.get('text') or ''}".strip()
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=10_000, help="passages per source")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--units-per-h", type=float, default=6.77)
    args = ap.parse_args()

    emb = OlmoEmbedder.from_run(args.run, max_length=args.max_len)
    gpu = torch.cuda.get_device_name(0)
    for name, repo in SOURCES.items():
        texts = sample(repo, args.n)
        lens = [len(x) for x in emb.tokenizer(texts, truncation=True, max_length=args.max_len)["input_ids"]]
        print(f"\n{name}: {len(texts)} passages, mean {sum(lens) / len(lens):.0f} tokens "
              f"(max {max(lens)}, {sum(l >= args.max_len for l in lens)} truncated at {args.max_len})", flush=True)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for bs in args.batch_sizes:
            try:
                emb.embed([texts[i] for i in order[:bs]])  # warm-up at the longest lengths (worst-case memory)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                for s in range(0, len(order), bs):
                    emb.embed([texts[i] for i in order[s : s + bs]])
                torch.cuda.synchronize()
                rate = len(texts) / (time.time() - t)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  batch {bs:>4}: OOM", flush=True)
                continue
            print(f"  batch {bs:>4}: {rate:6.1f} passages/s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  |  "
                  f"BEIR-15 lite ~{LITE_DOCS / rate / 3600:.1f} h ≈ {LITE_DOCS / rate / 3600 * args.units_per_h:.0f} units, "
                  f"full BEIR ~{FULL_DOCS / rate / 3600:.0f} h ≈ {FULL_DOCS / rate / 3600 * args.units_per_h:.0f} units "
                  f"({gpu})", flush=True)


if __name__ == "__main__":
    main()
