"""MTEB evaluation wrapper.

    uv run python -m olmo3_embed.eval --model BAAI/bge-base-en-v1.5 --suite repro
    uv run python -m olmo3_embed.eval --model Qwen/Qwen3-Embedding-0.6B --suite nano

Results land in results/<model>/<revision>/<Task>.json (same layout as embeddings-benchmark/results),
so scripts/compare_published.py can diff them against the leaderboard numbers.
"""
import argparse

import mteb

SUITES = {
    # Small full-size BEIR tasks that have published leaderboard scores: the Phase 0 reproduction check.
    "repro": ["SciFact", "NFCorpus", "ArguAna", "SCIDOCS", "FiQA2018"],
    # NanoBEIR (13 tasks, ~5k docs each): the inner-loop dev metric.
    "nano": [
        "NanoArguAnaRetrieval", "NanoClimateFeverRetrieval", "NanoDBPediaRetrieval", "NanoFEVERRetrieval",
        "NanoFiQA2018Retrieval", "NanoHotpotQARetrieval", "NanoMSMARCORetrieval", "NanoNFCorpusRetrieval",
        "NanoNQRetrieval", "NanoQuoraRetrieval", "NanoSCIDOCSRetrieval", "NanoSciFactRetrieval",
        "NanoTouche2020Retrieval",
    ],
    "beir": None,  # full 15-task benchmark (Colab)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", choices=SUITES, default="nano")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="override load precision")
    ap.add_argument("--tasks", nargs="+", help="explicit task names (overrides --suite)")
    args = ap.parse_args()

    if args.tasks:
        tasks = mteb.get_tasks(tasks=args.tasks, languages=["eng"])
    elif args.suite == "beir":
        tasks = mteb.get_benchmark("BEIR").tasks
    else:
        tasks = mteb.get_tasks(tasks=SUITES[args.suite], languages=["eng"])
    kwargs = {"model_kwargs": {"torch_dtype": args.dtype}} if args.dtype else {}
    model = mteb.get_model(args.model, **kwargs)  # uses the registered ModelMeta (prompts, pooling) when available
    mteb.evaluate(
        model,
        tasks,
        cache=mteb.ResultCache(args.out),
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="only-missing",
    )


if __name__ == "__main__":
    main()
