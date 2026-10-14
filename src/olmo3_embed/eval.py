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
    # Held-out task shapes at full query count (Nano has ~50 queries/task: too noisy for +1-point data changes).
    # ClimateFEVER/DBPedia use MTEB's HardNegatives corpora (same queries/labels, top-250 candidates per query); .v2
    # only adds a query prompt (v1 has none), so instruction-free models score the same on v1 and v2.
    # Touche2020 has only 49 test queries even in full, so it stays on Nano.
    "heldout": ["ArguAna", "SCIDOCS", "ClimateFEVERHardNegatives.v2", "DBPediaHardNegatives.v2"],
    "beir": None,  # full 15-task benchmark (Colab)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id (not needed with --run)")
    ap.add_argument("--suite", choices=SUITES, default="nano")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="override load precision")
    ap.add_argument("--tasks", nargs="+", help="explicit task names (overrides --suite)")
    ap.add_argument("--max-seq-length", type=int,
                    help="cap tokens per text. On Windows, VRAM overflow silently spills into system RAM "
                         "(driver sysmem fallback) instead of raising OOM, so cap long-doc tasks like Touche2020")
    ap.add_argument("--run", help="trained run dir (runs/<name>): loads its adapter + embed_config.json")
    ap.add_argument("--olmo-pooling", choices=["mean", "last"],
                    help="use our OlmoEmbedder (model.py) instead of mteb's registered loader")
    ap.add_argument("--no-instructions", action="store_true", help="OlmoEmbedder: no query instructions")
    ap.add_argument("--load-in-4bit", action="store_true", help="OlmoEmbedder: bitsandbytes nf4 (local 7B smoke tests)")
    ap.add_argument("--run-instructions", choices=["on", "off"],
                    help="--run: override the run's own instruction setting (diagnostics); results get a name suffix")
    ap.add_argument("--name-suffix", help="--run: results folder olmo3-embed__<run_name><suffix> (keeps variants apart)")
    ap.add_argument("--token-budget", type=int,
                    help="OlmoEmbedder: batch by padded tokens instead of --batch-size rows (faster on short docs)")
    ap.add_argument("--doc-cache", help="OlmoEmbedder: float16 document-embedding cache file on local disk, shared by "
                                        "the tasks of this process (FEVER + ClimateFEVER share a corpus)")
    ap.add_argument("--doc-cache-capacity", type=int, default=6_000_000, help="rows in --doc-cache")
    ap.add_argument("--shard", help="I/N: search only the I-th of N corpus slices and save its top-k to --shard-out "
                                     "(olmo3_embed.shards; merge with scripts/merge_shards.py)")
    ap.add_argument("--shard-out", help="--shard: gzipped JSON of the shard's per-query top-k")
    ap.add_argument("--predictions", help="save per-query rankings here (mteb prediction_folder), e.g. for judged@10")
    args = ap.parse_args()

    if args.tasks:
        tasks = mteb.get_tasks(tasks=args.tasks, languages=["eng"])
    elif args.suite == "beir":
        tasks = mteb.get_benchmark("BEIR").tasks
    else:
        tasks = mteb.get_tasks(tasks=SUITES[args.suite], languages=["eng"])
    if args.run:
        from olmo3_embed.model import OlmoEmbedder

        import json as _json
        import os as _os

        instr = None if args.run_instructions is None else args.run_instructions == "on"
        suffix = args.name_suffix or (f"-instr{args.run_instructions}" if args.run_instructions else "")
        run_name = None
        if suffix:
            run_name = _json.load(open(_os.path.join(args.run, "embed_config.json")))["run_name"] + suffix
        model = OlmoEmbedder.from_run(
            args.run, max_length=args.max_seq_length or 512, batch_size=args.batch_size, dtype=args.dtype,
            instructions=instr, run_name=run_name, token_budget=args.token_budget,
        )
    elif args.olmo_pooling:
        from olmo3_embed.model import OlmoEmbedder

        model = OlmoEmbedder(
            args.model, pooling=args.olmo_pooling, instructions=not args.no_instructions,
            dtype=args.dtype or "bfloat16", max_length=args.max_seq_length or 512, load_in_4bit=args.load_in_4bit,
            batch_size=args.batch_size, token_budget=args.token_budget,
        )
    else:
        kwargs = {"model_kwargs": {"torch_dtype": args.dtype}} if args.dtype else {}
        if args.max_seq_length:
            kwargs["max_seq_length"] = args.max_seq_length
        model = mteb.get_model(args.model, **kwargs)  # registered ModelMeta (prompts, pooling) when available
    if args.doc_cache:
        from olmo3_embed.model import DocEmbeddingCache

        model.doc_cache = DocEmbeddingCache(args.doc_cache, model.model.config.hidden_size, args.doc_cache_capacity)
    if args.shard:
        from olmo3_embed.shards import ShardSearch

        i, n = map(int, args.shard.split("/"))
        model = ShardSearch(model, i, n, args.shard_out)
    mteb.evaluate(
        model,
        tasks,
        cache=mteb.ResultCache(args.out),
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="only-missing",
        **({"prediction_folder": args.predictions} if args.predictions else {}),
    )


    if args.doc_cache:
        print(f"doc cache: {model.doc_cache.hits} hits, {model.doc_cache.misses} misses", flush=True)


if __name__ == "__main__":
    main()
