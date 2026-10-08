#!/usr/bin/env bash
# Experiment B: stance-flipped hard negatives for SynthCounterArgument rows, one subscription CLI per part.
#   scripts/run_negatives.sh claude 0 claude-opus-5-5     # parts: 0 claude, 1 codex (of 2); resumable
set -u
backend=$1; part=$2; model=${3:-}
cd "$(dirname "$0")/.."
uv run python -m olmo3_embed.data.synth negatives --data data/variety_v2b/filtered.parquet --backend "$backend" \
  ${model:+--model "$model"} --part "$part" --parts 2 --limit 800 --out "data/variety_v2b/neg_${backend}.jsonl" 2>&1 \
  | grep -v -i warning
echo "=== $backend done ($(date +%H:%M))"
