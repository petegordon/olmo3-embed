#!/usr/bin/env bash
# Synthetic pilot driver for one subscription CLI (docs/task-variety-plan.md): generates queries for its own disjoint
# quarter of every shape's passages, shape by shape, waiting until each passage file is complete (8000 lines).
#   scripts/run_synth_cli.sh claude 0 claude-opus-5-5     # parts: 0 claude, 1 codex, 2 gemini, 3 OLMo (Colab)
# Resumable: rerun the same command and finished ids are skipped.
set -u
backend=$1; part=$2; model=${3:-}
cd "$(dirname "$0")/.."
batch=8
for shape in argument counterargument citation entity claim; do
  f=data/synth/passages_$shape.jsonl
  until [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 8000 ]; do sleep 60; done
  echo "=== $backend part $part: $shape ($(date +%H:%M))"
  uv run python -m olmo3_embed.data.synth generate --passages "$f" --backend "$backend" ${model:+--model "$model"} \
    --part "$part" --parts 4 --batch $batch --out "data/synth/${shape}_${backend}.jsonl" 2>&1 | grep -v -i warning | tail -2
done
echo "=== $backend done ($(date +%H:%M))"
