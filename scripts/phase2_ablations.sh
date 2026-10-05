#!/usr/bin/env bash
# Phase 2 ablation matrix on OLMo-2-1B (local). Each arm: train (resumable) then NanoBEIR eval (only-missing).
# Usage: scripts/phase2_ablations.sh [arm ...]   (no args = all arms, in order)
set -euo pipefail
cd "$(dirname "$0")/.."
declare -A ARMS=(
  [a0-base]=""
  [a1-last]="pooling=last"
  [a2-bidir]="bidirectional=true"
  [a3-bidir-last]="bidirectional=true pooling=last"
  [a4-neg1]="num_negatives=1"
  [a5-neg7]="num_negatives=7"
  [a6-noinstr]="instructions=false"
  [a7-tau05]="temperature=0.05"
)
ORDER=(a0-base a1-last a2-bidir a3-bidir-last a4-neg1 a5-neg7 a6-noinstr a7-tau05)
for arm in "${@:-${ORDER[@]}}"; do
  name="p2-$arm"
  echo "=== $name train $(date +%T)"
  uv run python -m olmo3_embed.train --config configs/proxy-base.yaml --set run_name=$name ${ARMS[$arm]}
  echo "=== $name eval $(date +%T)"
  uv run python -m olmo3_embed.eval --run runs/$name --suite nano --max-seq-length 512 --batch-size 32
done
echo "=== ALL DONE $(date +%T)"
