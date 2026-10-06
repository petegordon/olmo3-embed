#!/usr/bin/env bash
# Append results/ (+ run logs) to the colab-results branch WITHOUT touching the working branch or index.
# Safe to call after every run: the branch accumulates (old files kept, new/changed files added; --ignore-removal
# matters: a fresh runtime has only its own results/, and plain `git add` would stage every other file as deleted).
#   scripts/push_results.sh "message" [--dry-run]
# Run logs: for every runs/<name>/ with embed_config.json, copies config.yaml, embed_config.json, log.jsonl into
# results/run_logs/<name>/ (adapters stay on Drive; they're too big for git).
set -euo pipefail
cd "$(dirname "$0")/.."
msg=${1:-"results"}; dry=${2:-}
branch=colab-results

for d in runs/*/; do
  [ -f "$d/embed_config.json" ] || continue
  name=$(basename "$d"); mkdir -p "results/run_logs/$name"
  cp "$d"/{config.yaml,embed_config.json,log.jsonl} "results/run_logs/$name/" 2>/dev/null || true
done

git fetch -q origin "$branch" 2>/dev/null && parent=$(git rev-parse "origin/$branch") || parent=$(git rev-parse HEAD)
export GIT_INDEX_FILE=$(mktemp -u)
git read-tree "$parent"
git add -f --ignore-removal results  # never stage deletions: a runtime only has its own results/, not the branch's
tree=$(git write-tree)
rm -f "$GIT_INDEX_FILE"; unset GIT_INDEX_FILE
if [ "$tree" = "$(git rev-parse "$parent^{tree}")" ]; then echo "nothing new to push"; exit 0; fi
commit=$(git -c user.email=colab@olmo3-embed -c user.name=colab commit-tree "$tree" -p "$parent" -m "$msg")
if [ "$dry" = "--dry-run" ]; then
  echo "dry run: would push $commit to $branch"; git diff --stat "$parent" "$commit" | tail -5
else
  git push -q origin "$commit:refs/heads/$branch" && echo "pushed $commit to $branch"
fi
