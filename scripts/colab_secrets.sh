#!/usr/bin/env bash
# Upload the GitHub token to a Colab CLI runtime as /root/.olmo3_GH_TOKEN (cell 1's secret() reads it).
# Colab secrets (userdata) only work from the Colab UI. `colab exec --env` would log the value locally, a file
# upload doesn't. The token lives in the macOS Keychain:
#   security add-generic-password -a petegordon -s olmo3-colab-gh -w
set -euo pipefail
session=${1:?usage: colab_secrets.sh SESSION}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
chmod 600 "$tmp"
security find-generic-password -a petegordon -s olmo3-colab-gh -w | tr -d '\n' > "$tmp"
colab upload -s "$session" "$tmp" root/.olmo3_GH_TOKEN </dev/null | tail -1 | cut -c1-40
colab exec -s "$session" --timeout 30 <<<'import os; os.chmod("/root/.olmo3_GH_TOKEN", 0o600); print("GH_TOKEN file:", os.path.getsize("/root/.olmo3_GH_TOKEN") > 0)'
