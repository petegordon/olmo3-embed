#!/usr/bin/env bash
# Upload a file to a GitHub release of this repo as an asset (replacing one of the same name). Needs GH_TOKEN.
#   bash scripts/upload_release_asset.sh TAG FILE
set -euo pipefail
tag=${1:?usage: upload_release_asset.sh TAG FILE}; file=${2:?usage: upload_release_asset.sh TAG FILE}
api=https://api.github.com/repos/petegordon/olmo3-embed; name=$(basename "$file")
auth=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
rel=$(curl -sSf "${auth[@]}" "$api/releases/tags/$tag")
rid=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$rel")
old=$(python3 -c 'import json,sys; print(" ".join(str(a["id"]) for a in json.load(sys.stdin)["assets"] if a["name"] == sys.argv[1]))' "$name" <<<"$rel")
for a in $old; do curl -sSf -X DELETE "${auth[@]}" "$api/releases/assets/$a"; done
curl -sSf -X POST "${auth[@]}" -H "Content-Type: application/octet-stream" --data-binary @"$file" \
  "https://uploads.github.com/repos/petegordon/olmo3-embed/releases/$rid/assets?name=$name" \
  | python3 -c 'import json,sys; a=json.load(sys.stdin); print("uploaded", a["name"], a["size"])'
