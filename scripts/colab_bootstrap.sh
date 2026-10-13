#!/usr/bin/env bash
# Colab runtime setup without Drive: clone/update the repo into /content/olmo3-embed, install it (keeping Colab's CUDA
# torch), and fetch a published adapter release into runs/.  Usage: bash colab_bootstrap.sh [RELEASE_TAG RUN_NAME]
set -euo pipefail
R=/content/olmo3-embed
if [ -d $R/.git ]; then git -C $R checkout -q main && git -C $R pull -q; else git clone -q https://github.com/petegordon/olmo3-embed.git $R; fi
cd $R
pip -q install uv && uv pip install --system -q --no-sources -e . 2>&1 | tail -2
pip uninstall -y -q torchao 2>/dev/null || true  # Colab's torchao 0.10 blocks peft >= 0.21
if [ $# -ge 2 ] && [ ! -f runs/$2/adapter/adapter_model.safetensors ]; then
  mkdir -p runs && curl -sSfL -o /tmp/$2.zip https://github.com/petegordon/olmo3-embed/releases/download/$1/$2.zip
  unzip -q -o /tmp/$2.zip -d runs/
fi
git log --oneline -1
