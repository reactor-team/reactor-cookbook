#!/usr/bin/env bash
# Build step: patch the pinned Matrix-Game-3.5 source and create its Python 3.10 worker.
# reactor.yaml runs it with this folder mounted at /build.
set -Eeuo pipefail
SOURCE=/opt/Matrix-Game-3.5
git -C "$SOURCE" apply /build/stateful_rollout.patch
uv venv --python 3.10 "$SOURCE/.venv"
uv pip install --python "$SOURCE/.venv/bin/python" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$SOURCE/.venv/bin/python" \
  --requirement "$SOURCE/requirements.txt" ftfy==6.3.1
