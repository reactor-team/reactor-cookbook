#!/usr/bin/env bash
# Build step: patch the pinned EVOKE source and create its Python 3.10 worker.
# reactor.yaml runs it with this folder mounted at /build.
set -Eeuo pipefail
SOURCE=/opt/Evoke
git -C "$SOURCE" apply /build/stateful_rollout.patch
uv venv --python 3.10 "$SOURCE/.reactor-venv"
uv pip install --python "$SOURCE/.reactor-venv/bin/python" \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
uv pip install --python "$SOURCE/.reactor-venv/bin/python" -r /build/worker-requirements.txt
