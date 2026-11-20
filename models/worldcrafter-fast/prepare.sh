#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export WORLDCRAFTER_ROOT="${WORLDCRAFTER_ROOT:-/local-ssd/worldcrafter-reactor}"
export WORLDCRAFTER_SOURCE="${WORLDCRAFTER_SOURCE:-$WORLDCRAFTER_ROOT/source}"
export WORLDCRAFTER_PYTHON="${WORLDCRAFTER_PYTHON:-$WORLDCRAFTER_ROOT/venv/bin/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORLDCRAFTER_ROOT/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORLDCRAFTER_ROOT/python}"
python3 prepare_source.py
command -v uv >/dev/null
export UV_PROJECT_ENVIRONMENT="$(dirname "$(dirname "$WORLDCRAFTER_PYTHON")")"
uv sync --project "$WORLDCRAFTER_SOURCE/uvenv" --frozen --extra demo --python 3.11.15
"$WORLDCRAFTER_PYTHON" -c 'import torch; print("WorldCrafter environment:",torch.__version__)'
