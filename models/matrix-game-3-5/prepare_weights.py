# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Matrix-Game-3.5 weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 60 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml numpy
    python prepare_weights.py --output ./weights

The folder gets ``checkpoints/`` with the distilled checkpoint, the Wan2.2
TI2V 5B model and the Depth-Anything-3 model, and ``load()`` reads it
read-only. A second run downloads only the files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matrix_game_3_5_config import check_weights, read_config

_MODEL_DIR = Path(__file__).resolve().parent
# Files that the Wan repository carries for its model card, not for loading.
_WAN_CARD_ONLY = ["assets/*", "examples/*"]


def prepare(config_path: Path, output: Path) -> None:
    """Download the three pinned snapshots into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import snapshot_download

    config = read_config(config_path, output)
    snapshot_download(
        repo_id=config.checkpoint.repo_id,
        revision=config.checkpoint.revision,
        local_dir=config.checkpoint.path.parent,
    )
    snapshot_download(
        repo_id=config.wan.repo_id,
        revision=config.wan.revision,
        local_dir=config.wan.path,
        ignore_patterns=_WAN_CARD_ONLY,
    )
    snapshot_download(
        repo_id=config.depth.repo_id,
        revision=config.depth.revision,
        local_dir=config.depth.path,
    )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument(
        "--config", type=Path, default=_MODEL_DIR / "matrix_game_3_5.yaml"
    )
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
