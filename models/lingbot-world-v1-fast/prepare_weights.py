# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the LingBot-World v1 Fast weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 90 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml reactor-runtime
    python prepare_weights.py --output ./weights

The folder gets ``checkpoints/lingbot-world-base-cam/`` with the VAE, the text
encoder and the ``lingbot_world_fast/`` shards, and ``load()`` reads it
read-only. A second run downloads only the files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lingbot_world_v1_config import BASE_PATTERNS, check_weights, read_config

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download both pinned snapshots into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import snapshot_download

    config = read_config(config_path, output)
    snapshot_download(
        repo_id=config.checkpoint.repo_id,
        revision=config.checkpoint.revision,
        local_dir=config.checkpoint.path,
        allow_patterns=list(BASE_PATTERNS),
    )
    snapshot_download(
        repo_id=config.fast_checkpoint.repo_id,
        revision=config.fast_checkpoint.revision,
        local_dir=config.fast_checkpoint.path,
    )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument(
        "--config", type=Path, default=_MODEL_DIR / "lingbot_world_v1.yaml"
    )
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
