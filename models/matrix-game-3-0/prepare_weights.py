# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Matrix-Game 3.0 weights folder from the pinned Hugging Face revision.

Run it once, on any machine with about 45 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml
    python prepare_weights.py --output ./weights

The folder holds the files that the model reads under
``checkpoints/Matrix-Game-3.0``: the distilled 5B DiT, the UMT5 encoder and
tokenizer, the Wan 2.2 VAE, and the MG-LightVAE v2 decoder. The repository also
holds the base model and a larger VAE, which this model does not read, so the
script skips them. ``load()`` reads the folder read-only. A second run
downloads only the missing files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matrix_game_3_0_assets import MODEL_ALLOW_PATTERNS, check_checkpoint, read_config

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import snapshot_download

    config = read_config(config_path, output)
    snapshot_download(
        repo_id=config.checkpoint_repo_id,
        revision=config.checkpoint_revision,
        local_dir=config.checkpoint_path,
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
    )
    check_checkpoint(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument(
        "--config", type=Path, default=_MODEL_DIR / "matrix_game_3_0.yaml"
    )
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
