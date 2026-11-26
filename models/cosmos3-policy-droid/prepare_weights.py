# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Cosmos3-Policy-DROID weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 13 GB free for Edge (37 GB for Nano),
before ``reactor run`` or ``reactor deploy``. Both repositories are public, so
no token is needed::

    pip install huggingface-hub pyyaml
    python prepare_weights.py --output ./weights

The folder holds ``<repo_id>/`` of the checkpoint ``cosmos3_policy_droid.yaml``
names, without its model card, media, and diffusers copy of the video
tokenizer, and ``Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth``, the video tokenizer
the checkpoint loads. Every file is plain, with no cache symlinks. ``load()``
reads the folder read-only. A second run downloads only the files that are
missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cosmos3_policy_droid_assets import (
    CHECKPOINT_IGNORE_PATTERNS,
    VAE,
    VAE_FILENAME,
    check_weights,
    checkpoint_dir,
    read_config,
    vae_path,
)

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download, snapshot_download

    config = read_config(config_path)
    snapshot_download(
        repo_id=config.checkpoint.repo_id,
        revision=config.checkpoint.revision,
        local_dir=checkpoint_dir(output, config.checkpoint),
        ignore_patterns=list(CHECKPOINT_IGNORE_PATTERNS),
    )
    hf_hub_download(
        repo_id=VAE.repo_id,
        revision=VAE.revision,
        filename=VAE_FILENAME,
        local_dir=vae_path(output).parent,
    )
    check_weights(config, output)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "cosmos3_policy_droid.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
