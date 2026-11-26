# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the SolarWM Stage2 weights folder from the pinned Hugging Face revision.

Run it once, on any machine with about 55 GB free, before ``reactor run`` or
``reactor deploy``. ``junchaoh-cs/SolarWM`` is gated: accept its terms on the
model card, then pass a token that can read it::

    pip install huggingface-hub pyyaml
    HF_TOKEN=<your token> python prepare_weights.py --output ./weights

The folder holds the files that Stage2 inference reads, at their repository
paths: the UMT5 text encoder, its tokenizer and the Wan2.2 VAE from
``SolarWM-5B-base``, and the ``SolarWM-5B-sgf-stage2-81f`` checkpoint. Stage2
loads the whole transformer from its checkpoint, so the script skips the base
transformer shards. ``load()`` reads the folder read-only. A second run
downloads only the files that are missing or empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solarwm_stage2_config import (
    ALLOW_PATTERNS,
    REQUIRED_FILES,
    check_weights,
    read_config,
)

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import snapshot_download

    config = read_config(config_path, output)
    # The hub can take an interrupted, empty file as current; fetch it again.
    for name in REQUIRED_FILES:
        path = config.weights_root / name
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
    snapshot_download(
        repo_id=config.repo_id,
        revision=config.checkpoint_revision,
        local_dir=config.weights_root,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument(
        "--config", type=Path, default=_MODEL_DIR / "solarwm_stage2.yaml"
    )
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
