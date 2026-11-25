# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the EVOKE weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 70 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml
    python prepare_weights.py --output ./weights

The folder gets ``evoke-base/``, ``evoke/stage3_post_distillation/`` and
``ViGeo1.1/vigeo.pt``, and ``load()`` reads it read-only. A second run
downloads only the files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evoke_config import check_weights, read_config

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned snapshots into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import snapshot_download

    config = read_config(config_path, output)
    snapshot_download(
        repo_id=config.weights.repo_id,
        revision=config.weights.revision,
        local_dir=output,
        allow_patterns=["evoke-base/**", "evoke/stage3_post_distillation/**"],
    )
    snapshot_download(
        repo_id=config.vigeo.repo_id,
        revision=config.vigeo.revision,
        local_dir=config.vigeo_path,
        allow_patterns=["vigeo.pt"],
    )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "evoke.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
