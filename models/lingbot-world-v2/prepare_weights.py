# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the LingBot-World-V2 weights folder from the pinned Hugging Face revision.

Run it once, on any machine with about 90 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml
    python prepare_weights.py --output ./weights

The folder holds the pinned checkpoint repository at its root, plus the
``transformers/config.json`` copy that the upstream loader reads. ``load()``
reads the folder read-only. A second run downloads only the missing files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lingbot_world_v2_assets import check_checkpoint, read_config

_MODEL_DIR = Path(__file__).resolve().parent
# Files that the repository carries for its model card, not for loading.
_CARD_ONLY = ["README.md", "assets/*"]


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned snapshot into ``output``, then check it as ``load()`` does."""
    from huggingface_hub import snapshot_download

    config = read_config(config_path, output)
    snapshot_download(
        repo_id=config.checkpoint_repo_id,
        revision=config.checkpoint_revision,
        local_dir=output,
        ignore_patterns=_CARD_ONLY,
    )
    transformer_config = output / "transformers" / "config.json"
    if not transformer_config.is_file():
        shutil.copyfile(output / "config.json", transformer_config)
    check_checkpoint(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument(
        "--config", type=Path, default=_MODEL_DIR / "lingbot_world_v2.yaml"
    )
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
