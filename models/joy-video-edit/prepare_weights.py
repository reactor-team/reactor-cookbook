# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the JoyAI-Video-Edit weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 50 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub==0.36.2 pyyaml
    python prepare_weights.py --output ./weights

The folder gets the layout that ``joy_video_edit.yaml``'s ``checkpoints``
section names, and ``load()`` reads it read-only. A second run downloads only
the files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from joy_video_edit_assets import read_config, resolve_checkpoints

_MODEL_DIR = Path(__file__).resolve().parent
# Files the text-encoder repository carries for its model card, not for loading.
_CARD_ONLY = ["*.gif", "*.png", "*.jpeg", "*.jpg"]


def prepare(config_path: Path, output: Path) -> None:
    """Download both pinned snapshots into ``output``, then check the result as ``load()`` does."""
    from huggingface_hub import snapshot_download

    section = read_config(config_path)["checkpoints"]
    joyai, text_encoder = section["joyai"], section["text_encoder"]
    snapshot_download(
        repo_id=joyai["repo_id"],
        revision=joyai["revision"],
        local_dir=output,
        allow_patterns=[joyai["dit"], f"{joyai['vae']}/*"],
    )
    snapshot_download(
        repo_id=text_encoder["repo_id"],
        revision=text_encoder["revision"],
        local_dir=output / text_encoder["path"],
        ignore_patterns=_CARD_ONLY,
    )
    resolve_checkpoints(read_config(config_path), output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "joy_video_edit.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
