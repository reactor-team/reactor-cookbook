# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the SANA-WM weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 125 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml
    python prepare_weights.py --output ./weights

The folder holds ``SANA-WM_streaming``, ``gemma-2-2b-it`` and ``Pi3X``, as
``sana_wm.yaml`` pins them. ``load()`` reads it read-only. A second run
downloads only the files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sana_wm_assets import (
    PI3X_REQUIRED_FILES,
    STREAMING_REQUIRED_FILES,
    TEXT_ENCODER_REQUIRED_FILES,
    check_weights,
    read_config,
)

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    config = read_config(config_path, output)
    plan = (
        (config.streaming, config.streaming_path, STREAMING_REQUIRED_FILES),
        (
            config.stage1_text_encoder,
            config.text_encoder_path,
            TEXT_ENCODER_REQUIRED_FILES,
        ),
        (config.pi3x_model, config.pi3x_path, PI3X_REQUIRED_FILES),
    )
    for asset, target, names in plan:
        for name in names:
            hf_hub_download(
                repo_id=asset.repo_id,
                revision=asset.revision,
                filename=name,
                local_dir=target,
            )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "sana_wm.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
