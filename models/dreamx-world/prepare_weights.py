# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the DreamX-World weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 25 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml reactor-runtime
    python prepare_weights.py --output ./weights

The folder gets the layout that ``dreamx_world.yaml``'s ``assets`` section
names, and ``load()`` reads it read-only. A second run downloads only the
files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamx_assets import WAN_REQUIRED_FILES, check_weights, read_config

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    config = read_config(config_path, output)
    hf_hub_download(
        repo_id=config.dreamx.repo_id,
        revision=config.dreamx.revision,
        filename=config.dreamx.path.name,
        local_dir=config.dreamx.path.parent,
    )
    for name in WAN_REQUIRED_FILES:
        hf_hub_download(
            repo_id=config.wan.repo_id,
            revision=config.wan.revision,
            filename=name,
            local_dir=config.wan.path,
        )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "dreamx_world.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
