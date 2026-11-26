# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the AlayaWorld weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 65 GB free, before ``reactor run`` or
``reactor deploy``. ``google/gemma-3-12b-it-qat-q4_0-unquantized`` is gated:
accept its terms on the model card, then pass a token that can read it::

    pip install huggingface-hub pyyaml
    HF_TOKEN=<your token> python prepare_weights.py --output ./weights

The folder holds one directory per repository, named by the ``assets``
entries of ``alayaworld.yaml``: the merged AlayaWorld checkpoint, the Gemma 3
text encoder, and the Depth-Anything-3 checkpoint, each at its repository
paths. ``load()`` reads the folder read-only. A second run downloads only the
files that are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alayaworld_assets import check_weights, read_config

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    config = read_config(config_path, output)
    for asset in (config.model, config.gemma, config.da3_model):
        # `local_dir` writes plain files; `reactor deploy` refuses symlinks.
        for name in asset.files:
            hf_hub_download(
                repo_id=asset.repo_id,
                revision=asset.revision,
                filename=name,
                local_dir=asset.path,
            )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "alayaworld.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output.resolve())
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
