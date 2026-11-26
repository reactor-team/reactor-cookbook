# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Open-Oasis weights folder from the pinned Hugging Face revision.

Run it once, on any machine with about 4 GB free, before ``reactor run`` or
``reactor deploy``. ``Etched/oasis-500m`` is gated: accept its terms on the
model card, then pass a token that can read it::

    pip install huggingface-hub pyyaml
    HF_TOKEN=<your token> python prepare_weights.py --output ./weights

The folder holds the model and VAE checkpoints that ``load()`` reads, and not
the repository's ``.pt`` copies of them. ``load()`` reads the folder read-only.
A second run downloads only the files that are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from open_oasis_assets import read_config, resolve_checkpoints

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned checkpoints into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    config = read_config(config_path)
    for filename in (config.model_filename, config.vae_filename):
        hf_hub_download(
            repo_id=config.checkpoint_repo_id,
            revision=config.checkpoint_revision,
            filename=filename,
            local_dir=output,
        )
    resolve_checkpoints(config, output)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "open_oasis.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
