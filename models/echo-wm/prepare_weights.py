# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Echo-WM Flash weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 75 GB free, before ``reactor run`` or
``reactor deploy``. Gemma 3 is gated: accept its terms on the model card, then
pass a token that can read it::

    pip install huggingface-hub pyyaml
    HF_TOKEN=<your token> python prepare_weights.py --output ./weights

The folder holds ``echo-wm/echo-wm-flash.safetensors`` (the four-step Flash
checkpoint, without the repository's base checkpoint) and
``gemma-3-12b-it-qat-q4_0-unquantized/`` (the Gemma 3 text encoder, tokenizer,
and processor files). ``load()`` reads it read-only. A second run downloads
only the files that are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echo_wm_assets import GEMMA_FILES, check_weights, read_config

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download, snapshot_download

    config = read_config(config_path, output)
    checkpoint = config.checkpoint
    if checkpoint.filename is None:
        raise ValueError("assets.checkpoint.filename is required")
    hf_hub_download(
        repo_id=checkpoint.repo_id,
        revision=checkpoint.revision,
        filename=checkpoint.filename,
        local_dir=checkpoint.path.parent,
    )
    snapshot_download(
        repo_id=config.gemma.repo_id,
        revision=config.gemma.revision,
        local_dir=config.gemma.path,
        allow_patterns=list(GEMMA_FILES),
    )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "echo_wm.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
