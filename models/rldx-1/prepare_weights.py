# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the RLDX-1 weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 14 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub==0.36.2 pyyaml
    python prepare_weights.py --output ./weights

The folder holds ``RLDX-1-FT-ROBOCASA`` (the finetuned policy) and
``RLDX-1-VLM`` (the config, tokenizer and processor files of the base model
that the policy names; its weights are not needed). ``config.yml`` pins both
repositories by commit. Both are public, so ``HF_TOKEN`` is optional.
``load()`` reads the folder read-only. A second run downloads only the files
that are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rldx1_assets import check_weights, resolve_weights

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    weights = resolve_weights(config, output)
    for source in (weights.checkpoint, weights.base_model):
        # `local_dir` writes plain files; `reactor deploy` refuses symlinks.
        for name in source.files:
            hf_hub_download(
                repo_id=source.repo_id,
                revision=source.revision,
                filename=name,
                local_dir=source.path,
            )
    check_weights(weights)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "config.yml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
