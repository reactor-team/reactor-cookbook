# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the WorldPlay2 weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 82 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub numpy pyyaml pillow
    python prepare_weights.py --output ./weights

The folder holds ``base/`` (Wan2.2 text encoder, VAE, tokenizer and configs) and
``fast/`` (the two few-step expert files). ``load()`` reads it read-only. A second
run downloads only the files that are missing or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldplay2_model import (
    BASE_FILES,
    BASE_REPO,
    BASE_REVISION,
    FAST_FILES,
    FAST_REPO,
    FAST_REVISION,
    check_weights,
)


def prepare(output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    plan = (
        (BASE_REPO, BASE_REVISION, "base", BASE_FILES),
        (FAST_REPO, FAST_REVISION, "fast", FAST_FILES),
    )
    for repo_id, revision, folder, names in plan:
        for name in names:
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=name,
                local_dir=output / folder,
            )
    check_weights(output)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "weights"
    )
    args = parser.parse_args(argv)
    prepare(args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
