# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Lyra 2.0 weights folder from the pinned Hugging Face revisions.

Run it once, on any machine with about 140 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub pyyaml
    python prepare_weights.py --output ./weights

The folder holds ``source/Lyra-2/checkpoints`` (the Lyra checkpoints) and
``hf-cache/hub`` (the depth, text and MoGe repos, in the Hugging Face cache
layout). ``load()`` reads it read-only. A second run downloads only the files
that are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lyra2_assets import (
    CHECKPOINT_DIR,
    CHECKPOINT_FILES,
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    HUB_CACHE_DIR,
    HUB_REPOS,
    check_weights,
    snapshot_dir,
)


def prepare(output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download

    for name in CHECKPOINT_FILES:
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            revision=CHECKPOINT_REVISION,
            filename=f"checkpoints/{name}",
            local_dir=output / CHECKPOINT_DIR.removesuffix("checkpoints"),
        )
    for repo_id, (revision, names) in HUB_REPOS.items():
        for name in names:
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=name,
                cache_dir=output / HUB_CACHE_DIR,
            )
        # A pinned commit does not write `refs/main`, which offline loads need.
        refs = snapshot_dir(output, repo_id, revision).parents[1] / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)
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
