# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the StreamRender-H3 weights folder from the pinned Hugging Face revision.

Run it once, on any machine with about 274 GB free, before ``reactor run`` or
``reactor deploy``::

    pip install huggingface-hub
    python prepare_weights.py --output ./weights

It downloads the pinned bundle, then runs the bundle's ``download_assets.py``
to rebuild the merged backbone. ``load()`` reads the folder read-only. A second
run downloads only missing or changed files and skips a finished rebuild.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from streamrender_h3_assets import MERGED_BACKBONE, REPO_ID, REVISION, check_weights


def prepare(output: Path) -> None:
    """Download the pinned bundle into ``output`` and rebuild the merged backbone."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=output)
    if not (output / MERGED_BACKBONE).exists():
        subprocess.run([sys.executable, str(output / "download_assets.py")], check=True)
    check_weights(output / "assets.json")


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
