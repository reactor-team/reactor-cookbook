# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the Matrix-Game-4 weights folder from the pinned release manifests.

Run it once, on any machine with about 90 GB free, before ``reactor run`` or
``reactor deploy``. The upstream downloader copies each file out of the Hugging
Face cache, so the cache holds a second copy; set ``HF_HUB_CACHE`` to a
scratch folder to choose where it goes::

    pip install huggingface-hub numpy pyyaml
    python prepare_weights.py --output ./weights

It fetches the pinned upstream source for its release manifests, then downloads
every scene's resources from the pinned Hugging Face revision and checks each
file's size and SHA256. ``load()`` verifies the same files and never downloads.
A second run downloads only the files that are missing or changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

from matrix_game4_assets import (
    MANIFESTS,
    RESOURCE_REPO,
    RESOURCE_REVISION,
    SCENES,
    SOURCE_REVISION,
    SOURCE_URL,
)


def fetch_source(target: Path) -> Path:
    """Fetch the pinned upstream source into ``target`` and return its Matrix-Game-4 folder."""
    for command in (
        ["git", "init", str(target)],
        [
            "git",
            "-C",
            str(target),
            "fetch",
            "--depth",
            "1",
            SOURCE_URL,
            SOURCE_REVISION,
        ],
        ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"],
    ):
        subprocess.run(command, check=True)
    return target / "Matrix-Game-4"


def _load_resources(source: Path) -> ModuleType:
    """Import the upstream manifest downloader from its file."""
    spec = importlib.util.spec_from_file_location(
        "matrix_game_resources", source / "matrix_game/resources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(output: Path, source: Path) -> None:
    """Download every scene into ``output/resources``, then verify it as ``load()`` does."""
    resources = _load_resources(source)
    for action in ("download", "verify"):
        for manifest in MANIFESTS:
            for scene in SCENES:
                arguments = [
                    action,
                    "--manifest",
                    str(source / "releases" / f"{manifest}.json"),
                    "--scene",
                    scene,
                    "--resources",
                    str(output / "resources"),
                ]
                if action == "download":
                    arguments += [
                        "--repo-id",
                        RESOURCE_REPO,
                        "--repo-type",
                        "model",
                        "--revision",
                        RESOURCE_REVISION,
                    ]
                resources.main(arguments)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "weights"
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Matrix-Game-4 folder of a checkout at the pinned revision; fetched when omitted",
    )
    args = parser.parse_args(argv)
    if args.source:
        prepare(args.output, args.source)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            prepare(args.output, fetch_source(Path(scratch) / "source"))
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
