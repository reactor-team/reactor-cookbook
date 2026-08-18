"""Download the paired public VPT sample used by the OpenDreamer demos."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


def _assets_module() -> Any:
    package = f"{__package__}.opendreamer_assets" if __package__ else "opendreamer_assets"
    return importlib.import_module(package)


def main() -> None:
    """Download the configured MP4 and its frame-aligned JSONL actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory; defaults to $OPENDREAMER_PATH/samples/vpt.",
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    assets = _assets_module()
    output_dir = arguments.output_dir or assets.upstream_demo_dir()
    video_path, actions_path = assets.ensure_demo_assets(
        output_dir,
        overwrite=arguments.overwrite,
    )
    print(f"video: {video_path}")
    print(f"actions: {actions_path}")


if __name__ == "__main__":
    main()
