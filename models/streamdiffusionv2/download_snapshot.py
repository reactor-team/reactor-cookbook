"""Download one pinned public Hugging Face snapshot into a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    """Download the requested revision and preserve resumable Hub metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--allow-pattern", action="append", default=[])
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=args.allow_pattern or None,
    )


if __name__ == "__main__":
    main()
