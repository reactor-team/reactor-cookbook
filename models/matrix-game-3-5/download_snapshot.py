#!/usr/bin/env python3
"""Download one pinned Hugging Face snapshot for the Matrix adapter."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

_SNAPSHOT_MARKER = ".reactor-snapshot.json"


def _parser() -> argparse.ArgumentParser:
    """Return the snapshot downloader argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--ignore-pattern", action="append", default=[])
    return parser


def main() -> None:
    """Download the requested immutable snapshot into its canonical directory."""
    args = _parser().parse_args()
    snapshot_download = importlib.import_module("huggingface_hub").snapshot_download

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        ignore_patterns=args.ignore_pattern or None,
    )
    marker = args.local_dir / _SNAPSHOT_MARKER
    pending = marker.with_suffix(".tmp")
    pending.write_text(
        json.dumps({"repo_id": args.repo_id, "revision": args.revision}, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(pending, marker)


if __name__ == "__main__":
    main()
