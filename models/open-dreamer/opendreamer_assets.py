"""Manage the separately distributed public VPT demo assets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json"
VERSION = "10.0"
STEM = "cheeky-cornflower-setter-02e496ce4abb-20220421-092639"
UPSTREAM_ENV = "OPENDREAMER_PATH"


def resolve_urls(index: dict[str, Any]) -> tuple[str, str]:
    """Return the paired sample URLs from the public VPT index."""
    basedir = index.get("basedir")
    relpaths = index.get("relpaths")
    if not isinstance(basedir, str) or not isinstance(relpaths, list):
        raise ValueError("The public VPT index has an unexpected format.")
    relative_stem = f"data/{VERSION}/{STEM}"
    video_path = f"{relative_stem}.mp4"
    actions_path = f"{relative_stem}.jsonl"
    if video_path not in relpaths:
        raise ValueError("The configured sample is absent from the public VPT index.")
    return (
        f"{basedir.rstrip('/')}/{video_path}",
        f"{basedir.rstrip('/')}/{actions_path}",
    )


def demo_paths(output_dir: Path) -> tuple[Path, Path]:
    """Return the expected MP4 and JSONL paths in one demo directory."""
    return output_dir / f"{STEM}.mp4", output_dir / f"{STEM}.jsonl"


def upstream_demo_dir() -> Path:
    """Return the demo directory inside the configured upstream checkout."""
    configured = os.environ.get(UPSTREAM_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {UPSTREAM_ENV} to the OpenDreamer repository checkout or pass --output-dir"
        )
    root = Path(configured).expanduser().resolve()
    if not (root / "dreamer").is_dir():
        raise RuntimeError(f"{UPSTREAM_ENV}={root} is not an OpenDreamer checkout")
    return root / "samples" / "vpt"


def ensure_demo_assets(output_dir: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Download missing paired VPT demo assets and return their local paths."""
    video_path, actions_path = demo_paths(output_dir)
    if not overwrite and video_path.is_file() and actions_path.is_file():
        return video_path, actions_path

    try:
        with urlopen(INDEX_URL) as response:
            index = json.load(response)
        if not isinstance(index, dict):
            raise ValueError("The public VPT index must contain a JSON object.")
        video_url, actions_url = resolve_urls(index)
        _download(video_url, video_path, overwrite=overwrite)
        _download(actions_url, actions_path, overwrite=overwrite)
    except URLError as error:
        raise RuntimeError(f"Could not download the public VPT demo: {error}") from error
    return video_path, actions_path


def _download(url: str, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        with urlopen(url) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r{destination.name}: {downloaded / total:.0%}",
                        end="",
                        flush=True,
                    )
            if total:
                print()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
