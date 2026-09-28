# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Parse ``joy_video_edit.yaml`` and resolve the pinned checkpoints under the weights root.

First load downloads the pinned Hugging Face revisions into the weights root; a marker per
snapshot records the revision it holds, so later loads skip the network. ``checkpoint_dir``
points at a directory that already holds the same layout instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Files the text-encoder repository carries for its model card, not for loading.
_CARD_ONLY = ["*.gif", "*.png", "*.jpeg", "*.jpg"]


@dataclass(frozen=True)
class Checkpoints:
    """Where the three checkpoints are on disk."""

    dit: Path
    vae: Path
    text_encoder: Path


def read_config(config_path: Path | None) -> dict[str, Any]:
    """Parse ``joy_video_edit.yaml``; no path means the defaults."""
    if config_path is None:
        return {}
    document = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    return document


def resolve_checkpoints(config: dict[str, Any], weights_root: Path | str) -> Checkpoints:
    """Return the checkpoint paths, downloading the pinned snapshots first when they are missing.

    Raises:
        ValueError: The ``checkpoints`` section is missing a field, or ``checkpoint_dir`` is relative.
        FileNotFoundError: A checkpoint is absent after resolution.
    """
    section = _mapping(config.get("checkpoints"), "checkpoints")
    joyai = _mapping(section.get("joyai"), "checkpoints.joyai")
    text_encoder = _mapping(section.get("text_encoder"), "checkpoints.text_encoder")
    dit_rel = _field(joyai, "dit", "checkpoints.joyai")
    vae_rel = _field(joyai, "vae", "checkpoints.joyai")
    te_rel = _field(text_encoder, "path", "checkpoints.text_encoder")

    fixed = config.get("checkpoint_dir")
    if fixed:
        root = Path(fixed).expanduser()
        if not root.is_absolute():
            raise ValueError(f"checkpoint_dir must be an absolute path, got {fixed!r}")
    else:
        root = Path(weights_root)
        _download(
            _field(joyai, "repo_id", "checkpoints.joyai"),
            _field(joyai, "revision", "checkpoints.joyai"),
            root,
            allow_patterns=[dit_rel, f"{vae_rel}/*"],
        )
        _download(
            _field(text_encoder, "repo_id", "checkpoints.text_encoder"),
            _field(text_encoder, "revision", "checkpoints.text_encoder"),
            root / te_rel,
            ignore_patterns=_CARD_ONLY,
        )

    checkpoints = Checkpoints(dit=root / dit_rel, vae=root / vae_rel, text_encoder=root / te_rel)
    for name, path in (("DiT", checkpoints.dit), ("VAE", checkpoints.vae), ("text encoder", checkpoints.text_encoder)):
        if not path.exists():
            raise FileNotFoundError(f"JoyAI-Video-Edit {name} checkpoint not found at {path}")
    return checkpoints


def _download(
    repo_id: str,
    revision: str,
    local_dir: Path,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    """Fetch one pinned snapshot into ``local_dir`` unless its marker already records the revision."""
    marker = local_dir / ".revisions" / repo_id.replace("/", "--")
    if marker.is_file() and marker.read_text().strip() == revision:
        return
    from huggingface_hub import snapshot_download

    logger.info("downloading %s@%s into %s", repo_id, revision, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(revision + "\n")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"joy_video_edit.yaml: {name} must be a mapping")
    return value


def _field(mapping: dict[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"joy_video_edit.yaml: {name}.{key} must be a non-empty string")
    return value
