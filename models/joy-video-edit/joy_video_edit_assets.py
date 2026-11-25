# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Parse ``joy_video_edit.yaml`` and find the pinned checkpoints in the weights folder.

``prepare_weights.py`` builds the folder; this module only reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The command that builds a weights folder with this layout.
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"


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
    """Return the checkpoint paths under ``weights_root``.

    Raises:
        ValueError: The ``checkpoints`` section is missing a field.
        FileNotFoundError: A checkpoint is absent; the message names the command that builds the folder.
    """
    section = _mapping(config.get("checkpoints"), "checkpoints")
    joyai = _mapping(section.get("joyai"), "checkpoints.joyai")
    text_encoder = _mapping(section.get("text_encoder"), "checkpoints.text_encoder")
    root = Path(weights_root)
    checkpoints = Checkpoints(
        dit=root / _field(joyai, "dit", "checkpoints.joyai"),
        vae=root / _field(joyai, "vae", "checkpoints.joyai"),
        text_encoder=root / _field(text_encoder, "path", "checkpoints.text_encoder"),
    )
    for name, path in (("DiT", checkpoints.dit), ("VAE", checkpoints.vae), ("text encoder", checkpoints.text_encoder)):
        if not path.exists():
            raise FileNotFoundError(
                f"JoyAI-Video-Edit {name} checkpoint not found at {path}. "
                f"Build the weights folder with `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
            )
    return checkpoints


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"joy_video_edit.yaml: {name} must be a mapping")
    return value


def _field(mapping: dict[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"joy_video_edit.yaml: {name}.{key} must be a non-empty string")
    return value
