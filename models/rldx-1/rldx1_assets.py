# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""RLDX-1 weight sources, the weights folder layout, and its check.

The check only resolves and verifies files. It never downloads. Build the
weights folder with ``prepare_weights.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
FIX = f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."

_REVISION = re.compile(r"[0-9a-f]{40}")

# The finetuned policy: model config, weights, and the processor that holds the
# modality configs and normalization statistics.
CHECKPOINT_FILES = (
    "LICENSE.md",
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "processor/embodiment_id.json",
    "processor/processor_config.json",
    "processor/statistics.json",
)
# The base VLM that the checkpoint names as `model_name`: the backbone config
# and the Qwen3-VL tokenizer and processor files. Its weights are not read,
# because the checkpoint carries the whole backbone.
BASE_MODEL_FILES = (
    "LICENSE.md",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class WeightSource:
    """One pinned Hugging Face repository and the folder it fills."""

    repo_id: str
    revision: str
    path: Path
    files: tuple[str, ...]


@dataclass(frozen=True)
class RLDXWeights:
    """The two folders that ``load()`` reads from the weights root."""

    checkpoint: WeightSource
    base_model: WeightSource


def resolve_weights(config: dict[str, Any], weights_root: Path) -> RLDXWeights:
    """Resolve the ``checkpoint`` and ``base_model`` config sections under ``weights_root``."""
    root = Path(weights_root)
    return RLDXWeights(
        checkpoint=_source(config, "checkpoint", root, CHECKPOINT_FILES),
        base_model=_source(config, "base_model", root, BASE_MODEL_FILES),
    )


def check_weights(weights: RLDXWeights) -> None:
    """Raise ``FileNotFoundError`` naming the first missing file and the fix."""
    missing = [
        source.path / name
        for source in (weights.checkpoint, weights.base_model)
        for name in source.files
        if not (source.path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"RLDX-1 weights are incomplete: {missing[0]} is missing "
            f"({len(missing)} files). {FIX}"
        )


def _source(
    config: dict[str, Any], key: str, root: Path, files: tuple[str, ...]
) -> WeightSource:
    section = config.get(key)
    if not isinstance(section, dict):
        raise TypeError(
            f"config.yml needs a `{key}` section with repo_id, revision and dir"
        )
    repo_id = str(section.get("repo_id", ""))
    if repo_id.count("/") != 1:
        raise ValueError(f"{key}.repo_id must be a Hugging Face <org>/<repo> id")
    revision = str(section.get("revision", ""))
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"{key}.revision must be a full 40-character commit")
    folder = Path(str(section.get("dir", "")))
    if not folder.parts or folder.is_absolute() or ".." in folder.parts:
        raise ValueError(f"{key}.dir must be a relative folder inside the weights root")
    return WeightSource(repo_id, revision, root / folder, files)
