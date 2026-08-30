"""Config parsing and weights-bundle validation for FastH3.

``fasth3.yaml`` is read here and nowhere else: ``load_config`` turns it into
one validated :class:`FastH3Config`, and ``require_weights`` fails startup
loudly when the bundle on disk is incomplete. Pure file and dict work — no
torch, no fastvideo — so the schema renders and the tests run on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import fasth3_clip_plan as clip_plan

# The HF snapshot directory inside the weights bundle.
DEFAULT_CHECKPOINT_DIR = "FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"

# Component directories the T2VA pipeline loads. An incomplete bundle must kill
# startup, not surface as a loader traceback on the first clip.
REQUIRED_COMPONENTS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


@dataclass(frozen=True)
class FastH3Config:
    """Everything ``fasth3.yaml`` configures, validated once at load.

    The session-level fields are the defaults a fresh session starts from and
    the queue's fixed capacity. ``inference`` and ``runtime`` are the raw
    blocks; the backend reads its engine knobs (attention kernels, compile
    flags, parallelism, offload policy) straight from them.
    """

    aspect: str
    clip_frames: int
    seed: int
    num_inference_steps: int
    queue_size: int
    warmup_aspects: tuple[str, ...]
    inference: dict[str, Any]
    runtime: dict[str, Any]


def load_config(config_path: Path | None) -> FastH3Config:
    """Parse ``fasth3.yaml`` into a validated :class:`FastH3Config`.

    Args:
        config_path: Path the runtime hands over from ``runtime.config`` in
            ``reactor.yaml``, or ``None`` when the manifest names no config.

    Raises:
        ValueError: If the configured aspect is not one this model offers, or
            the queue size is not positive.
    """
    document: dict[str, Any] = {}
    if config_path is not None:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inference: dict[str, Any] = document.get("inference") or {}
    runtime: dict[str, Any] = document.get("runtime") or {}

    aspect = str(inference.get("aspect", "16:9"))
    if aspect not in clip_plan.ASPECT_CHOICES:
        raise ValueError(
            f"inference.aspect must be one of {list(clip_plan.ASPECT_CHOICES)}, got {aspect!r}"
        )

    queue_size = int(inference.get("queue_size", 10))
    if queue_size < 1:
        raise ValueError(f"inference.queue_size must be positive, got {queue_size}")

    return FastH3Config(
        aspect=aspect,
        clip_frames=clip_plan.frames_for_seconds(
            float(inference.get("clip_seconds", clip_plan.MAX_SECONDS))
        ),
        seed=int(inference.get("seed", 1000)),
        # Sigma-grid POINTS, not transformer forwards: the distilled schedule is
        # five points and exactly four forwards.
        num_inference_steps=int(inference.get("num_inference_steps", 5)),
        queue_size=queue_size,
        warmup_aspects=tuple(str(a) for a in (inference.get("warmup_aspects") or [aspect])),
        inference=inference,
        runtime=runtime,
    )


def resolve_model_path(config: FastH3Config, weights_root: Path) -> Path:
    """The checkpoint directory inside the mounted weights bundle."""
    return weights_root / str(config.runtime.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))


def require_weights(root: Path, model_path: Path) -> None:
    """Fail startup loudly when the weights bundle is incomplete."""
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in REQUIRED_COMPONENTS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")
    if problems:
        raise FileNotFoundError(
            f"FastH3 weights bundle under {root} is incomplete:\n  " + "\n  ".join(problems)
        )


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "REQUIRED_COMPONENTS",
    "FastH3Config",
    "load_config",
    "require_weights",
    "resolve_model_path",
]
