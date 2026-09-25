"""DreamX-World causal rollout state and plain step contracts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dreamx_camera import CameraChunk

if TYPE_CHECKING:
    from dreamx_backend import DreamXBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepositoryAsset:
    """One pinned checkpoint repository and its local path."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class DreamXConfig:
    """Validated source, checkpoint, inference, and interaction settings."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    transformer_config: Path
    evaluation_inputs: Path
    random_images: tuple[Path, ...]
    dreamx: RepositoryAsset
    wan: RepositoryAsset
    seed: int
    motion_speed: float
    color_correction_strength: float
    max_chunks_per_rollout: int
    default_upload_prompt: str


@dataclass(frozen=True)
class DreamXAnchor:
    """Image and seed applied at the start of a world."""

    image: Path | bytes
    seed: int


@dataclass(frozen=True)
class DreamXInput:
    """One world identity, optional anchor, and native text and camera conditions."""

    world_id: int
    anchor: DreamXAnchor | None
    prompt: str
    poses: np.ndarray
    reference_pose: np.ndarray | None


@dataclass(frozen=True)
class DreamXResult:
    """Decoded frames and acknowledged progress in a causal world."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    prompt: str
    complete: bool


class NoAnchor(Exception):
    """A new world needs an image to seed its causal cache."""


class RolloutExhausted(Exception):
    """The rollout cannot continue past its configured chunk limit."""


class DreamXModel:
    """Own loaded weights and native rollout state independently of the runtime."""

    def __init__(self) -> None:
        self._backend: DreamXBackend | None = None
        self._config: DreamXConfig | None = None
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, config: DreamXConfig) -> None:
        """Load native weights from the application's resolved configuration."""
        from dreamx_backend import DreamXBackend

        self._config = config
        self._backend = DreamXBackend(config)
        logger.info(
            "DreamX ready: source=%s checkpoint=%s local_attention_frames=%s",
            config.source_revision,
            config.dreamx.revision,
            self._backend.local_attention_frames,
        )

    def generate(self, input: DreamXInput) -> DreamXResult:
        """Apply a requested world identity and generate one native camera chunk."""
        backend, config = self._backend, self._config
        if backend is None or config is None:
            raise RuntimeError("DreamX-World was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            backend.reset(input.anchor.seed, input.anchor.image)
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= config.max_chunks_per_rollout:
            raise RolloutExhausted(input.world_id)
        frames = backend.generate_chunk(
            input.prompt, CameraChunk(input.poses, input.reference_pose)
        )
        self._chunk_index += 1
        return DreamXResult(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            prompt=input.prompt,
            complete=self._chunk_index >= config.max_chunks_per_rollout,
        )

    def reset(self) -> None:
        """Release causal caches and progress while retaining loaded weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
