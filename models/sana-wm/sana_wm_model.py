"""SANA-WM weights and native causal state behind plain step contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sana_wm_assets import SanaWMConfig

if TYPE_CHECKING:
    from sana_wm_backend import SanaStreamingBackend


@dataclass(frozen=True)
class SanaAnchor:
    """Image, calibration and text applied only when a world starts."""
    image: Path | bytes
    prompt: str
    seed: int
    intrinsics: Path | bytes | None
    trajectory_frames: int


@dataclass(frozen=True)
class SanaInput:
    """World identity and 24 camera-to-world poses; None marks trajectory end."""
    world_id: int
    anchor: SanaAnchor | None
    poses: np.ndarray | None


@dataclass(frozen=True)
class SanaResult:
    """Native frames and acknowledged model progress, independent of the input."""
    frames: np.ndarray
    world_id: int
    prompt: str
    chunk_index: int


class NoAnchor(Exception):
    """The requested world needs an image and conditioning."""


class RolloutExhausted(Exception):
    """The native streaming rollout reached its configured capacity."""


class TrajectoryCompleteError(Exception):
    """A finite trajectory has no complete native chunk left."""

    def __init__(self, chunk_index: int, trajectory_frames: int):
        super().__init__("The finite trajectory has ended")
        self.chunk_index = chunk_index
        self.trajectory_frames = trajectory_frames


class SanaModel:
    """Own the three native streaming caches and model progress."""

    def __init__(self) -> None:
        self._backend: SanaStreamingBackend | None = None
        self._config: SanaWMConfig | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._prompt = ""
        self._trajectory_frames = 0

    def load(self, config: SanaWMConfig) -> None:
        """Load the native backend from resolved, runtime-independent settings."""
        from sana_wm_backend import SanaStreamingBackend

        self._config = config
        self._backend = SanaStreamingBackend(config)

    def generate(self, input: SanaInput) -> SanaResult:
        """Apply a requested world and complete one native three-stage chunk."""
        backend, config = self._backend, self._config
        if backend is None or config is None:
            raise RuntimeError("SANA-WM was not loaded")
        if input.world_id != self._world_id:
            anchor = input.anchor
            if anchor is None:
                raise NoAnchor(input.world_id)
            backend.reset(
                anchor.image,
                anchor.prompt,
                anchor.seed,
                intrinsics_source=anchor.intrinsics,
            )
            self._world_id = input.world_id
            self._chunk_index = 0
            self._prompt = anchor.prompt
            self._trajectory_frames = anchor.trajectory_frames
        if self._chunk_index >= config.max_chunks:
            raise RolloutExhausted(input.world_id)
        if input.poses is None:
            raise TrajectoryCompleteError(self._chunk_index, self._trajectory_frames)
        frames = backend.generate_chunk(input.poses)
        self._chunk_index += 1
        return SanaResult(frames, input.world_id, self._prompt, self._chunk_index)

    def reset(self) -> None:
        """Drop causal state and progress while retaining loaded model weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
        self._prompt = ""
        self._trajectory_frames = 0
