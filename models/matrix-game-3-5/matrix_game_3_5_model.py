"""Own the native Matrix 3.5 worker and its causal rollout without Reactor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from upstream_backend import MatrixWorkerBackend, WorkerSettings

OUTPUT_FRAMES_PER_CHUNK = 12


def normalize_output_frames(value: np.ndarray) -> np.ndarray:
    """Return exactly one contiguous uint8 RGB Matrix output chunk."""
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"Matrix output must have shape (T, H, W, 3), got {frames.shape}"
        )
    if int(frames.shape[0]) != OUTPUT_FRAMES_PER_CHUNK:
        raise RuntimeError(
            f"Matrix output must contain {OUTPUT_FRAMES_PER_CHUNK} frames, "
            f"got {int(frames.shape[0])}"
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


@dataclass(frozen=True)
class MatrixGame35Anchor:
    """Carry the reference and seed used only when starting a fresh world."""

    image: Path | bytes
    suffix: str
    seed: int


@dataclass(frozen=True)
class MatrixGame35Input:
    """Describe one world and its next chunk's text and camera trajectory."""

    world_id: int
    anchor: MatrixGame35Anchor | None
    prompt: str
    trajectory: np.ndarray


@dataclass(frozen=True)
class MatrixGame35Result:
    """Report the generated frames and successful native rollout position."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    complete: bool


class NoAnchor(Exception):
    """A new world was requested without its reference image."""


class RolloutExhausted(Exception):
    """The rollout has reached its configured native cap."""


class MatrixGame35Model:
    """Keep the native worker, world identity, and chunk count between steps."""

    def __init__(self) -> None:
        self._backend: MatrixWorkerBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._seed = 0
        self._max_chunks = 0

    def load(self, settings: WorkerSettings, intrinsics: np.ndarray) -> None:
        """Start the isolated worker and load weights once."""
        self._backend = MatrixWorkerBackend(settings, intrinsics)
        self._max_chunks = settings.max_chunks

    def generate(self, input: MatrixGame35Input) -> MatrixGame35Result:
        """Apply a requested world and generate one native 12-frame chunk."""
        if self._backend is None:
            raise RuntimeError("Matrix-Game-3.5 was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            self._backend.reset(
                seed=input.anchor.seed,
                anchor_image=input.anchor.image,
                suffix=input.anchor.suffix,
                prompt=input.prompt.strip(),
            )
            self._seed = input.anchor.seed
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= self._max_chunks:
            raise RolloutExhausted(self._chunk_index)
        frames = normalize_output_frames(
            self._backend.generate_chunk(input.trajectory, self._seed, input.prompt)
        )
        self._chunk_index += 1
        return MatrixGame35Result(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            complete=self._chunk_index >= self._max_chunks,
        )

    def reset(self) -> None:
        """Release native rollout caches while keeping the worker's weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
        self._seed = 0
