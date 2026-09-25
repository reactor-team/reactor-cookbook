"""Own the native Matrix-Game 3.0 rollout independently of Reactor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matrix_game_3_0_assets import MatrixGame30Config
from matrix_game_3_0_backend import MatrixGame30Backend, NativeAction
from numpy.typing import NDArray

FIRST_CHUNK_FRAMES = 57
LATER_CHUNK_FRAMES = 40
FRAMES_PER_CHUNK = max(FIRST_CHUNK_FRAMES, LATER_CHUNK_FRAMES)


def normalize_output_frames(
    value: NDArray[np.generic], chunk_index: int
) -> NDArray[np.uint8]:
    """Return one contiguous uint8 RGB native Matrix chunk."""
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"Matrix output must have shape (T, H, W, 3), got {frames.shape}"
        )
    expected = FIRST_CHUNK_FRAMES if chunk_index == 0 else LATER_CHUNK_FRAMES
    if int(frames.shape[0]) != expected:
        raise RuntimeError(
            f"Matrix chunk {chunk_index + 1} must contain {expected} frames, "
            f"got {int(frames.shape[0])}"
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


@dataclass(frozen=True)
class MatrixGame30Anchor:
    """Image, prompt and seed encoded once at the start of a world."""

    image: Path | bytes
    prompt: str
    seed: int


@dataclass(frozen=True)
class MatrixGame30Input:
    """Request one native iteration in an explicitly identified world."""

    world_id: int
    anchor: MatrixGame30Anchor | None
    action: NativeAction


@dataclass(frozen=True)
class MatrixGame30Result:
    """Report native frames and successfully committed world progress."""

    frames: NDArray[np.uint8]
    world_id: int
    chunk_index: int
    complete: bool


class NoAnchor(Exception):
    """A new world cannot start without its image condition."""


class RolloutExhausted(Exception):
    """The official iteration limit has been reached."""


class MatrixGame30Model:
    """Keep weights, the official rollout thread, and causal memory alive."""

    def __init__(self) -> None:
        self._backend: MatrixGame30Backend | None = None
        self._max_chunks = 0
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, config: MatrixGame30Config) -> None:
        """Load the native distilled backend from resolved asset paths."""
        self._backend = MatrixGame30Backend(config)
        self._backend.load()
        self._max_chunks = config.max_chunks

    def generate(self, input: MatrixGame30Input) -> MatrixGame30Result:
        """Start a requested world or resume it for one official iteration."""
        backend = self._backend
        if backend is None:
            raise RuntimeError("Matrix-Game 3.0 was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            backend.reset(input.anchor.prompt, input.anchor.seed, input.anchor.image)
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= self._max_chunks:
            raise RolloutExhausted(input.world_id)
        frames = normalize_output_frames(
            backend.generate_chunk(input.action), self._chunk_index
        )
        self._chunk_index += 1
        return MatrixGame30Result(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            complete=self._chunk_index >= self._max_chunks,
        )

    def reset(self) -> None:
        """Stop the native rollout at its action boundary and release its memory."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
