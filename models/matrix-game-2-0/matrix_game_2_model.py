"""Own Matrix-Game-2.0 weights and causal worlds independently of Reactor."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matrix_game_2_backend import ChunkAction, MatrixGame2Backend
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class MatrixGame2Anchor:
    """Carry a new world's encoded image and sampling seed."""

    image: Path | bytes
    seed: int


@dataclass(frozen=True)
class MatrixGame2Input:
    """Select a world and the native keyboard and mouse action for one chunk."""

    world_id: int
    anchor: MatrixGame2Anchor | None
    action: ChunkAction


@dataclass(frozen=True)
class MatrixGame2Result:
    """Report committed frames, world progress, and consumed controls."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    action: ChunkAction
    complete: bool


class NoAnchor(Exception):
    """A new world requires an image before it can generate."""


class RolloutExhausted(Exception):
    """The native latent horizon has no room for another chunk."""


def _load_input_image(value: Path | bytes) -> Image.Image:
    """Decode one EXIF-corrected RGB anchor without a runtime upload object."""
    source = io.BytesIO(value) if isinstance(value, bytes) else value
    try:
        with Image.open(source) as decoded:
            return ImageOps.exif_transpose(decoded).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise RuntimeError("failed to load Matrix starting image") from error


class MatrixGame2Model:
    """Keep the native backend and its bounded autoregressive rollout alive."""

    def __init__(self) -> None:
        self._backend: MatrixGame2Backend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._max_chunks = 0

    def load(
        self,
        *,
        source_path: Path,
        model_path: Path,
        checkpoint_file: str,
        max_latent_frames: int,
    ) -> None:
        """Load the pinned native backend from application-resolved asset paths."""
        self._backend = MatrixGame2Backend(
            source_path=source_path,
            model_path=model_path,
            checkpoint_file=checkpoint_file,
            max_latent_frames=max_latent_frames,
        )
        self._max_chunks = max_latent_frames // 3

    def generate(self, input: MatrixGame2Input) -> MatrixGame2Result:
        """Apply a requested world and advance one native three-latent chunk."""
        backend = self._backend
        if backend is None:
            raise RuntimeError("Matrix-Game-2.0 was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            backend.reset(_load_input_image(input.anchor.image), input.anchor.seed)
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= self._max_chunks:
            raise RolloutExhausted(input.world_id)
        frames = backend.generate_chunk(input.action)
        self._chunk_index += 1
        return MatrixGame2Result(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            action=input.action,
            complete=self._chunk_index >= self._max_chunks,
        )

    def reset(self) -> None:
        """Release session caches while keeping the loaded weights."""
        try:
            if self._backend is not None:
                self._backend.end_rollout()
        finally:
            self._world_id = None
            self._chunk_index = 0
