"""HY-World 1.5 weights and causal rollout, independent of the application."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from hy_world_1_5_assets import HYWorld15Config
from hy_world_1_5_camera import CameraChunk
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from hy_world_1_5_backend import HYWorld15Backend


@dataclass(frozen=True)
class HYWorld15Anchor:
    """Encoded reference and seed carried until the world is acknowledged."""

    image: Path | bytes
    seed: int


@dataclass(frozen=True)
class HYWorld15Input:
    """One world's optional anchor, current prompt, and native camera arrays."""

    world_id: int
    anchor: HYWorld15Anchor | None
    prompt: str
    viewmats: np.ndarray
    intrinsics: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class HYWorld15Result:
    """Native frames and the model's acknowledged world and completion state."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    prompt: str
    complete: bool


class NoAnchor(Exception):
    """A fresh world requires a reference image."""


class RolloutExhausted(Exception):
    """The configured world capacity has been reached."""


class HYWorld15Model:
    """Own the native backend, geometric memory, and successful chunk count."""

    def __init__(self) -> None:
        self._backend: HYWorld15Backend | None = None
        self._config: HYWorld15Config | None = None
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, config: HYWorld15Config) -> None:
        """Load the native backend from application-resolved asset paths."""
        from hy_world_1_5_backend import HYWorld15Backend

        self._config = config
        self._backend = HYWorld15Backend(config)
        self._backend.load()

    def generate(self, input: HYWorld15Input) -> HYWorld15Result:
        """Apply an explicit new world or advance the existing causal history."""
        if self._backend is None or self._config is None:
            raise RuntimeError("HY-World 1.5 model was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor("A fresh world requires a reference image")
            source = input.anchor.image
            with Image.open(
                io.BytesIO(source) if isinstance(source, bytes) else source
            ) as image:
                reference = ImageOps.exif_transpose(image).convert("RGB").copy()
            self._backend.reset(
                image=reference, prompt=input.prompt, seed=input.anchor.seed
            )
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= self._config.max_chunks:
            raise RolloutExhausted("Reset the world before requesting another chunk")
        camera = CameraChunk(input.viewmats, input.intrinsics, input.actions)
        frames = np.asarray(self._backend.generate_chunk(camera, input.prompt))
        expected = 13 if self._chunk_index == 0 else 16
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] != expected:
            raise RuntimeError(
                f"HY-World chunk must contain {expected} RGB frames, got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        self._chunk_index += 1
        return HYWorld15Result(
            frames=np.ascontiguousarray(frames),
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            prompt=input.prompt,
            complete=self._chunk_index >= self._config.max_chunks,
        )

    def reset(self) -> None:
        """Release world-specific state while retaining the loaded checkpoint."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
