"""Own native LingBot-V2 causal inference independently of Reactor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lingbot_world_v2_assets import LingBotConfig
from lingbot_world_v2_backend import LingBotBackend


@dataclass(frozen=True)
class LingbotV2Anchor:
    """Describe the image, calibration, and seed of a fresh world."""

    image: Path | bytes
    intrinsics: np.ndarray
    seed: int


@dataclass(frozen=True)
class LingbotV2Input:
    """Carry one world's prompt and four planned relative camera poses."""

    world_id: int
    anchor: LingbotV2Anchor | None
    prompt: str
    poses: np.ndarray


@dataclass(frozen=True)
class LingbotV2Result:
    """Report decoded RGB frames and the completed native rollout position."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    prompt: str
    complete: bool


class NoAnchor(Exception):
    """A fresh world has no image to condition on."""


class RolloutExhausted(Exception):
    """The native temporal position window cannot accept another chunk."""


class LingbotV2Model:
    """Retain weights and native causal caches between single-chunk calls."""

    def __init__(self) -> None:
        self._backend: LingBotBackend | None = None
        self._max_chunks = 0
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, config: LingBotConfig) -> None:
        """Load native weights from the application's resolved configuration."""
        self._backend = LingBotBackend(config)
        self._max_chunks = config.max_chunks

    def generate(self, input: LingbotV2Input) -> LingbotV2Result:
        """Apply a requested new world, then generate one native chunk."""
        if self._backend is None:
            raise RuntimeError("LingBot-World-V2 was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            self._backend.reset(
                image=input.anchor.image,
                prompt=input.prompt,
                seed=input.anchor.seed,
                intrinsics=input.anchor.intrinsics,
            )
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= self._max_chunks:
            raise RolloutExhausted(self._chunk_index)
        frames = self._backend.generate_chunk(
            prompt=input.prompt, relative_poses=input.poses
        )
        self._chunk_index += 1
        return LingbotV2Result(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            prompt=input.prompt.strip(),
            complete=self._chunk_index >= self._max_chunks,
        )

    def reset(self) -> None:
        """Release rollout caches while retaining weights for the next world."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
