"""SolarWM model ownership and immutable CPU step contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolarWMAnchor:
    """Prepared uint8 RGB (480,864,3) image and new-world conditions."""

    image: np.ndarray
    prompt: str
    seed: int


@dataclass(frozen=True)
class SolarWMInput:
    world_id: int
    anchor: SolarWMAnchor | None
    poses: np.ndarray


@dataclass(frozen=True)
class SolarWMResult:
    world_id: int
    chunk_index: int
    frames: np.ndarray
    complete: bool
    max_chunks: int


class NoAnchor(Exception):
    """A fresh world requires its image and conditioning."""


class RolloutExhausted(Exception):
    """The configured native rollout limit has been reached."""


class SolarWMModel:
    def __init__(self) -> None:
        self.backend = None
        self.world_id: int | None = None
        self._complete = False
        self.max_chunks = 320

    def load(self, config) -> None:
        from solarwm_stage2_backend import BackendSettings, SolarWMBackend

        self.max_chunks = config.max_chunks
        self.backend = SolarWMBackend(
            BackendSettings(
                config.upstream_config,
                config.base_path,
                config.checkpoint_path,
                config.runtime_root,
            )
        )

    def generate(self, input: SolarWMInput) -> SolarWMResult:
        if input.world_id != self.world_id:
            if input.anchor is None:
                raise NoAnchor("a fresh world requires an anchor")
            self.backend.reset(
                input.anchor.seed, input.anchor.image, input.anchor.prompt
            )
            self.world_id = input.world_id
            self._complete = False
        if self._complete:
            raise RolloutExhausted("reset the world before continuing")
        frames, chunk_index = self.backend.generate_chunk(input.poses)
        self._complete = chunk_index >= self.max_chunks
        return SolarWMResult(
            world_id=self.world_id,
            chunk_index=chunk_index,
            frames=frames,
            complete=self._complete,
            max_chunks=self.max_chunks,
        )

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.end_session()
        self.world_id = None
        self._complete = False
