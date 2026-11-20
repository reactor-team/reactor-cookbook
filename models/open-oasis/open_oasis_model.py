"""Runtime-independent Open-Oasis rollout and step contract."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from open_oasis_assets import OpenOasisConfig
from open_oasis_backend import OpenOasisBackend


@dataclass(frozen=True)
class OpenOasisInput:
    """One frame request; conditioning is present until the world is acknowledged."""

    world_id: int
    """Identity chosen by the application for this world."""
    conditioning: np.ndarray | None
    """New world's uint8 RGB frames, shaped (T, 360, 640, 3)."""
    seed: int
    """Seed applied only when starting a world."""
    action: np.ndarray
    """Native float32 control vector shaped (25,)."""


@dataclass(frozen=True)
class OpenOasisResult:
    """Actual world progress and one uint8 RGB (360, 640, 3) frame."""

    world_id: int
    frame: np.ndarray
    index: int
    """Zero returns the last conditioning frame; positive values count new frames."""


class NotSeeded(Exception):
    """A new world requires visual conditioning."""


class OpenOasisModel:
    def __init__(self) -> None:
        self._backend: OpenOasisBackend | None = None
        self._world_id: int | None = None
        self._index = 0

    def load(self, config: OpenOasisConfig, model_path: Path, vae_path: Path) -> None:
        self._backend = OpenOasisBackend(config, model_path, vae_path)

    def reset(self) -> None:
        self._world_id = None
        self._index = 0
        if self._backend is not None:
            self._backend.clear()

    def generate(self, input: OpenOasisInput) -> OpenOasisResult:
        if self._backend is None:
            raise RuntimeError("Open-Oasis was not loaded")
        if input.world_id != self._world_id:
            if input.conditioning is None:
                raise NotSeeded("no conditioning for new world")
            self._backend.reset(input.conditioning, input.seed)
            self._world_id = input.world_id
            self._index = 0
            frame = input.conditioning[-1]
        else:
            frame = self._backend.generate_one(input.action)
            self._index += 1
        return OpenOasisResult(
            world_id=self._world_id,
            frame=np.ascontiguousarray(frame),
            index=self._index,
        )
