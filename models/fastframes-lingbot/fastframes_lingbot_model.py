"""Resident FastFrames-LingBot worker and its CPU step contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

CameraAction = Literal[
    "idle", "forward", "backward", "left", "right", "up", "down",
    "yaw_left", "yaw_right", "pitch_up", "pitch_down",
]
WIDTH = 832
HEIGHT = 464
CHUNKS = 256
CHUNK_FRAMES = 16
WORLD_FRAMES = CHUNKS * CHUNK_FRAMES - 3


@dataclass(frozen=True)
class Settings:
    fastframes_source: Path
    lingbot_source: Path
    networks: Path
    dit: Path
    shared: Path


@dataclass(frozen=True)
class Anchor:
    image: np.ndarray
    seed: int


@dataclass(frozen=True)
class LingBotInput:
    world_id: int
    anchor: Anchor | None
    prompt: str
    prompt_revision: int
    action: CameraAction


@dataclass(frozen=True)
class LingBotResult:
    world_id: int
    prompt_revision: int
    action: CameraAction
    chunk: int
    frames: np.ndarray


class WorldStateError(RuntimeError):
    """A step is incompatible with the worker's current world."""


class FastFramesLingBotModel:
    def load(self, settings: Settings) -> None:
        from fastframes_lingbot_backend import NativeBackend

        self.backend = NativeBackend(settings)
        self.backend.warmup()
        self.world_id: int | None = None

    def generate(self, input: LingBotInput) -> LingBotResult:
        if self.world_id != input.world_id:
            if input.anchor is None:
                raise WorldStateError("The first step of a world needs an anchor")
            self.backend.start(input.anchor, input.prompt)
            self.world_id = input.world_id
        else:
            self.backend.set_prompt(input.prompt)
        frames = self.backend.step(input.action)
        return LingBotResult(
            world_id=input.world_id,
            prompt_revision=input.prompt_revision,
            action=input.action,
            chunk=self.backend.chunk,
            frames=frames,
        )

    def reset(self) -> None:
        self.world_id = None
        self.backend.reset()
