"""Model-side bookkeeping for one FastFrames latent per step."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

LATENT_STEPS = 21
FOLLOWUP_FRAMES = 4
VIDEO_FRAMES = 1 + (LATENT_STEPS - 1) * FOLLOWUP_FRAMES
Variant = Literal["ff-s", "ff-l"]


@dataclass(frozen=True)
class Settings:
    variant: Variant
    source: Path
    apple: Path
    wan: Path
    generator: Path


@dataclass(frozen=True)
class FastFramesInput:
    world_id: int
    prompt: str
    prompt_revision: int
    seed: int | None  # Present until the first result acknowledges the world.


@dataclass(frozen=True)
class FastFramesResult:
    world_id: int
    chunk: int
    prompt_revision: int
    complete: bool
    frames: np.ndarray  # uint8 RGB, [1 or 4, 480, 832, 3]


class RolloutStateError(RuntimeError):
    """The requested step cannot run in the current rollout."""


class Backend(Protocol):
    def start(self, seed: int) -> None: ...
    def set_prompt(self, prompt: str) -> None: ...
    def step(self, index: int) -> np.ndarray: ...
    def reset(self) -> None: ...


class FastFramesModel:
    def __init__(self) -> None:
        self._backend: Backend
        self._world_id: int | None = None
        self._prompt_revision: int | None = None
        self._next = 0

    def load(self, settings: Settings) -> None:
        from apple_fastframes_backend import NativeBackend

        self._backend = NativeBackend(settings)

    def generate(self, input: FastFramesInput) -> FastFramesResult:
        if self._world_id != input.world_id:
            if input.seed is None:
                raise RolloutStateError("A new world requires its seed")
            self._backend.start(input.seed)
            self._world_id = input.world_id
            self._next = 0
            self._prompt_revision = None
        if self._next >= LATENT_STEPS:
            raise RolloutStateError("The released FastFrames clip is complete")
        if self._prompt_revision != input.prompt_revision:
            self._backend.set_prompt(input.prompt)
            self._prompt_revision = input.prompt_revision
        frames = self._backend.step(self._next)
        self._next += 1
        return FastFramesResult(
            world_id=input.world_id,
            chunk=self._next,
            prompt_revision=input.prompt_revision,
            complete=self._next == LATENT_STEPS,
            frames=frames,
        )

    def reset(self) -> None:
        self._backend.reset()
        self._world_id = None
        self._prompt_revision = None
        self._next = 0
