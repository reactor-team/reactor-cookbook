"""Runtime-independent native Zing rollout and CPU step contract."""

from dataclasses import dataclass

import numpy as np
from zing_assets import ZingAdapterConfig

# Released generator memory geometry, not tunable serving controls.
LOCAL_ATTN_SIZE = 97
SINK_SIZE = 9
NATIVE_KEYS = ("w", "a", "s", "d", "i", "j", "k", "l")


def action_values(pressed) -> list[float]:
    """Native training order: translation WASD followed by view IJKL."""
    active = set(pressed)
    return [float(key in active) for key in NATIVE_KEYS]


@dataclass(frozen=True)
class ZingInput:
    """One block; a fresh image world supplies uint8 RGB (704,1248,3)."""

    world_id: int
    image: np.ndarray | None
    prompt: str
    seed: int
    pressed_keys: frozenset[str]
    image_required: bool = False


@dataclass(frozen=True)
class ZingResult:
    """Actual progress and CPU uint8 RGB (T,H,W,3) video; no echoed controls."""

    world_id: int
    frames: np.ndarray
    index: int
    complete: bool


class RolloutComplete(Exception):
    """The configured native rollout horizon has been reached."""


class NotSeeded(Exception):
    """An image-conditioned world requires its anchor on its first step."""


class ZingModel:
    def __init__(self) -> None:
        self.backend = None
        self.config = None
        self.world_id: int | None = None
        self.index = 0

    def load(self, config: ZingAdapterConfig) -> None:
        from zing_backend import ZingBackend

        self.config = config
        self.backend = ZingBackend(config)

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.end_session()
        self.world_id = None
        self.index = 0

    def generate(self, input: ZingInput) -> ZingResult:
        if self.backend is None or self.config is None:
            raise RuntimeError("Zing is not loaded")
        if input.world_id != self.world_id:
            if input.image_required and input.image is None:
                raise NotSeeded("no anchor for new image-conditioned world")
            self.backend.reset(image=input.image, prompt=input.prompt, seed=input.seed)
            self.world_id = input.world_id
            self.index = 0
        if self.index >= self.config.max_chunks:
            raise RolloutComplete("rollout limit reached; reset required")
        frames = self.backend.generate_chunk(
            prompt=input.prompt, pressed_keys=input.pressed_keys
        )
        self.index += 1
        return ZingResult(
            world_id=self.world_id,
            frames=frames,
            index=self.index,
            complete=self.index >= self.config.max_chunks,
        )
