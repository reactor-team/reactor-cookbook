"""Runtime-independent Lyra weights, native caches, and rollout bookkeeping."""

from dataclasses import dataclass

import numpy as np

from lyra2_backend import Lyra2Backend


@dataclass(frozen=True)
class Lyra2Input:
    world_id: int
    anchor: np.ndarray | None
    prompt: str
    seed: int
    w2c: np.ndarray | None
    intrinsics: np.ndarray | None


@dataclass(frozen=True)
class Lyra2Result:
    world_id: int
    chunk: int
    frames: np.ndarray | None
    corrected_c2w: np.ndarray | None
    intrinsics: np.ndarray | None


class NoAnchor(Exception):
    """A new world needs an image."""


class NoCamera(Exception):
    """A seeded world needs calibrated camera poses."""


class Lyra2Model:
    def __init__(self) -> None:
        self.backend: Lyra2Backend | None = None
        self.world_id: int | None = None
        self.chunk = 0

    def load(self, config: dict) -> None:
        """Load weights using application-resolved paths inside the worker."""
        self.backend = Lyra2Backend(config)

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.clear()
        self.world_id = None
        self.chunk = 0

    def generate(self, input: Lyra2Input) -> Lyra2Result:
        if self.backend is None:
            raise RuntimeError("Lyra-2 not loaded")
        if input.world_id != self.world_id:
            if input.anchor is None:
                raise NoAnchor("A new Lyra world requires an anchor image")
            c2w, intrinsics = self.backend.reset(
                input.anchor, prompt=input.prompt, seed=input.seed
            )
            self.world_id = input.world_id
            self.chunk = 0
            return Lyra2Result(
                world_id=input.world_id,
                chunk=0,
                frames=None,
                corrected_c2w=c2w,
                intrinsics=intrinsics,
            )
        if input.w2c is None or input.intrinsics is None:
            raise NoCamera("Camera poses must follow seed calibration")
        frames, corrected = self.backend.generate_chunk(
            input.w2c,
            input.intrinsics,
            prompt=input.prompt,
            chunk=self.chunk + 1,
        )
        self.chunk += 1
        return Lyra2Result(
            world_id=input.world_id,
            chunk=self.chunk,
            frames=frames,
            corrected_c2w=corrected,
            intrinsics=None,
        )
