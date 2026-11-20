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
    # A camera path's first K, which seeds the world instead of DA3's.
    seed_intrinsics: np.ndarray | None = None


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
        self.intrinsics: np.ndarray | None = None
        self.chunk = 0

    def load(self, config: dict) -> None:
        """Load weights using application-resolved paths inside the worker."""
        self.backend = Lyra2Backend(config)

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.clear()
        self.world_id = None
        self.intrinsics = None
        self.chunk = 0

    def generate(self, input: Lyra2Input) -> Lyra2Result:
        """Run one native 80-frame chunk; a new world is seeded first, in the same step.

        Every Runtime step yields video: seeding (DA3/MoGe depth, cache) and the
        world's first chunk run together, in the official script's order.
        ``input.intrinsics`` None means the seed's intrinsics.
        """
        if self.backend is None:
            raise RuntimeError("Lyra-2 not loaded")
        if input.w2c is None:
            raise NoCamera("Each Lyra step needs its 80 camera poses")
        fresh = input.world_id != self.world_id
        seed_k = None
        if fresh:
            if input.anchor is None:
                raise NoAnchor("A new Lyra world requires an anchor image")
            self.world_id = None
            seed_c2w, seed_k = self.backend.reset(
                input.anchor,
                prompt=input.prompt,
                seed=input.seed,
                intrinsics=input.seed_intrinsics,
            )
            # Paths and held controls are planned from the identity seed camera.
            if not np.array_equal(seed_c2w, np.eye(4, dtype=seed_c2w.dtype)):
                raise RuntimeError("Lyra seeded a non-identity camera")
            self.intrinsics = seed_k
            self.chunk = 0
        intrinsics = input.intrinsics
        if intrinsics is None:
            intrinsics = np.repeat(
                np.asarray(self.intrinsics, dtype=np.float32)[None], len(input.w2c), 0
            )
        frames, corrected = self.backend.generate_chunk(
            input.w2c,
            intrinsics,
            prompt=input.prompt,
            chunk=self.chunk + 1,
        )
        self.world_id = input.world_id
        self.chunk += 1
        return Lyra2Result(
            world_id=input.world_id,
            chunk=self.chunk,
            frames=frames,
            corrected_c2w=corrected,
            intrinsics=seed_k,
        )
