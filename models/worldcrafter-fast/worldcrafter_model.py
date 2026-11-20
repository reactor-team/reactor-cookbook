"""Plain, immutable step contracts and the stateful upstream model boundary."""
from dataclasses import dataclass

import numpy as np

from worldcrafter_camera import FRAMES


@dataclass(frozen=True)
class WorldCrafterInput:
    """One immutable step: encoded anchor once, plus cumulative CPU poses.

    Both pose arrays are float32 ``(33 * (chunk_index + 1), 3, 4)`` c2w.
    ``local_poses`` restarts at each chunk's first frame; ``world_poses`` is
    relative to the initial camera and is used for memory retrieval.
    ``image`` is validated original PNG/JPEG/WebP bytes, not resized pixels.

    A camera-path step instead carries ``path_poses``: the uploaded float32
    ``(33 * (chunk_index + 1), 4, 4)`` global prefix, unaligned. The worker
    aligns it to gravity and derives local poses exactly as upstream
    ``inference.py --camera-path`` does; both planned arrays are then None.
    """
    world_id: int
    image: bytes | None
    prompt: str
    seed: int
    gravity_align: bool
    local_poses: np.ndarray | None
    world_poses: np.ndarray | None
    path_poses: np.ndarray | None = None


@dataclass(frozen=True)
class WorldCrafterResult:
    """CPU RGB uint8 ``(33,384,640,3)`` in [0,255] and acknowledged world.

    ``camera_end`` is the last sampled c2w pose, shape ``(3,4)`` relative to
    the initial camera. ``world_up`` is a calibrated unit vector in that
    camera's coordinates, or None when disabled/unavailable.
    """
    world_id: int
    chunk_index: int
    frames: np.ndarray
    camera_end: list[list[float]]
    gravity_aligned: bool
    gpu_count: int
    world_up: list[float] | None


class NoAnchor(Exception):
    """A step asks for a world the model does not hold and carries no anchor image."""


class RolloutExhausted(Exception):
    """The world has generated its maximum number of chunks."""


class InvalidPoses(Exception):
    """The step's cumulative poses do not match the chunk the model generates next."""


class WorldCrafterModel:
    def __init__(self):
        self.backend = None
        self._max_chunks = 0
        self._world_id: int | None = None
        self._next_chunk = 0

    def load(self, settings):
        from upstream_backend import WorldCrafterBackend
        self._max_chunks = settings.max_chunks
        self.backend = WorldCrafterBackend(settings)

    @property
    def healthy(self) -> bool:
        """Whether the inference worker accepts calls; False once it failed or was closed."""
        return self.backend is not None and self.backend.healthy

    def generate(self, input: WorldCrafterInput) -> WorldCrafterResult:
        new_world = input.world_id != self._world_id
        if new_world and input.image is None:
            raise NoAnchor("A new world requires an anchor image")
        index = 0 if new_world else self._next_chunk
        if index >= self._max_chunks:
            raise RolloutExhausted("World chunk limit reached")
        if input.path_poses is not None:
            expected, values = (FRAMES * (index + 1), 4, 4), (input.path_poses,)
            if input.local_poses is not None or input.world_poses is not None:
                raise InvalidPoses("A camera-path step carries only path poses")
        else:
            expected, values = (FRAMES * (index + 1), 3, 4), (input.local_poses, input.world_poses)
        for value in values:
            if (not isinstance(value, np.ndarray) or value.shape != expected
                    or value.dtype != np.float32 or not np.isfinite(value).all()):
                raise InvalidPoses(f"Expected finite float32 poses of shape {expected}")
        result = self.backend.generate(input)
        self._world_id, self._next_chunk = result.world_id, result.chunk_index + 1
        return result

    def reset(self):
        self.backend.reset()
        self._world_id, self._next_chunk = None, 0

    def close(self):
        if self.backend is not None:
            self.backend.close()
