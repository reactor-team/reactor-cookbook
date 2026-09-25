"""The model half of LingBot-World v1 Fast: weights, causal state, one chunk per step.

Plain Python. This module imports nothing from ``reactor_runtime`` and knows
nothing about clients, tracks, or commands. The application half
(``lingbot_world_v1.py``) hands :class:`LingbotV1Model` to a Runner, which
constructs and loads it in a worker process. The application calls ``generate``
once per step and ``reset`` when a session ends. The two halves meet on
:class:`LingbotV1Input` and :class:`LingbotV1Result`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lingbot_world_v1_backend import LingBotBackend, WorkerSettings


@dataclass(frozen=True)
class AnchorImage:
    """Describe the image a fresh world starts from.

    Attributes:
        image: The image file on disk, or its encoded bytes.
        suffix: The file suffix the bytes decode as (``.png``, ``.jpg``, ...);
            ignored when ``image`` is a path.
        intrinsics: The camera calibration that pairs with the image.
        seed: The random seed the fresh world is sampled with.
    """

    image: Path | bytes
    suffix: str
    intrinsics: Path
    seed: int


@dataclass(frozen=True)
class LingbotV1Input:
    """Carry exactly what one chunk needs across to the model.

    Attributes:
        world_id: Identifies the world the application wants this chunk from.
            An id the model has not applied asks for a fresh world.
        anchor: The image to start that fresh world from. Carried only while
            the application has not seen ``world_id`` reported back on a
            result; ``None`` continues the current world.
        prompt: The scene prompt in effect for this chunk.
        poses: Relative camera-to-world transforms for the chunk's three
            latent frames, float32 ``(3, 4, 4)``.
    """

    world_id: int
    anchor: AnchorImage | None
    prompt: str
    poses: np.ndarray


@dataclass(frozen=True)
class LingbotV1Result:
    """Carry what one chunk produced back to the application.

    Attributes:
        frames: Decoded RGB frames, uint8 ``(n, H, W, 3)`` on the CPU: 9 on
            the first chunk of a world, 12 after.
        world_id: The world these frames belong to. The application reads it
            to learn the fresh world it asked for has started.
        chunk_index: The model's own one-based count of chunks in this world.
    """

    frames: np.ndarray
    world_id: int
    chunk_index: int


class NoAnchor(Exception):
    """A chunk asked for a world the model has not started, and carried no anchor."""


class LingbotV1Model:
    """Hold the upstream model and step it one causal chunk at a time.

    The Runner constructs this class in its worker process. This class owns
    the native backend, the active world, and its chunk count.
    """

    def __init__(self) -> None:
        # DistributedRunner injects these before load(); direct use stays single-GPU.
        self.rank = 0
        self.world_size = 1
        self._backend: LingBotBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, settings: WorkerSettings) -> None:
        """Load native weights once in the current worker process."""
        self._backend = LingBotBackend(
            settings, rank=self.rank, world_size=self.world_size
        )
        self.reset()

    def generate(self, input: LingbotV1Input) -> LingbotV1Result:
        """Generate one chunk, starting a fresh world first when the input asks for one.

        Raises:
            NoAnchor: The input names a world this model has not started and
                carries no anchor image to start it from.
        """
        backend = self._require_backend()
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            backend.reset(
                seed=input.anchor.seed,
                anchor_image=input.anchor.image,
                suffix=input.anchor.suffix,
                intrinsics=input.anchor.intrinsics,
                prompt=input.prompt,
            )
            self._world_id = input.world_id
            self._chunk_index = 0
        frames = backend.generate_chunk(input.poses, input.prompt)
        self._chunk_index += 1
        return LingbotV1Result(
            frames=frames, world_id=input.world_id, chunk_index=self._chunk_index
        )

    def reset(self) -> None:
        """Forget the current world and release its caches; keep the weights."""
        backend = self._require_backend()
        if self._world_id is not None:
            backend.end_session()
        self._world_id = None
        self._chunk_index = 0

    def _require_backend(self) -> LingBotBackend:
        if self._backend is None:
            raise RuntimeError("LingBot-World v1 was not loaded")
        return self._backend
