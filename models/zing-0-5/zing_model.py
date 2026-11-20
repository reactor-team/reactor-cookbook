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
    """Carry exactly what one chunk needs across to the model.

    Attributes:
        world_id: Identifies the world the application wants this chunk from.
            An id the model has not applied asks for a fresh world.
        image: The anchor for a fresh image world, uint8 RGB ``(704, 1248, 3)``
            on the CPU. Carried only while the application has not seen
            ``world_id`` reported back on a result; ``None`` otherwise, and
            always ``None`` for a text world.
        prompt: The scene prompt in effect for this chunk.
        seed: The random seed a fresh world is sampled with.
        pressed_keys: The movement and view keys held for this chunk, a subset
            of ``NATIVE_KEYS``.
        image_required: Whether a fresh world must start from ``image``.
    """

    world_id: int
    image: np.ndarray | None
    prompt: str
    seed: int
    pressed_keys: frozenset[str]
    image_required: bool = False


@dataclass(frozen=True)
class ZingResult:
    """Carry what one chunk produced back to the application.

    Attributes:
        world_id: The world these frames belong to. The application reads it
            to learn the fresh world it asked for has started.
        frames: Decoded RGB frames, uint8 ``(16, 704, 1248, 3)`` on the CPU.
        index: The model's own one-based count of chunks in this world.
        complete: Whether this chunk reached the world's chunk limit; the next
            step on the same world raises ``RolloutComplete``.
    """

    world_id: int
    frames: np.ndarray
    index: int
    complete: bool


class RolloutComplete(Exception):
    """The configured native rollout horizon has been reached."""


class NotSeeded(Exception):
    """An image-conditioned world requires its anchor on its first step."""


class ZingModel:
    """Hold the Zing weights and step one world 16 frames at a time.

    The application constructs this class in its ``load()`` and reads it only
    through ``ZingResult``. This class owns the native backend, the active
    world, and its chunk count.
    """

    def __init__(self) -> None:
        self.backend = None
        self.config = None
        self.world_id: int | None = None
        self.index = 0

    def load(self, config: ZingAdapterConfig) -> None:
        """Load the weights the prepared ``config`` names."""
        from zing_backend import ZingBackend

        self.config = config
        self.backend = ZingBackend(config)

    def reset(self) -> None:
        """Forget the current world and release its caches; keep the weights."""
        if self.backend is not None:
            self.backend.end_session()
        self.world_id = None
        self.index = 0

    def generate(self, input: ZingInput) -> ZingResult:
        """Generate one chunk, starting a fresh world first when the input asks for one.

        Raises:
            NotSeeded: The input names an image world this model has not
                started and carries no anchor image.
            RolloutComplete: The current world already reached its chunk limit.
        """
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
