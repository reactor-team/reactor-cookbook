"""One native Murea-Fast clip per call, continued from its 17-frame tail."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

FPS = 24
SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
PREFIX_FRAMES = 17
WIDTH, HEIGHT = 1280, 704
CHUNK_FRAMES = (24, 48, 72, 120, 240, 360)


class NoAnchor(ValueError):
    """A fresh world was requested without its starting inputs."""


if TYPE_CHECKING:
    from inworld_backend import InWorldBackend


@dataclass(frozen=True)
class InWorldSettings:
    """Resolved source/weight paths and worker port."""

    source: Path
    base: Path
    checkpoint: Path
    connectors: Path
    master_port: int
    attention_backend: str = "fa"
    cuda_graph: bool = True
    num_gpus: int = 1


@dataclass(frozen=True)
class InWorldAnchor:
    """Start a video with ordered RGB uint8 HWC references or encoded parent video.

    An empty media selection starts from text at WIDTH x HEIGHT;
    new_frames is a multiple of eight. The first text/image clip adds one opening
    frame. Seed follows upstream's per-request generator semantics.
    """

    # RGB uint8 HWC images in the author's reference order.
    seed: int
    new_frames: int
    references: tuple[np.ndarray, ...] = ()
    # A validated 24 FPS parent clip; mutually exclusive with references.
    video: bytes | None = None


@dataclass(frozen=True)
class InWorldInput:
    """One segment's text and identity, with an anchor until world_id is acknowledged."""

    world_id: int
    prompt: str
    anchor: InWorldAnchor | None = None


@dataclass(frozen=True)
class InWorldResult:
    """Generated CPU media and the applied world ID / one-based segment count.

    Frames are RGB uint8 (T, H, W, 3); audio is float32 (1, T * SAMPLES_PER_FRAME) at
    48 kHz. Only new frames/samples follow a continuation, excluding its prefix.
    """

    world_id: int
    chunk_index: int
    frames: np.ndarray  # RGB uint8 THWC, with the continuation prefix removed.
    audio: np.ndarray  # float32 channels x samples, 48 kHz.


class InWorldModel:
    """Own resident weights and the bounded native MP4 continuation history."""

    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1
        self._backend: InWorldBackend | None = None
        self._settings: InWorldSettings | None = None
        self._temporary: TemporaryDirectory | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._anchor: InWorldAnchor | None = None
        self._parent: Path | None = None
        self._references: tuple[Path, ...] = ()

    def load(self, settings: InWorldSettings) -> None:
        """Load the pinned upstream worker once in this process."""
        from inworld_backend import InWorldBackend

        self._settings = settings
        self._backend = InWorldBackend(
            settings, rank=self.rank, world_size=self.world_size
        )
        self._backend.warmup()

    def reset(self) -> None:
        """Discard session media and counters, retaining loaded model weights."""
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = None
        self._world_id = None
        self._chunk_index = 0
        self._anchor = None
        self._parent = None
        self._references = ()

    def generate(self, input: InWorldInput) -> InWorldResult | None:
        """Coordinate a native step, including failures before GPU collectives."""
        if self._backend is None:
            raise RuntimeError("InWorld has not been loaded")
        if self.world_size == 1:
            return self._generate(input)
        if self.rank != 0:
            return self._backend.follow_step()
        self._backend.begin_step()
        try:
            return self._generate(input)
        except Exception as exc:
            self._backend.cancel_unsubmitted_step(str(exc))
            raise

    def _generate(self, input: InWorldInput) -> InWorldResult:
        """Generate one clip; invalid inputs raise ValueError, native failures propagate.

        Only a successfully decoded clip advances the retained parent and count.
        A new world requires its anchor; a known world continues its last 17 frames.
        """
        if self._backend is None or self._settings is None:
            raise RuntimeError("InWorld has not been loaded")
        if not input.prompt.strip():
            raise ValueError("A non-empty prompt is required")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor("A new world requires its anchor")
            anchor = input.anchor
            if anchor.video is not None and anchor.references:
                raise ValueError("Video continuation cannot include reference images")
            if len(anchor.references) > 5:
                raise ValueError("At most five ordered references are supported")
            if anchor.new_frames not in CHUNK_FRAMES:
                raise ValueError("Choose one of the offered chunk lengths")
            self.reset()
            self._temporary = TemporaryDirectory(prefix="inworld-")
            root = Path(self._temporary.name)
            self._references = self._backend.store_references(root, anchor.references)
            if anchor.video is not None:
                self._parent = root / "uploaded.mp4"
                self._parent.write_bytes(anchor.video)
            self._anchor = anchor
            self._world_id = input.world_id

        anchor = self._anchor
        if anchor is None or self._temporary is None:
            raise RuntimeError("No active world")
        root = Path(self._temporary.name)
        index = self._chunk_index + 1
        # Each request uses the author's seed semantics, not a fabricated KV cache.
        path, frames, audio = self._backend.generate_clip(
            root=root,
            index=index,
            prompt=input.prompt,
            anchor=anchor,
            parent=self._parent,
            references=self._references,
        )
        previous = self._parent
        self._parent = path
        self._chunk_index = index
        if previous is not None and previous != path:
            previous.unlink(missing_ok=True)
        return InWorldResult(
            world_id=input.world_id,
            chunk_index=index,
            frames=frames,
            audio=audio,
        )
