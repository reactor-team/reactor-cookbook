"""Plain model lifecycle and CPU contracts for History-Forcing DMD."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from history_forcing_backend import NativeBackend


@dataclass(frozen=True)
class Settings:
    """Resolved, pinned assets passed to the isolated worker at load time."""

    source: Path
    """Upstream checkout containing the released DMD implementation."""
    model_root: Path
    """Directory containing the Wan backbone, text encoder, and VAE."""
    checkpoint: Path
    """Released DMD checkpoint, loaded with its EMA weights."""


@dataclass(frozen=True)
class Anchor:
    seed: int
    """Seed applied once when a new video begins."""


@dataclass(frozen=True)
class HistoryForcingInput:
    world_id: int
    """Application-owned video identity; changing it starts a new video."""
    prompt: str
    """Text conditioning for the next native block."""
    anchor: Anchor | None
    """Required for a new video; omitted after successful acknowledgement."""


@dataclass(frozen=True)
class HistoryForcingResult:
    """One native chunk: CPU uint8 RGB [9 or 12, 480, 832, 3]."""

    world_id: int
    """Identity of the video that actually produced these frames."""
    chunk_index: int
    """One-based count of successfully generated blocks in that video."""
    frames: np.ndarray
    """RGB pixels ready for Runtime's media encoder."""


class MissingAnchorError(ValueError):
    """A caller requested a new video without supplying its seed."""


class HistoryForcingModel:
    """Own one video's native state and count successfully generated chunks.

    A new world_id requires an Anchor, otherwise MissingAnchorError is raised.
    The first successful result acknowledges that world so later inputs may omit
    the anchor. reset() discards video state and retains the loaded weights.
    """

    def __init__(self) -> None:
        self._backend: NativeBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, settings: Settings) -> None:
        """Load resident native modules once, inside the worker process."""
        from history_forcing_backend import NativeBackend

        self._backend = NativeBackend(settings)

    def generate(self, input: HistoryForcingInput) -> HistoryForcingResult:
        """Advance one native block, resetting only for a new video identity."""
        if self._backend is None:
            raise RuntimeError("load() must complete before generation")
        if not input.prompt.strip():
            raise ValueError("The model requires a nonempty prompt")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise MissingAnchorError("A new video requires its seed")
            self.reset()
            self._backend.begin(input.anchor.seed)
            self._world_id = input.world_id
        frames = self._backend.step(input.prompt)
        self._chunk_index += 1
        return HistoryForcingResult(
            world_id=input.world_id, chunk_index=self._chunk_index, frames=frames
        )

    def reset(self) -> None:
        """Discard video state while retaining the loaded model weights."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None
        self._chunk_index = 0
