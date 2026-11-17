"""Runtime-independent MovieRoll lifecycle and CPU step contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from movieroll_backend import MovieRollBackend

Mode = Literal["t2v", "i2v"]
WIDTH, HEIGHT = 864, 480
CHUNK_SIZE, WINDOW_SIZE = 4, 5
# One sink, one midpoint per event, three recent chunks, and the rolling
# window must fit the released 120-frame RoPE grid, including the extra frame.
MAX_PROMPTS = 20


@dataclass(frozen=True)
class Settings:
    source: Path
    weights: Path
    sessions: Path
    history_bytes: int
    gpu_count: int = 1


@dataclass(frozen=True)
class Anchor:
    seed: int
    mode: Mode
    image: np.ndarray | None
    """Fitted RGB uint8 [480, 864, 3], required for I2V only."""


@dataclass(frozen=True)
class MovieRollInput:
    world_id: int
    prompt: str
    anchor: Anchor | None
    """New-video settings, sent until the result acknowledges world_id."""


@dataclass(frozen=True)
class MovieRollResult:
    world_id: int
    chunk_index: int
    frames: np.ndarray
    """Owned CPU uint8 RGB [13 or 16, 480, 864, 3]; nonleaders return zero frames."""


class MissingAnchorError(ValueError):
    """A new video needs explicit seed, mode, and optional initial image."""


class HistoryFullError(RuntimeError):
    """The configured temporary latent archive has reached its storage budget."""

    world_id: int | None = None


class MovieRollModel:
    """Resident weights with one-based chunk counts and world/anchor acknowledgment.

    An unseen world requires an anchor. Repeating its id continues the same run;
    reset forgets its caches and history while retaining both checkpoints.
    """

    def __init__(self) -> None:
        # DistributedRunner supplies these before load; direct use defaults to one GPU.
        self.rank = 0
        self.world_size = 1
        self._backend: MovieRollBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, settings: Settings) -> None:
        """Load prepared native weights once in the isolated worker."""
        import sys

        if self.world_size != settings.gpu_count:
            raise ValueError("Worker count must match the configured GPU count")
        sys.path.insert(0, str(settings.source))
        from movieroll_backend import MovieRollBackend

        self._backend = MovieRollBackend(settings, rank=self.rank)

    def generate(self, input: MovieRollInput) -> MovieRollResult:
        """Advance one chunk; raise MissingAnchorError for an unanchored new world."""
        if self._backend is None:
            raise RuntimeError("load() must complete before generation")
        if not input.prompt.strip():
            raise ValueError("A scene description is required")
        try:
            if input.world_id != self._world_id:
                if input.anchor is None:
                    raise MissingAnchorError("A new video requires an anchor")
                self.reset()
                self._backend.begin(input.anchor, input.prompt)
                self._world_id = input.world_id
            frames = self._backend.step(input.prompt)
        except HistoryFullError as error:
            error.world_id = input.world_id
            raise
        self._chunk_index += 1
        return MovieRollResult(
            world_id=input.world_id, chunk_index=self._chunk_index, frames=frames
        )

    def reset(self) -> None:
        """Remove the current video's history and caches, keeping weights resident."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None
        self._chunk_index = 0
