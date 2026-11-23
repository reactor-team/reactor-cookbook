"""What a talking-avatar model gives the application that drives it.

An avatar model renders a short block of video from a block of speech, one
block per call, and keeps the clip's state between calls. The application
(:mod:`avatar_app`) owns everything else: the speech arriving from the room,
the playout clock, the playback reports and the commands. This module is the
whole contract between the two halves. It is plain data, and it imports
nothing from ``reactor_runtime``, so a model can be built and tested without
the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AvatarInput:
    """What one block needs.

    Attributes:
        audio: The speech this block lip-syncs to, int16 ``(block_samples,)`` mono at the model's
            ``sample_rate``. Zeros for a block with nothing to say.
        silent: Whether the block carries no speech.
        listening: Whether the avatar is listening to the user rather than speaking. A model without a
            listening pose renders it like any silent block.
        avatar: The portrait to bind, PNG or JPEG bytes, or ``None`` when the application has no new one.
        avatar_id: Identifies the portrait. A block whose ``avatar_id`` differs from the one the model
            last bound starts a fresh clip from ``avatar``.
    """

    audio: np.ndarray
    silent: bool
    listening: bool
    avatar: bytes | None
    avatar_id: int


@dataclass(frozen=True)
class AvatarResult:
    """What one block produced.

    Attributes:
        frames: The rendered frames, RGB uint8 ``(N, height, width, 3)``. ``N`` is at most the model's
            ``block_frames``; the frames play with the first ``N`` frames' worth of the block's audio.
        block: The model's count of this block within the current clip, from 0.
        avatar_id: The id of the portrait the clip was bound from.
        timing: The model's own timing of the block in milliseconds, by stage, when it measures one.
    """

    frames: np.ndarray
    block: int
    avatar_id: int
    timing: dict[str, float] | None = None


class AvatarModel(Protocol):
    """A model the avatar application drives.

    ``generate`` runs on one worker thread, one block at a time; ``reset`` runs on that thread too,
    between blocks.
    """

    #: Video frames per second.
    fps: int
    #: Video frames one block advances.
    block_frames: int
    #: Rate of the speech a block carries.
    sample_rate: int

    def generate(self, input: AvatarInput) -> AvatarResult:
        """Render the next block of the clip, binding ``input.avatar`` first when the portrait changed."""
        ...

    def reset(self) -> None:
        """Forget the clip, so the next ``generate`` binds its portrait again."""
        ...
