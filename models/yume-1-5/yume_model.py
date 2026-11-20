"""Runtime-independent YUME rolling-latent model and CPU step contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from yume_assets import YumeConfig
    from yume_backend import YumeBackend

Movement = Literal[
    "none",
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
]
View = Literal[
    "none",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "tilt_up_left",
    "tilt_up_right",
    "tilt_down_left",
    "tilt_down_right",
]


@dataclass(frozen=True)
class YumeAnchor:
    """New world's unresized RGB uint8 image (H,W,3) or video (33,H,W,3)."""

    mode: str
    media: np.ndarray | None
    seed: int


@dataclass(frozen=True)
class YumeInput:
    """One continuation; anchor stays present until a result acknowledges world_id."""

    world_id: int
    anchor: YumeAnchor | None
    prompt: str
    movement: Movement
    view: View


@dataclass(frozen=True)
class YumeResult:
    """Actual progress and CPU uint8 RGB frames (29,H,W,3)."""

    world_id: int
    chunk_index: int
    frames: np.ndarray
    complete: bool


class NoAnchor(Exception):
    """A fresh YUME world needs its selected scene."""


class RolloutExhausted(Exception):
    """The configured, finite rollout has completed."""


class YumeModel:
    """Hold the YUME weights and step one world 29 frames at a time.

    A world is identified by ``YumeInput.world_id``. When the input names a
    world this model has not started, ``generate`` starts it from
    ``input.anchor`` and then produces that world's first chunk; an input that
    names the current world continues it. ``YumeResult.world_id`` reports the
    world the frames belong to, so the application knows when to stop sending
    the anchor.

    ``YumeResult.chunk_index`` is the one-based count of chunks in the current
    world. The world ends after ``YumeConfig.max_chunks`` chunks: the result
    that reaches the limit has ``complete`` set, and any later step on the same
    world raises ``RolloutExhausted``. A step that asks for a new world without
    an anchor, or with an image or video mode and no media, raises
    ``NoAnchor``.
    """

    def __init__(self) -> None:
        self._backend: YumeBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._max_chunks = 0

    def load(self, config: YumeConfig) -> None:
        """Load the weights the prepared ``config`` names."""
        from yume_backend import YumeBackend

        self._max_chunks = config.max_chunks
        self._backend = YumeBackend(config)

    def generate(self, input: YumeInput) -> YumeResult:
        """Produce the next chunk, starting a new world first when the input asks for one."""
        if self._backend is None:
            raise RuntimeError("YUME was not loaded")
        if input.world_id != self._world_id:
            anchor = input.anchor
            if anchor is None:
                raise NoAnchor("a new world requires a scene anchor")
            if anchor.mode != "text_to_video" and anchor.media is None:
                raise NoAnchor("image and video worlds require decoded media")
            self._backend.reset(
                image=anchor.media if anchor.mode == "image_to_video" else None,
                video=anchor.media if anchor.mode == "video_to_video" else None,
                seed=anchor.seed,
            )
            self._world_id = input.world_id
            self._chunk_index = 0
        if self._chunk_index >= self._max_chunks:
            raise RolloutExhausted("select or reset a scene to continue")
        frames = self._backend.generate_chunk(
            prompt=input.prompt, movement=input.movement, view=input.view
        )
        self._chunk_index += 1
        return YumeResult(
            world_id=self._world_id,
            chunk_index=self._chunk_index,
            frames=np.ascontiguousarray(frames),
            complete=self._chunk_index == self._max_chunks,
        )

    def reset(self) -> None:
        """Forget the current world and release its context; keep the weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
