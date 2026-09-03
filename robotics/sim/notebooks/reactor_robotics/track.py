# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""A video publisher that repeats one captured frame until it is replaced.

Robotics policies on Reactor read their cameras from continuous video tracks,
but an observation only changes when the robot (or the recorded example file)
has a new one. ``RepeatingFrameTrack`` holds that observation while
``ReactorSession`` pushes it at a steady rate.

The capture stamp belongs to the observation, not to each repeated send. All
views passed to ``ReactorSession.set_frames`` receive one shared stamp, so the
model can pair the views even when their pushes land a few microseconds apart.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VideoTrack(Protocol):
    """The current SDK surface used by this wrapper."""

    def push_frame(
        self,
        data: np.ndarray,
        *,
        capture_time_us: int | None = None,
    ) -> None: ...


class RepeatingFrameTrack:
    """Hold one RGB observation and push it through a bound SDK track."""

    kind = "video"

    def __init__(
        self,
        name: str,
        *,
        fps: int = 15,
        size: tuple[int, int] = (240, 320),
    ) -> None:
        self.name = name
        self.fps = int(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self._frame = np.zeros((*size, 3), dtype=np.uint8)
        self._capture_time_us: int | None = None
        self._track: VideoTrack | None = None
        #: Count of observations supplied by the caller.
        self.pushes = 0
        #: Count of frames sent on the wire, including repeated observations.
        self.sent = 0

    def bind(self, track: VideoTrack) -> None:
        """Bind the SDK track returned by ``Reactor.publish_track``."""
        self._track = track

    def unbind(self) -> None:
        self._track = None

    def set_frame(self, rgb: np.ndarray, *, capture_time_us: int | None = None) -> None:
        """Replace the current ``(H, W, 3)`` uint8 RGB observation.

        ``capture_time_us`` must come from ``reactor_sdk.time_micros``. When it
        is omitted, the observation is stamped when this method is called.
        """
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"track {self.name!r}: expected (H, W, 3) RGB, got {arr.shape}"
            )
        if arr.dtype != np.uint8:
            raise TypeError(
                f"track {self.name!r}: expected uint8 RGB frames, got "
                f"{arr.dtype}. If your frames are floats in [-1, 1], convert "
                "with np.clip(np.round((v + 1.0) * 127.5), 0, 255)"
                ".astype(np.uint8) first."
            )
        if capture_time_us is None:
            from reactor_sdk import time_micros

            capture_time_us = time_micros()
        if isinstance(capture_time_us, bool) or not isinstance(capture_time_us, int):
            raise TypeError("capture_time_us must be an integer")

        self._frame = np.ascontiguousarray(arr)
        self._capture_time_us = capture_time_us
        self.pushes += 1

    def push_current(self) -> None:
        """Push the held observation once with its original capture stamp."""
        if self._track is None:
            raise RuntimeError(f"track {self.name!r} is not published")
        self._track.push_frame(
            self._frame,
            capture_time_us=self._capture_time_us,
        )
        self.sent += 1
