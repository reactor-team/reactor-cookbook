# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""A video track that repeats one frame until you replace it.

Robotics policies on Reactor read their cameras from continuous video tracks,
but an observation only changes when the robot (or the recorded example file)
has a new one. So the track's job is to keep RTP flowing at a steady rate while
holding the current observation:

    track.set_frame(rgb_uint8)   # new observation
    ...                          # track keeps sending it at `fps`

Repetition is what makes the request/reply pairing work: the model answers a
request only once **every** view has delivered a frame that arrived after the
request, so a view whose observation has not changed still has to deliver
something, or the request never gets answered. See
``robot-policy-client-contract.md`` ("Video tracks", "Lock-step semantics").

``fps`` is honoured exactly rather than inheriting aiortc's fixed 30 fps
``next_timestamp()`` cadence, because it also sets the settle interval the
callers wait out before sending a request.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
from av import VideoFrame


class RepeatingFrameTrack(VideoStreamTrack):
    """Sendonly track emitting the current frame at a steady ``fps``.

    Starts on a black frame so the track is live (and negotiable) before the
    first observation exists.
    """

    kind = "video"

    def __init__(
        self,
        name: str,
        *,
        fps: int = 15,
        size: tuple[int, int] = (240, 320),
    ) -> None:
        super().__init__()
        self.name = name
        self.fps = int(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self._frame = np.zeros((*size, 3), dtype=np.uint8)
        #: Count of observations pushed. The scripts assert this advanced.
        self.pushes = 0
        self._t0: float | None = None
        self._sent = 0

    def set_frame(self, rgb: np.ndarray) -> None:
        """Replace the repeating frame with an ``(H, W, 3)`` uint8 RGB frame.

        Validates rather than coerces: a float frame in [-1, 1] cast to uint8
        becomes almost-black, and the server accepts it.
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
        self._frame = np.ascontiguousarray(arr)
        self.pushes += 1

    async def recv(self) -> VideoFrame:
        await self._pace()
        frame = VideoFrame.from_ndarray(self._frame, format="rgb24")
        frame.pts = int(self._sent * VIDEO_CLOCK_RATE / self.fps)
        frame.time_base = VIDEO_TIME_BASE
        self._sent += 1
        return frame

    async def _pace(self) -> None:
        """Sleep until this frame's slot in a steady ``fps`` schedule.

        Anchored to a start time rather than sleeping ``1/fps`` per frame, so
        encode time does not accumulate into drift.
        """
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
            return
        target = self._t0 + self._sent / self.fps
        delay = target - now
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -1.0:
            # Fell more than a second behind (a stalled event loop): re-anchor
            # instead of bursting a second of backlog at the encoder.
            self._t0 = now - self._sent / self.fps
