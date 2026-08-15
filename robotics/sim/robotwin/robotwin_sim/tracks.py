# ──────────────────────────────────────────────────────────────────────────
# Camera video tracks: one per view, each repeating the current observation
# at a steady rate until it is replaced.
#
# Repetition is not padding. The model answers a request only once EVERY
# view has delivered a frame that arrived after the request, so a view whose
# observation has not changed still has to deliver something or the request
# is never answered. That is also what makes the pairing deterministic: the
# reply belongs to the newest frame on every track, and the gateway does not
# have to guess.
#
# `fps` is real here, and it is load-bearing twice over: it paces the track,
# and it sets the settle interval bridge.py waits out before sending a
# request (a few track periods, so the swapped observation clears the
# encoder first).
# ──────────────────────────────────────────────────────────────────────────
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
    simulator has sent its first observation.
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
        #: Observations pushed to this track; the diagnostics report it.
        self.pushes = 0
        self._t0: float | None = None
        self._sent = 0

    def set_frame(self, rgb: np.ndarray) -> None:
        """Replace the repeating frame with an ``(H, W, 3)`` uint8 RGB frame.

        Validated rather than coerced: the client's own frames are floats in
        ``[-1, 1]``, and casting one of those to uint8 yields an almost-black
        image the model would accept without complaint. Convert with
        :func:`contract.frames_from_video` first.
        """
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"track {self.name!r}: expected (H, W, 3) RGB, got {arr.shape}"
            )
        if arr.dtype != np.uint8:
            raise TypeError(
                f"track {self.name!r}: expected uint8 RGB frames, got "
                f"{arr.dtype}; see contract.frames_from_video()"
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
            # More than a second behind (a stalled event loop): re-anchor
            # instead of bursting a second of backlog at the encoder.
            self._t0 = now - self._sent / self.fps
