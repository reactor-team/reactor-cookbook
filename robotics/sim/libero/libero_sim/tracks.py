# ──────────────────────────────────────────────────────────────────────────
# Camera video tracks.
#
# Each published view is an aiortc VideoStreamTrack sourced from the shared
# latest-frame slot the sim renders into (loop.py). Unlike a track that
# samples at a fixed rate, this one emits exactly ONE frame per render: the
# policy pairs one frame with one action (its training data is 1:1), so
# repeating the last render while the sim waits for the next chunk would feed
# it a still video alongside a moving arm.
#
# The sim renders in bursts (16 frames while it executes a chunk, then
# nothing for as long as inference takes), so the track goes quiet between
# chunks. HEARTBEAT_S bounds that silence: a multi-second RTP gap risks the
# receiver's decoder stalling or waiting on a keyframe. A heartbeat repeat is
# nearly free: it lands while the engine is busy inferring, and the engine
# keeps only the newest frame per view.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
from av import VideoFrame

from .contract import CAM_SIZE

# How long the track may stay silent before repeating the last render.
HEARTBEAT_S = 0.5
# How often to check for a new render. Bounds the latency this adds per frame.
POLL_S = 0.002


class CameraTrack(VideoStreamTrack):
    """A sendonly video track that emits one frame per env render.

    ``reader`` returns ``(latest HxWx3 uint8 RGB frame or None, render
    sequence)``; see :meth:`loop.RolloutState.frame_reader`.
    """

    kind = "video"

    def __init__(
        self,
        name: str,
        reader: Callable[[], tuple[np.ndarray | None, int]],
        size: int = CAM_SIZE,
    ):
        super().__init__()
        self.name = name
        self._reader = reader
        self._size = size
        self._seq = -1
        self._t0: float | None = None
        self._last_pts = -1

    async def recv(self) -> VideoFrame:
        img = await self._next_render()
        if img is None:
            img = np.zeros((self._size, self._size, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts, frame.time_base = self._timestamp(), VIDEO_TIME_BASE
        return frame

    async def _next_render(self) -> np.ndarray | None:
        """Wait for a render newer than the last one sent, or for the heartbeat."""
        # asyncio.sleep rather than a threading primitive: the sim signals
        # from the main thread while this runs on the bridge thread's loop,
        # and polling a lock-guarded counter keeps that hand-off one-way (see
        # loop.RolloutState).
        deadline = time.monotonic() + HEARTBEAT_S
        while True:
            img, seq = self._reader()
            if seq != self._seq:
                self._seq = seq
                return img
            if time.monotonic() >= deadline:
                return img  # heartbeat: resend the last render, keep RTP flowing
            await asyncio.sleep(POLL_S)

    def _timestamp(self) -> int:
        """A pts from the wall clock, not a fixed frame counter.

        Frames leave in bursts, so counting 1/30s per frame (what aiortc's
        next_timestamp does) would claim a steady 30 fps and drift ever
        further behind real time.
        """
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        pts = max(int((now - self._t0) * VIDEO_CLOCK_RATE), self._last_pts + 1)
        self._last_pts = pts
        return pts
