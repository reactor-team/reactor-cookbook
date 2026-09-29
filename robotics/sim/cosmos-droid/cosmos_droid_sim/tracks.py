# ──────────────────────────────────────────────────────────────────────────
# Camera video tracks: same shape as the other examples' CameraTrack, with
# one difference in WHAT paces a frame: here a new frame exists once per
# RoboLab REQUEST (RoboLab only ships observations when it asks for a
# chunk, every ~2.1 s), not once per sim step. So each track emits exactly
# one frame per request and then heartbeats: a multi-second RTP silence
# risks the receiver's decoder stalling, and a heartbeat repeat is nearly
# free, because the engine keeps only the newest frame per view.
#
# The three views have different sizes (wrist is full-resolution, the two
# exterior views are half-size), so each track just publishes whatever the
# composite split produced, no resizing here.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np
from reactor_sdk import Track

HEARTBEAT_S = 0.5
POLL_S = 0.005

# Before the first request there is nothing to show; a small black frame
# keeps the track negotiable until RoboLab connects.
PLACEHOLDER_HW = (180, 320)


class CameraTrack:
    """A sendonly track that emits one frame per gateway request.

    ``reader`` returns ``(latest HxWx3 uint8 RGB frame or None, request
    sequence)``. See :meth:`gateway.GatewayState.frame_reader`.
    """

    kind = "video"

    def __init__(self, name: str, reader: Callable[[], tuple[np.ndarray | None, int]]):
        self.name = name
        self._reader = reader
        self._seq = -1

    async def pump(self, track: Track) -> None:
        """Feed paced RGB arrays to the SDK's native video sender."""
        while True:
            img = await self._next_request_frame()
            if img is None:
                img = np.zeros((*PLACEHOLDER_HW, 3), dtype=np.uint8)
            track.push_frame(img)

    async def _next_request_frame(self) -> np.ndarray | None:
        deadline = time.monotonic() + HEARTBEAT_S
        while True:
            img, seq = self._reader()
            if seq != self._seq:
                self._seq = seq
                return img
            if time.monotonic() >= deadline:
                return img  # heartbeat: repeat the last frame, keep RTP flowing
            await asyncio.sleep(POLL_S)
