# ──────────────────────────────────────────────────────────────────────────
# Camera video tracks: queue-fed, and that is the whole design decision.
#
# `recv` blocks until the gateway pushes a frame, so the track emits EXACTLY
# the frames the evaluation produced and nothing else. That matters here in a
# way it does not in the repo's other examples:
#
#   The model consumes the 4 newest frames per camera as its temporal
#   context. Each request pushes exactly one frame per camera, once, so the
#   window holds the last 4 requests. At RoboLab's --open-loop-horizon of 24
#   sim steps, that is -72/-48/-24/0, the shape the checkpoint was trained
#   for. A track that repeated its frame at a steady rate to keep RTP
#   flowing would fill the window with four copies of the newest observation
#   and quietly delete the model's temporal context.
#
# So there is no heartbeat here. cosmos_droid_sim can heartbeat freely (its
# model keeps only the newest frame per view); this one cannot. The cost is
# that RTP goes silent between requests, which is why bridge.py connects
# lazily; see its header.
#
# Timestamps come from the wall clock rather than a fixed frame rate, because
# the gaps between pushes are the model's own inference time and inventing a
# steady timeline across them would misdate every frame.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import time

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
from av import VideoFrame


class QueueVideoTrack(MediaStreamTrack):
    """Outbound video track fed by a queue of RGB frames."""

    kind = "video"

    def __init__(self, name: str, queue_size: int = 8) -> None:
        super().__init__()
        self.name = name
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._start: float | None = None
        self._last_pts = -1
        #: Frames handed to the encoder: one per request per camera.
        self.frames_sent = 0

    async def push(self, frame: np.ndarray) -> None:
        """Enqueue one ``(H, W, 3)`` uint8 RGB frame.

        Validated rather than coerced: a float frame silently cast to uint8
        becomes near-black, and the model would accept it without complaint.
        """
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"track {self.name!r}: expected (H, W, 3) RGB, got {arr.shape}"
            )
        if arr.dtype != np.uint8:
            raise TypeError(
                f"track {self.name!r}: expected uint8 RGB frames, got {arr.dtype}"
            )
        await self._queue.put(np.ascontiguousarray(arr))

    async def recv(self) -> VideoFrame:
        rgb = await self._queue.get()

        now = time.monotonic()
        if self._start is None:
            self._start = now
        pts = int((now - self._start) * VIDEO_CLOCK_RATE)
        # Strictly increasing, or the encoder drops the frame.
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts

        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = pts
        frame.time_base = VIDEO_TIME_BASE
        self.frames_sent += 1
        return frame
