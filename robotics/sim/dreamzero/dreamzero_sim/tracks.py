# ──────────────────────────────────────────────────────────────────────────
# Camera video tracks: queue-fed, and that is the whole design decision.
#
# The publisher waits until the gateway pushes a frame, so it emits exactly
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
# The SDK timestamps each native push; no synthetic fixed-rate timeline is
# imposed across the model's inference gaps.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging

import numpy as np

log = logging.getLogger("dreamzero_sim.tracks")


class QueueVideoTrack:
    """Outbound video track fed by a queue of RGB frames."""

    kind = "video"

    def __init__(self, name: str, queue_size: int = 8) -> None:
        self.name = name
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._publisher = None
        self._task: asyncio.Task | None = None
        #: Frames handed to the encoder: one per request per camera.
        self.frames_sent = 0

    async def push(self, frame: np.ndarray) -> None:
        """Enqueue one ``(H, W, 3)`` uint8 RGB frame.

        Validated rather than coerced: a float frame silently cast to uint8
        becomes near-black, and the model would accept it without complaint.
        """
        if self._task is not None and self._task.done():
            self._task.result()  # surface an encoder failure before accepting more frames
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

    async def start(self, reactor) -> None:
        self._publisher = await reactor.publish_track(self.name)
        self._task = asyncio.create_task(self._publish())

    async def _publish(self) -> None:
        try:
            while True:
                rgb = await self._queue.get()
                self._publisher.push_frame(rgb)
                self.frames_sent += 1
        except Exception:
            log.exception("native publisher %s failed", self.name)
            raise

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._publisher is not None:
            self._publisher.unpublish()
            self._publisher = None
        while not self._queue.empty():
            self._queue.get_nowait()
