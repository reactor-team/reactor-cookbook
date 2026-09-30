# ──────────────────────────────────────────────────────────────────────────
# Camera video tracks.
#
# Each published view is a native SDK Track fed from the shared
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
import logging
import time
from typing import Callable

import numpy as np

log = logging.getLogger("libero_sim.tracks")


# How long the track may stay silent before repeating the last render.
HEARTBEAT_S = 0.5
# How often to check for a new render. Bounds the latency this adds per frame.
POLL_S = 0.002


class CameraTrack:
    """A sendonly video track that emits one frame per env render.

    ``reader`` returns ``(latest HxWx3 uint8 RGB frame or None, render
    sequence)``; see :meth:`loop.RolloutState.frame_reader`.
    """

    kind = "video"

    def __init__(
        self,
        name: str,
        reader: Callable[[], tuple[np.ndarray | None, int]],
    ):
        self.name = name
        self._reader = reader
        self._seq = -1
        self._publisher = None
        self._task: asyncio.Task | None = None

    async def start(self, reactor) -> None:
        self._publisher = await reactor.publish_track(self.name)
        self._task = asyncio.create_task(self._publish())

    async def _publish(self) -> None:
        try:
            while True:
                img = await self._next_render()
                if img is not None:
                    self._publisher.push_frame(img)
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
