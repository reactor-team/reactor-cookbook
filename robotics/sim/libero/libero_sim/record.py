# ──────────────────────────────────────────────────────────────────────────
# Optional MP4 export of the published camera views, side by side: visual
# proof a run executed real, model-driven steps rather than just exchanging
# network frames. Off by default (--record PATH). Samples frame_source() on
# a fixed wall-clock schedule so the clip plays back at real time, holds and
# all. That's the honest record of what happened, not just the ticks that
# rendered something new. Scheduled against an absolute deadline rather than
# a plain time.sleep(period) loop, which drifts over the length of a run.
# `active_only` keeps that schedule but drops non-RUNNING ticks, so the
# lock-step holds between chunks collapse and the clip is pure policy motion.
#
# `status`, if given, is polled once per sampled frame and its return value
# (CONNECTING/WAITING/RUNNING) is burned into the frame. The clip is the
# only artifact a viewer has after the fact, so the state has to travel with
# the pixels rather than live in a log they don't have.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import threading
import time
from collections.abc import Callable

import cv2
import imageio.v2 as imageio
import numpy as np

_STATUS_COLOR = {
    "CONNECTING": (255, 200, 0),
    "WAITING": (255, 80, 80),
    "RUNNING": (80, 220, 80),
}


class FrameRecorder:
    def __init__(
        self,
        path: str,
        sources: dict[str, Callable[[], np.ndarray | None]],
        hz: float,
        status: Callable[[], str] | None = None,
        overlay: bool = False,
        active_only: bool = False,
    ):
        self._path = path
        self._sources = sources
        self._hz = hz
        self._status = status
        self._overlay = overlay
        # active_only: drop non-RUNNING ticks so the lock-step holds between
        # chunks collapse and the clip is continuous policy-driven motion.
        # Needs `status` to gate on, independent of whether `overlay` draws it.
        self._active_only = active_only
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        period = 1.0 / self._hz
        next_tick = time.monotonic()
        with imageio.get_writer(
            self._path, fps=self._hz, macro_block_size=None,
            quality=8, output_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            while not self._stop.is_set():
                frames = [get() for get in self._sources.values()]
                if all(f is not None for f in frames):
                    state = self._status() if self._status is not None else None
                    if not (self._active_only and state != "RUNNING"):
                        frame = np.concatenate(frames, axis=1)
                        if self._overlay and state is not None:
                            _burn_status(frame, state)
                        writer.append_data(frame)
                next_tick += period
                self._stop.wait(max(0.0, next_tick - time.monotonic()))


def _burn_status(frame: np.ndarray, text: str) -> None:
    color = _STATUS_COLOR.get(text, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (14 + 9 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
