# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""A drop-in replacement for the RoboCasa365 vendor `EvalClient`.

Talks to a Reactor-served `xr1-robocasa365` through the Reactor python SDK
instead of the vendor's raw TCP socket. The vendor rollout loop runs
UNMODIFIED: this class matches its client interface, so `entry.py` swaps only
the transport.

Vendor-identical conditioning over a streaming transport. The model is
launched with `obs_interval: 1`, so its per-view history is exactly
obs_history (4) deep at stride 1. Each `infer()` pushes the loop's own
sampled 4-frame histories, after which the model holds exactly those four
frames, the same set the socket client would have sent. Older frames,
including a previous episode's, are evicted by construction, so no reset is
needed between episodes. State history passes through verbatim as JSON.

What does differ from the socket path, deliberately, is the video codec:
frames reach the model H264-compressed rather than lossless.

Threading: the sim loop is synchronous and the SDK is asyncio, so this hosts
a dedicated event-loop thread and bridges with run_coroutine_threadsafe.
"""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import queue
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# The model consumes ONE stacked track: [left|right|wrist] side by side
# (see xr1-robocasa365/model_types.py for why). Stack in this exact order.
CAMERA_ORDER = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
# One named track per camera, matching xr1-robocasa365/model_types.py. Order
# matters: it is the order the views enter the model's prompt template.
TRACK_FOR_CAMERA = {
    "video.robot0_agentview_left": "left_agentview",
    "video.robot0_agentview_right": "right_agentview",
    "video.robot0_eye_in_hand": "wrist_view",
}
TRACK_ORDER = tuple(TRACK_FOR_CAMERA[c] for c in CAMERA_ORDER)

# The runtime disconnects a client that sends nothing for 20 s, and
# reactor-sdk 0.8.0 leaves keepalive to the client. RoboCasa spends
# longer than that building the next environment between tasks, so
# without this the session dies in the gap. Ping at half the watchdog.
_PING_INTERVAL_S = 10.0

_VIDEO_CLOCK = 90_000
_FPS = 20

# Pin the H264 encoder bitrate, per track. aiortc's defaults (1 Mbps, hard
# max 3 Mbps) visibly degrade the observations, at a measured cost of ~25
# points of episode success, concentrated in visual-state tasks. On
# localhost/LAN there is no reason to starve the encoder.
import os as _os

_BITRATE = int(_os.environ.get("XR1_EVAL_H264_BITRATE", "10000000"))


def _pin_h264_bitrate() -> None:
    import aiortc.codecs.h264 as _h264

    _h264.DEFAULT_BITRATE = _h264.MIN_BITRATE = _h264.MAX_BITRATE = _BITRATE


_pin_h264_bitrate()


class _QueueVideoTrack:  # MediaStreamTrack built lazily to keep import light
    pass


def _make_track_class():
    import av
    from aiortc import VideoStreamTrack

    class QueueVideoTrack(VideoStreamTrack):
        """Sendonly track fed frame-by-frame from an asyncio queue.

        recv() blocks on the queue, so frames go on the wire exactly when
        the eval pushes them (sim time, not wall-clock); pts advance at a
        nominal FPS purely for codec bookkeeping.
        """

        def __init__(self) -> None:
            super().__init__()
            self.q: asyncio.Queue = asyncio.Queue()
            self._pts = 0

        async def recv(self):
            arr = await self.q.get()
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = self._pts
            frame.time_base = fractions.Fraction(1, _VIDEO_CLOCK)
            self._pts += _VIDEO_CLOCK // _FPS
            return frame

    return QueueVideoTrack


class ReactorEvalClient:
    """Vendor-`EvalClient`-compatible client over the Reactor SDK.

    infer(states, images, instruction) -> np.ndarray[horizon, 12]
    """

    ACTION_DIM = 12  # dims the benchmark embodiment consumes (vendor slice)

    def __init__(
        self,
        api_url: str,
        model: str = "xr1-robocasa365",
        settle_s: float = 0.05,  # receipt gate on the model side owns sync now
        chunk_timeout_s: float = 120.0,
    ) -> None:
        self._api_url = api_url
        self._model = model
        self._settle_s = settle_s
        self._timeout = chunk_timeout_s
        self._msgs: queue.Queue = queue.Queue()
        self._echo_step = 0
        self._last_chunk_step = -1
        self._task_sent: str | None = None
        self._latencies: list[float] = []
        # Sessions were observed to stall after roughly 1.4k predictions
        # (action_prediction messages stop arriving; a fresh session
        # recovers). Recycle the WebRTC session between infer() calls well
        # before that. Predictions are stateless server-side, so a recycle
        # is invisible to the eval loop. 0 disables.
        self._recycle_after = int(
            _os.environ.get("XR1_CLIENT_SESSION_RECYCLE", "600")
        )

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._run(self._connect())

    # -- asyncio side ---------------------------------------------------------

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _connect(self) -> None:
        from reactor_sdk import Reactor

        # sdk 0.8.0 local mode HARDCODES api_url to http://localhost:8080 --
        # override the private attribute so a non-default port / remote box
        # works. Multi-video-in needs no runtime-side patch on 3.x: the
        # libwebrtc transport delivers each named track to its own input.
        self._reactor = Reactor(self._model, local=True)
        self._reactor._api_url = self._api_url
        ready = asyncio.Event()

        def on_status(status: str) -> None:
            logger.info("[reactor] status=%s", status)
            if str(status).lower().endswith("ready"):
                ready.set()

        def on_message(message: Any) -> None:
            self._msgs.put(message)

        def on_error(err: Any) -> None:
            logger.error("[reactor] error: %s", err)

        self._reactor.on("status_changed", on_status)
        self._reactor.on("message", on_message)
        self._reactor.on("error", on_error)

        await self._reactor.connect()
        await asyncio.wait_for(ready.wait(), timeout=120)

        track_cls = _make_track_class()
        self._tracks = {name: track_cls() for name in TRACK_ORDER}
        for name in TRACK_ORDER:
            await self._reactor.publish_track(name, self._tracks[name])
        logger.info("[reactor] connected; tracks published: %s", ", ".join(TRACK_ORDER))

        async def _keepalive() -> None:
            from reactor_sdk.types import MessageScope

            while True:
                try:
                    await self._reactor.send_command(
                        "ping", {}, scope=MessageScope.RUNTIME
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - transport hiccup
                    logger.warning("[reactor] keepalive ping failed: %s", exc)
                await asyncio.sleep(_PING_INTERVAL_S)

        self._keepalive = asyncio.ensure_future(_keepalive())

    async def _push_frames(self, images: dict[str, list[np.ndarray]]) -> None:
        # Stack per history slot: images[cam][i] -> one [H, 3W, 3] frame.
        # Frames are pushed SPACED (not burst): the receive side paces frame
        # release by RTP timestamp, so a burst of 4 frames 50ms apart in pts
        # can emerge over ~200ms -- racing the echo and letting the model
        # predict on a clamp-padded stale history (prime suspect for the
        # replan16 reactor gap; see the report).
        n = len(images[CAMERA_ORDER[0]])
        for i in range(n):
            # All three cameras for slot i, then the gap. The model pairs the
            # tracks frame-for-frame in arrival order, so the three must be
            # pushed as a set: a camera that skips a slot shifts its whole
            # history against the other two.
            for cam in CAMERA_ORDER:
                frame = np.ascontiguousarray(np.asarray(images[cam][i], dtype=np.uint8))
                self._tracks[TRACK_FOR_CAMERA[cam]].q.put_nowait(frame)
            if i < n - 1:
                await asyncio.sleep(1.0 / _FPS)

    async def _send(self, command: str, data: dict) -> None:
        await self._reactor.send_command(command, data)

    # -- sync surface (called from the sim loop) --------------------------------

    def infer(
        self,
        state_history: np.ndarray,
        image_history: dict[str, list[np.ndarray]],
        instruction: str,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        state_history = np.asarray(state_history, dtype=np.float32)

        if self._recycle_after and self._echo_step >= self._recycle_after:
            self._recycle()

        if instruction != self._task_sent:
            self._run(
                self._send("set_task_description", {"task_description": instruction})
            )
            self._task_sent = instruction

        self._run(
            self._send(
                "set_state_history_json",
                {
                    "state_history_json": json.dumps(
                        {"state_history": state_history.tolist()}
                    )
                },
            )
        )
        dump_dir = _os.environ.get("XR1_DEBUG_DUMP_CLIENT")
        if dump_dir and self._echo_step < 24:
            from PIL import Image

            _os.makedirs(dump_dir, exist_ok=True)
            for k in range(len(image_history[CAMERA_ORDER[0]])):
                for cam in CAMERA_ORDER:
                    view = TRACK_FOR_CAMERA[cam]
                    Image.fromarray(
                        np.asarray(image_history[cam][k], dtype=np.uint8)
                    ).save(
                        f"{dump_dir}/pred{self._echo_step + 1:03d}_{view}_slot{k}.png"
                    )
        self._run(self._push_frames(image_history))
        # Let the pushed frames traverse encode -> wire -> decode -> input
        # buffer before the echo opens the gate (a couple of engine ticks).
        time.sleep(self._settle_s)

        step = self._echo_step
        self._echo_step += 1
        self._run(
            self._send(
                "set_executed_step_json",
                {"executed_step_json": json.dumps({"step": step})},
            )
        )

        chunk = self._await_chunk()
        self._latencies.append(time.perf_counter() - t0)
        return chunk[:, : self.ACTION_DIM]

    def _await_chunk(self) -> np.ndarray:
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no action_prediction within {self._timeout}s "
                    f"(last step {self._last_chunk_step})"
                )
            try:
                msg = self._msgs.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            payload = msg if isinstance(msg, dict) else None
            if not payload:
                continue
            if payload.get("type") != "action_prediction":
                continue
            data = payload.get("data") or {}
            step = int(data.get("step", -1))
            if step <= self._last_chunk_step:
                continue  # stale/duplicate
            self._last_chunk_step = step
            action = np.asarray(data["action"], dtype=np.float32)
            if action.ndim != 2:
                raise RuntimeError(f"bad action shape {action.shape}")
            return action

    def latency_stats(self) -> dict:
        if not self._latencies:
            return {}
        lat = np.asarray(self._latencies)
        return {
            "n": int(lat.size),
            "p50_ms": float(np.percentile(lat, 50) * 1e3),
            "p95_ms": float(np.percentile(lat, 95) * 1e3),
            "mean_ms": float(lat.mean() * 1e3),
        }

    def _stop_keepalive(self) -> None:
        task = getattr(self, "_keepalive", None)
        if task is not None:
            self._loop.call_soon_threadsafe(task.cancel)
            self._keepalive = None

    def _recycle(self) -> None:
        """Tear down and re-establish the WebRTC session in place.

        A fresh session resets per-session state on BOTH ends (client aiortc
        + runtime GStreamer/session); step echo, task description, and the
        message queue are re-primed so the next infer() is indistinguishable
        from a first one."""
        logger.info("[reactor] recycling session at step %d", self._echo_step)
        self._stop_keepalive()
        try:
            self._run(self._reactor.disconnect())
        except Exception as exc:  # noqa: BLE001 - old session may be wedged
            logger.warning("[reactor] recycle disconnect failed: %s", exc)
        self._echo_step = 0
        self._last_chunk_step = -1
        self._task_sent = None
        while True:
            try:
                self._msgs.get_nowait()
            except queue.Empty:
                break
        self._run(self._connect())

    def close(self) -> None:
        self._stop_keepalive()
        try:
            self._run(self._reactor.disconnect())
        except Exception as exc:  # noqa: BLE001 - teardown best-effort
            logger.warning("disconnect failed: %s", exc)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
