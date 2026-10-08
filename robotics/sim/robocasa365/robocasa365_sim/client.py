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
import json
import logging
import queue
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Keep camera identity and history-slot ordering across the three named tracks.
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

# The native SDK owns transport keepalive and encoding.
_FPS = 20

import os as _os


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
        self._recycle_after = int(_os.environ.get("XR1_CLIENT_SESSION_RECYCLE", "600"))

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        try:
            self._run(self._connect())
        except BaseException:
            self.close()
            raise

    # -- asyncio side ---------------------------------------------------------

    def _run(self, coro, *, timeout_s=None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(
                timeout=self._timeout + 5 if timeout_s is None else timeout_s
            )
        except FutureTimeoutError:
            future.cancel()
            raise

    async def _connect(self) -> None:
        from reactor_sdk import Reactor

        self._reactor = Reactor(self._model, api_url=self._api_url, local=True)

        @self._reactor.on_status
        def on_status(status) -> None:
            logger.info("[reactor] status=%s", status)

        @self._reactor.on_message
        def on_message(message: Any) -> None:
            self._msgs.put(message)

        @self._reactor.on_error
        def on_error(err: Any) -> None:
            logger.error("[reactor] error: %s", err)

        await asyncio.wait_for(self._reactor.connect(), timeout=self._timeout)
        self._tracks = {}
        for name in TRACK_ORDER:
            self._tracks[name] = await self._reactor.publish_track(name)
        logger.info("[reactor] connected; tracks published: %s", ", ".join(TRACK_ORDER))

    async def _push_frames(self, images: dict[str, list[np.ndarray]]) -> None:
        from reactor_sdk import time_micros

        # Pace history slots to avoid dropping intermediate frames in the encoder.
        n = len(images[CAMERA_ORDER[0]])
        for i in range(n):
            captured = time_micros()
            # All three cameras for slot i, then the gap. The model pairs the
            # tracks frame-for-frame in arrival order, so the three must be
            # pushed as a set: a camera that skips a slot shifts its whole
            # history against the other two.
            for cam in CAMERA_ORDER:
                frame = np.ascontiguousarray(np.asarray(images[cam][i], dtype=np.uint8))
                self._tracks[TRACK_FOR_CAMERA[cam]].push_frame(
                    frame, capture_time_us=captured
                )
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
            if action.shape != (16, 60) or not np.isfinite(action).all():
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

    async def _disconnect(self) -> None:
        reactor = getattr(self, "_reactor", None)
        if reactor is not None:
            try:
                await asyncio.wait_for(reactor.disconnect(), timeout=10)
            finally:
                reactor.close()

    def _recycle(self) -> None:
        """Tear down and re-establish the WebRTC session in place.

        A fresh session resets per-session state on BOTH ends (client transport
        + model session); step echo, task description, and the
        message queue are re-primed so the next infer() is indistinguishable
        from a first one."""
        logger.info("[reactor] recycling session at step %d", self._echo_step)
        try:
            self._run(self._disconnect())
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
        if not self._thread.is_alive():
            return
        try:
            self._run(self._disconnect())
        except Exception as exc:  # noqa: BLE001 - teardown best-effort
            logger.warning("disconnect failed: %s", exc)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
