# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""FLUX 0.3.0: one checkpoint per session, one outstanding prediction at a time."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from reactor_sdk import Reactor, ReactorStatus

CHECKPOINTS = ("base-bf16", "base-fp8", "gd-bf16", "gd-fp8", "sd-bf16", "sd-fp8")
VIEWS = ("wrist_view", "exterior_view_1", "exterior_view_2")
ACTION_SHAPE = (32, 8)
SEED_MAX = 2**63 - 1


def validate_observation(frames: dict[str, np.ndarray], proprio: np.ndarray) -> None:
    if set(frames) != set(VIEWS):
        raise ValueError(f"Provide exactly these RGB views: {VIEWS}")
    for name, frame in frames.items():
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) == 0
        ):
            raise ValueError(f"{name}: expected nonempty (H, W, 3) uint8 RGB")
    if proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError(
            "proprio must contain seven finite joint angles and one gripper value"
        )
    if np.any(np.abs(proprio) > np.finfo(np.float32).max):
        raise ValueError("proprio must fit finite float32 values")
    if not 0 <= proprio[7] <= 1:
        raise ValueError("gripper closed fraction must be in [0, 1]")


async def select_checkpoint(reactor: Reactor, checkpoint: str) -> dict:
    """Discover and confirm the pin; never silently fall back to another policy."""
    reply = await reactor.send_command("get_checkpoint", {})
    if not reply or reply.get("type") != "checkpoint_selected":
        raise RuntimeError(
            "Checkpoint discovery failed; this client requires FLUX 0.3.0"
        )
    status = reply.get("data") or {}
    if checkpoint not in status.get("available", []):
        raise ValueError(
            f"{checkpoint!r} unavailable; choices: {status.get('available', [])}"
        )
    if status.get("locked") and status.get("checkpoint") != checkpoint:
        raise RuntimeError(
            "Session already pinned to another checkpoint; create a new session"
        )
    reply = await reactor.send_command("select_checkpoint", {"checkpoint": checkpoint})
    selected = (reply or {}).get("data") or {}
    if (
        not reply
        or reply.get("type") != "checkpoint_selected"
        or selected.get("checkpoint") != checkpoint
        or selected.get("locked") is not True
    ):
        raise RuntimeError("Checkpoint selection was not confirmed")
    return selected


@dataclass
class Prediction:
    actions: np.ndarray
    step: int
    checkpoint: str
    inference_seconds: float
    round_trip_ms: float


class FluxClient:
    def __init__(
        self,
        checkpoint: str = "base-bf16",
        *,
        model: str = "reactor/flux3-action-droid",
        api_url: str | None = None,
        settle_s: float = 0.3,
        timeout_s: float = 120,
    ) -> None:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"Choose one of {CHECKPOINTS}")
        if not math.isfinite(settle_s) or settle_s < 0:
            raise ValueError("settle_s must be finite and nonnegative")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        key = os.environ.get("REACTOR_API_KEY")
        if not key:
            raise RuntimeError("Set REACTOR_API_KEY to a key with access to this model")
        self.reactor = Reactor(
            model,
            api_key=key,
            api_url=api_url
            or os.environ.get("REACTOR_API_URL")
            or "https://api.reactor.inc",
        )
        self.checkpoint = checkpoint
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self.available: list[str] = []
        self._frames = {view: np.zeros((360, 640, 3), np.uint8) for view in VIEWS}
        self._tracks: dict = {}
        self._messages: asyncio.Queue = asyncio.Queue()
        self._publisher: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._next_id = (
            0  # Remains monotonic across reset, rejecting delayed old replies.
        )
        self._task: str | None = None

    async def __aenter__(self) -> FluxClient:  # noqa: PYI034 - supports Python 3.10
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()

        @self.reactor.on_status
        def on_status(status):
            if status == ReactorStatus.READY:
                loop.call_soon_threadsafe(ready.set)

        @self.reactor.on_message
        def on_message(message):
            if isinstance(message, dict) and message.get("type") in (
                "action_prediction",
                "command_error",
            ):
                loop.call_soon_threadsafe(self._messages.put_nowait, message)

        try:
            await asyncio.wait_for(self.reactor.connect(), timeout=900)
            await asyncio.wait_for(ready.wait(), timeout=900)
            status = await select_checkpoint(self.reactor, self.checkpoint)
            self.available = status["available"]
            for view in VIEWS:
                self._tracks[view] = await self.reactor.publish_track(view)
            self._publisher = asyncio.create_task(self._publish())
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            if self._publisher is not None:
                self._publisher.cancel()
                await asyncio.gather(self._publisher, return_exceptions=True)
                self._publisher = None
        finally:
            await self.reactor.disconnect()

    async def _publish(self) -> None:
        try:
            while True:
                started = time.monotonic()
                for view, track in self._tracks.items():
                    track.push_frame(self._frames[view])
                await asyncio.sleep(max(0, 1 / 15 - (time.monotonic() - started)))
        except Exception as exc:  # noqa: BLE001 - surface background failures to predict()
            self._messages.put_nowait(
                {
                    "type": "command_error",
                    "data": {"command": "publish_frame", "reason": str(exc)},
                }
            )

    async def reset(self) -> None:
        """Reset episode memory, preserving checkpoint and monotonically increasing IDs."""
        async with self._lock:
            await self.reactor.send_command("reset", {})
            # Reset has an empty acknowledgement; confirm the checkpoint separately.
            status = await self.reactor.send_command("get_checkpoint", {})
            data = (status or {}).get("data") or {}
            if (
                data.get("checkpoint") != self.checkpoint
                or data.get("locked") is not True
            ):
                raise RuntimeError("Checkpoint pin was not preserved after reset")
            self._task = None

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        proprio: np.ndarray,
        task: str,
        *,
        seed: int | None = None,
    ) -> Prediction:
        proprio = np.asarray(proprio, dtype=np.float64)
        validate_observation(frames, proprio)
        if not isinstance(task, str) or not task.strip() or len(task) > 300:
            raise ValueError("task must be nonempty text of at most 300 characters")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= SEED_MAX):
            raise ValueError(f"seed must be an integer between 0 and {SEED_MAX}")
        async with self._lock:
            if self._publisher is None or self._publisher.done():
                raise RuntimeError(
                    "No active camera publisher; open a new client session"
                )
            if task != self._task:
                await self.reactor.send_command(
                    "set_task_description", {"task_description": task}
                )
                self._task = task
            self._frames = {
                view: np.array(frames[view], copy=True, order="C") for view in VIEWS
            }
            # Settling is a heuristic, NOT a frame-identity acknowledgement.
            await asyncio.sleep(self.settle_s)
            chunk_id = self._next_id
            self._next_id += 1
            body = {"proprio": proprio.tolist(), "chunk_id": chunk_id}
            if seed is not None:
                body["seed"] = seed
            started = time.perf_counter()
            await self.reactor.send_command(
                "set_state_json", {"state_json": json.dumps(body)}
            )
            deadline = started + self.timeout_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"No action_prediction for step {chunk_id}")
                message = await asyncio.wait_for(
                    self._messages.get(), timeout=remaining
                )
                data = message.get("data") or {}
                if message["type"] == "command_error":
                    raise RuntimeError(f"{data.get('command')}: {data.get('reason')}")
                if data.get("step") != chunk_id:
                    continue  # A delayed reply is not the answer to this request.
                actions = np.asarray(data.get("actions"), dtype=np.float64)
                if actions.shape != ACTION_SHAPE or not np.isfinite(actions).all():
                    raise ValueError("Expected a finite (32, 8) action chunk")
                if data.get("checkpoint") != self.checkpoint:
                    raise ValueError("Reply came from the wrong checkpoint")
                seconds = float(data["inference_seconds"])
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("Invalid model inference time")
                return Prediction(
                    actions,
                    chunk_id,
                    self.checkpoint,
                    seconds,
                    (time.perf_counter() - started) * 1000,
                )
