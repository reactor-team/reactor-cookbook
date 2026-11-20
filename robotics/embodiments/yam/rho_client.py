# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Snapshot inference against hosted Rho; no simulation or robot drivers."""
from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Any

import numpy as np
from reactor_sdk import Reactor, time_micros

MODEL = "reactor/rho-yam-box"
TRACKS = ("scene_view", "left_wrist_view", "right_wrist_view")


def api_key() -> str:
    key = os.environ.get("REACTOR_API_KEY", "").strip()
    if not key:
        raise ValueError("Set REACTOR_API_KEY to your own Reactor API key")
    return key


def validate_poses(values: np.ndarray) -> None:
    for offset in (0, 10):
        x, y = values[..., offset + 3:offset + 6], values[..., offset + 6:offset + 9]
        if np.any(np.linalg.norm(x, axis=-1) < 1e-6) or np.any(
            np.linalg.norm(np.cross(x, y), axis=-1) < 1e-6
        ):
            raise ValueError("Degenerate 6D rotation columns")


def observation(frames: dict[str, np.ndarray], state: Any) -> tuple[dict, np.ndarray]:
    if set(frames) != set(TRACKS):
        raise ValueError(f"Expected camera names {TRACKS}")
    copied = {}
    for name, frame in frames.items():
        if not isinstance(frame, np.ndarray) or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
            raise ValueError(f"{name}: expected RGB uint8 (480, 640, 3)")
        copied[name] = frame.copy()
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError("Expected 20 finite state values")
    validate_poses(values)
    if np.any((values[[9, 19]] < 0) | (values[[9, 19]] > 1)):
        raise ValueError("Measured gripper openings must be in [0, 1]")
    return copied, values.copy()


def validate_reply(body: dict, request_id: int, stamp: int) -> np.ndarray:
    if body.get("step") != request_id or body.get("source_capture_us") != stamp:
        raise ValueError("Reply does not match the requested observation")
    if body.get("view_skew_us") != 0:
        raise ValueError("Camera views were not synchronized")
    actions = np.asarray(body.get("actions"), dtype=np.float64)
    if actions.shape != (50, 20) or not np.isfinite(actions).all():
        raise ValueError("Expected a finite 50 x 20 action chunk")
    if body.get("execution_horizon") != 25:
        raise ValueError("Unexpected Rho execution horizon")
    validate_poses(actions)
    return actions


class RhoClient:
    """One in-flight snapshot per session, with serialized request IDs.

    Snapshots are repeated with fresh transport timestamps to feed the video
    encoder. These timestamps do not assert that recorded images are live.
    A timeout or protocol error requires a new session; a late response must
    never be interpreted as the answer to a subsequent request.
    """

    def __init__(self, task: str, *, model: str = MODEL, timeout: float = 25):
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 300:
            raise ValueError("Task must contain 1–300 characters")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.task, self.model, self.timeout = task.strip(), model, timeout
        self.client = None
        self.tracks = {}
        self.messages: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.request_id = 0
        self.failed = False
        self.schema = None

    async def __aenter__(self) -> RhoClient:
        self.client = Reactor(self.model, api_key=api_key(), max_session_duration_seconds=600)

        @self.client.on_message
        def message(value):
            if isinstance(value, dict):
                self.messages.put_nowait(value)

        try:
            await asyncio.wait_for(self.client.connect(), 60)
            self.schema = await asyncio.wait_for(self.client.request_schema(), 15)
            operations = {v.get("post", {}).get("operationId")
                          for v in self.schema.get("paths", {}).values()}
            if not {"reset", "set_task_description", "set_state_json"} <= operations:
                raise ValueError("Runtime commands differ from the Rho contract")
            self.tracks = {name: await self.client.track(name).publish() for name in TRACKS}
            await asyncio.wait_for(self.client.send_command("reset", {}), 15)
            await asyncio.wait_for(self.client.send_command(
                "set_task_description", {"task_description": self.task}), 15)
            return self
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self.client is not None:
            client, self.client = self.client, None
            try:
                await asyncio.wait_for(client.disconnect(), 10)
            finally:
                client.close()

    async def __aexit__(self, *args) -> None:
        await self.close()

    def publish(self, frames: dict) -> int:
        stamp = time_micros()
        for name, track in self.tracks.items():
            track.push_frame(frames[name], capture_time_us=stamp)
        return stamp

    async def predict(self, frames: dict[str, np.ndarray], state: Any, *, seed: int = 0) -> dict:
        frames, state = observation(frames, state)
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("Seed must be an integer in [0, 2**31)")
        async with self.lock:
            if self.client is None or self.failed:
                raise RuntimeError("Open a fresh Rho session before predicting")
            try:
                return await self._predict(frames, state, seed)
            except BaseException:
                self.failed = True
                raise

    async def _predict(self, frames: dict, state: np.ndarray, seed: int) -> dict:
        loop = asyncio.get_running_loop()
        started = loop.time()
        # Repeat this frozen snapshot to warm the transport encoder.
        for _ in range(12):
            stamp = self.publish(frames)
            await asyncio.sleep(1 / 30)
        request = {"proprio": state.tolist(), "chunk_id": self.request_id,
                   "capture_us": stamp, "seed": seed}
        requested = loop.time()
        await asyncio.wait_for(self.client.send_command(
            "set_state_json", {"state_json": json.dumps(request)}), self.timeout)
        deadline = loop.time() + self.timeout
        while loop.time() < deadline:
            self.publish(frames)
            try:
                value = await asyncio.wait_for(self.messages.get(), min(1 / 30, deadline - loop.time()))
            except TimeoutError:
                continue
            if value.get("type") == "command_error":
                raise RuntimeError(str(value.get("data")))
            if value.get("type") != "action_prediction":
                continue
            body = value.get("data", {})
            actions = validate_reply(body, self.request_id, stamp)
            self.request_id += 1
            invalid = np.flatnonzero(((actions[:, [9, 19]] < 0) |
                                      (actions[:, [9, 19]] > 1)).any(axis=1)).tolist()
            return {"model": self.model, "session_id": self.client.session_id,
                    "task": self.task, "request": request, "prediction": body,
                    "invalid_gripper_rows": invalid,
                    "normalization": "production; no client remapping",
                    "timing": {"snapshot_to_reply_wall_s": loop.time() - started,
                               "request_to_reply_wall_s": loop.time() - requested},
                    "input_mode": "frozen snapshot with replay transport timestamps",
                    "actions_executed": 0}
        raise TimeoutError("Rho produced no prediction before the deadline")
