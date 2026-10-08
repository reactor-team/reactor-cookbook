# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Rho (YAM box) client: three cameras and the arm state in, one action chunk out.

The client sends one request at a time. Each request gets 50 absolute
end-effector targets for both arms. Execute the first 25, then ask again.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from reactor_sdk import Reactor, ReactorStatus

MODEL = "reactor/rho-yam-box"
# Track names, in the checkpoint's camera order.
VIEWS = ("scene_view", "left_wrist_view", "right_wrist_view")
# Per arm: x, y, z (3), 6D rotation (6), gripper (1). The left arm comes first.
ARM_DIM = 10
STATE_DIM = 2 * ARM_DIM
CHUNK_ROWS = 50
ACTION_SHAPE = (CHUNK_ROWS, STATE_DIM)
# The checkpoint config sets 15 control steps per second.
CONTROL_HZ = 15
TRACK_FPS = 15
TASK_MAX_CHARS = 300
SEED_MAX = 2**63 - 1


# --- State layout helpers -----------------------------------------------------


def rot6d_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return the 6D rotation: the first two columns of a 3x3 matrix, column by column."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    return rotation[:, :2].T.reshape(6)


def matrix_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix of a 6D rotation (Gram-Schmidt, as the model does)."""
    first, second = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    x = first / np.linalg.norm(first)
    y = second - np.dot(x, second) * x
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def arm_vector(xyz: np.ndarray, rotation: np.ndarray, gripper: float) -> np.ndarray:
    """Pack one arm as 10 values: x, y, z, 6D rotation, gripper."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError("xyz must have 3 values")
    return np.concatenate([xyz, rot6d_from_matrix(rotation), [float(gripper)]])


def pack_state(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Join two 10-value arm vectors into the 20-value state, left arm first."""
    return np.concatenate([np.asarray(left, np.float64), np.asarray(right, np.float64)])


def split_arms(row: np.ndarray) -> dict[str, dict[str, np.ndarray | float]]:
    """Split one 20-value state or action row into named parts for each arm."""
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (STATE_DIM,):
        raise ValueError(f"expected {STATE_DIM} values")
    arms = {}
    for side, arm in (("left", row[:ARM_DIM]), ("right", row[ARM_DIM:])):
        arms[side] = {
            "xyz": arm[0:3],
            "rotation": matrix_from_rot6d(arm[3:9]),
            "gripper": float(arm[9]),
        }
    return arms


def position_jump(state: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return the left and right position distance between a state and a target row."""
    state = np.asarray(state, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    left = np.linalg.norm(target[0:3] - state[0:3])
    right = np.linalg.norm(target[ARM_DIM : ARM_DIM + 3] - state[ARM_DIM : ARM_DIM + 3])
    return float(left), float(right)


# --- Validation ---------------------------------------------------------------


def validate_observation(frames: dict[str, np.ndarray], state: np.ndarray) -> None:
    """Raise ValueError for an observation that the model must not receive."""
    if set(frames) != set(VIEWS):
        raise ValueError(f"Provide exactly these RGB views: {VIEWS}")
    for name, frame in frames.items():
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) == 0
        ):
            raise ValueError(f"{name}: expected nonempty (H, W, 3) uint8 RGB")
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"state must contain {STATE_DIM} finite numbers")
    if np.any(np.abs(state) > np.finfo(np.float32).max):
        raise ValueError("state must fit finite float32 values")
    for side, offset in (("left", 0), ("right", ARM_DIM)):
        first, second = state[offset + 3 : offset + 9].reshape(2, 3)
        norms = np.linalg.norm(first) * np.linalg.norm(second)
        # The model cannot build a rotation from a zero or parallel column pair.
        if norms < 1e-12 or np.linalg.norm(np.cross(first, second)) < 1e-3 * norms:
            raise ValueError(f"{side} arm: 6D rotation columns are zero or parallel")


# --- Session ------------------------------------------------------------------


@dataclass
class Prediction:
    """One answered request.

    actions: (50, 20) absolute targets. Row k is for control step k + 1.
    execution_horizon: the number of leading rows to execute (25).
    """

    actions: np.ndarray
    step: int
    execution_horizon: int
    inference_seconds: float
    round_trip_ms: float

    @property
    def to_execute(self) -> np.ndarray:
        """Return the rows to execute before the next request."""
        return self.actions[: self.execution_horizon]


class RhoClient:
    """One session on reactor/rho-yam-box. Use it as an async context manager."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        api_url: str | None = None,
        settle_s: float = 0.2,
        timeout_s: float = 60,
    ) -> None:
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
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self._frames = {view: np.zeros((480, 640, 3), np.uint8) for view in VIEWS}
        self._tracks: dict = {}
        self._messages: asyncio.Queue = asyncio.Queue()
        self._publisher: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        # The id increases across reset, so a delayed old reply never matches.
        self._next_id = 0
        self._task: str | None = None

    async def __aenter__(self) -> RhoClient:  # noqa: PYI034 - supports Python 3.10
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()

        # Register the handlers before connect(): READY can arrive early.
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
            # A cold start can take several minutes.
            await asyncio.wait_for(self.reactor.connect(), timeout=900)
            await asyncio.wait_for(ready.wait(), timeout=900)
            # The SDK refuses a track before READY.
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
        """Stop the camera publisher and end the session. The session holds a GPU."""
        try:
            if self._publisher is not None:
                self._publisher.cancel()
                await asyncio.gather(self._publisher, return_exceptions=True)
                self._publisher = None
        finally:
            await self.reactor.disconnect()

    async def _publish(self) -> None:
        """Send the current frame of each view at TRACK_FPS, repeating it between requests."""
        try:
            while True:
                started = time.monotonic()
                for view, track in self._tracks.items():
                    track.push_frame(self._frames[view])
                await asyncio.sleep(
                    max(0, 1 / TRACK_FPS - (time.monotonic() - started))
                )
        except Exception as exc:  # noqa: BLE001 - predict() reports the failure
            self._messages.put_nowait(
                {
                    "type": "command_error",
                    "data": {"command": "publish_frame", "reason": str(exc)},
                }
            )

    async def reset(self) -> None:
        """Start a new episode. The task and the camera tracks stay."""
        async with self._lock:
            await self.reactor.send_command("reset", {})

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
        *,
        seed: int | None = None,
    ) -> Prediction:
        """Send one observation and return the action chunk for it."""
        state = np.asarray(state, dtype=np.float64)
        validate_observation(frames, state)
        if not isinstance(task, str) or not task.strip() or len(task) > TASK_MAX_CHARS:
            raise ValueError(
                f"task must be nonempty text of at most {TASK_MAX_CHARS} characters"
            )
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
            # The model answers only after each track delivers a frame newer
            # than the request. Give the new frames time to arrive first.
            await asyncio.sleep(self.settle_s)
            chunk_id = self._next_id
            self._next_id += 1
            body = {"proprio": state.tolist(), "chunk_id": chunk_id}
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
                    continue  # A delayed reply for another request. 3.0 == 3 here.
                return self._parse_reply(data, chunk_id, started)

    @staticmethod
    def _parse_reply(data: dict, chunk_id: int, started: float) -> Prediction:
        """Check one action_prediction payload and return it as a Prediction."""
        actions = np.asarray(data.get("actions"), dtype=np.float64)
        if actions.shape != ACTION_SHAPE or not np.isfinite(actions).all():
            raise ValueError(f"Expected a finite {ACTION_SHAPE} action chunk")
        # The data channel can deliver an integer field as a float, for example 25.0.
        horizon = data.get("execution_horizon")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, float))
            or not float(horizon).is_integer()
            or not 1 <= horizon <= CHUNK_ROWS
        ):
            raise ValueError("Invalid execution_horizon")
        seconds = data.get("inference_seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Invalid model inference time")
        return Prediction(
            actions,
            chunk_id,
            int(horizon),
            float(seconds),
            (time.perf_counter() - started) * 1000,
        )
