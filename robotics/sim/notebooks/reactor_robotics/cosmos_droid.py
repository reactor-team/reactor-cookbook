# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Cosmos-Nano-Policy-DROID client: stateless, one chunk per executed-step report.

NVIDIA Cosmos3-Nano-Policy-DROID is a vision-language-action policy on the
DROID/Franka embodiment. Three camera views plus proprioception in, a
``(32, 8)`` chunk of absolute joint targets out.

    client = CosmosDroidClient()
    await client.connect()
    pred = await client.predict(frames, joints, gripper, task="put it in the bowl")
    pred.actions      # (32, 8): 7 absolute joint positions (rad) + gripper
    await client.close()

The working reference for this protocol is in this repo:
[`../cosmos-droid/`](../cosmos-droid) relays NVIDIA's RoboLab
benchmark to the same model through an openpi-compatible gateway. This module
drops the gateway and drives the model directly.

## Stateless, so there is no episode

No KV cache, no ``reset`` event on the wire at all. The task and the proprio
are sent with every prediction, so a new episode or a task change needs no
ceremony: just call :meth:`CosmosDroidClient.predict` with the new task.

## One chunk per executed-step report

| Step | Wire |
|---|---|
| on change | ``set_task_description {task_description}`` (≤300 chars) |
| per chunk | ``set_proprio_json {proprio_json}`` (≤8000 chars) |
| per chunk | ``set_executed_step_json {executed_step_json}`` (≤8000 chars) |
| per chunk | model → ``action_prediction {action: [32,8], step}`` |

The **first** prediction needs only task + proprio + a full frame set. Every
one after it needs an echo whose ``step`` is **strictly greater** than the last
one the model advanced on; anything else (a repeat, a lower value, malformed
JSON) produces no reply and no error. This stops the model running ahead of a
client that is still executing, and stops a stalled control loop's retry from
triggering a spurious re-prediction.

The reply's own ``step`` is the model's monotonic prediction counter, from 0.
Echo the ``step`` of the chunk you executed and the counters line up, which is
what :meth:`predict` does.

## Actions are absolute joint targets

7 absolute joint positions in radians plus a gripper command, in DROID's
joint-position convention. Rows are poses, not deltas.

## Timing

| | Value |
|---|---|
| chunk budget (32 rows at 15 Hz) | 2133 ms |
| expected model compute per chunk | ~568 ms |
| expected p50 think + wire | ~745 ms |

Whole-chunk, open-loop execution is the measured optimum for this policy
rather than a compromise: success strictly increases with open-loop horizon and
mid-chunk replanning collapses it. Both are reference figures for this
deployment, not guarantees; the p50 is also quoted in
[`../cosmos-droid/README.md`](../cosmos-droid/README.md).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .session import ReactorSession

#: Track names, in the checkpoint's declaration order (``wrist_image_left``,
#: ``exterior_image_1_left``, ``exterior_image_2_left``).
TRACKS: tuple[str, str, str] = ("wrist_view", "exterior_view_1", "exterior_view_2")

#: (action horizon, DoF). 7 absolute joint positions + 1 gripper.
ACTION_SHAPE: tuple[int, int] = (32, 8)

#: The DROID control rate. One chunk is 32 rows = 2133 ms of motion.
CONTROL_HZ = 15.0

#: Chunk budget in ms: how long executing one chunk takes, and therefore how
#: much time a client has to get the next one.
CHUNK_BUDGET_MS = 1000.0 * ACTION_SHAPE[0] / CONTROL_HZ

#: Reference figures for this deployment, not guarantees.
EXPECTED_COMPUTE_MS = 568.0
EXPECTED_WIRE_P50_MS = 745.0

#: ``InputField(max_length=...)`` on the three state fields.
TASK_MAX_LEN = 300
PROPRIO_MAX_LEN = 8000
EXECUTED_STEP_MAX_LEN = 8000

#: Franka Research 3 joint limits (radians), from the FR3 datasheet. Used by
#: the script to confirm the predicted joint targets are commandable.
FR3_JOINT_LIMITS = np.array(
    [
        (-2.7437, 2.7437),
        (-1.7837, 1.7837),
        (-2.9007, 2.9007),
        (-3.0421, -0.1518),
        (-2.8065, 2.8065),
        (0.5445, 4.5169),
        (-3.0159, 3.0159),
    ],
    dtype=np.float64,
)

#: DROID's reset pose (radians).
DROID_RESET_JOINTS = np.array(
    [0.0, -0.2 * np.pi, 0.0, -0.8 * np.pi, 0.0, 0.6 * np.pi, 0.0],
    dtype=np.float64,
)

log = logging.getLogger("reactor_robotics.cosmos_droid")


def encode_proprio(
    joints: Sequence[float] | np.ndarray, gripper: float
) -> str:
    """Build ``proprio_json`` for one timestep.

    Both keys are **lists of rows** so a client can send a short history in one
    update; the last row is the current state. The server treats anything it
    cannot parse (empty, bad JSON, a missing key, a wrong row width) as "no
    proprio yet" and simply does not predict, so this validates locally.
    """
    j = np.atleast_2d(np.asarray(joints, dtype=np.float64))
    g = np.asarray(gripper, dtype=np.float64).reshape(-1, 1)
    if j.ndim != 2 or j.shape[1] != 7:
        raise ValueError(f"joints must be (N, 7), got {j.shape}")
    if g.shape[1] != 1:
        raise ValueError(f"gripper must be (N, 1), got {g.shape}")
    for name, arr in (("joint_position", j), ("gripper_position", g)):
        if not np.isfinite(arr).all():
            raise ValueError(f"proprio key {name!r} is not finite")
    out = json.dumps(
        {"joint_position": j.tolist(), "gripper_position": g.tolist()},
        separators=(",", ":"),
    )
    if len(out) > PROPRIO_MAX_LEN:
        raise ValueError(
            f"proprio_json is {len(out)} chars, over the field's "
            f"{PROPRIO_MAX_LEN} limit"
        )
    return out


def encode_executed_step(step: int, rows: np.ndarray) -> str:
    """Build ``executed_step_json``.

    The model advances on ``step`` alone; the echoed rows report the chunk
    that was actually executed.
    """
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"executed rows have shape {arr.shape}, want (n, cols)")
    out = json.dumps(
        {"step": int(step), "action": arr.tolist()}, separators=(",", ":")
    )
    if len(out) > EXECUTED_STEP_MAX_LEN:
        raise ValueError(
            f"executed_step_json is {len(out)} chars, over the field's "
            f"{EXECUTED_STEP_MAX_LEN} limit ({arr.shape[0]} rows); echo fewer rows"
        )
    return out


@dataclass
class CosmosDroidPrediction:
    """One chunk."""

    actions: np.ndarray
    """``(32, 8)``: 7 absolute joint targets in radians + 1 gripper."""

    step: int
    """The model's monotonic prediction counter for this session, from 0."""

    latency_ms: float
    """Wall-clock from the executed-step report (or, for the first chunk, the
    proprio) to this chunk arriving. Excludes the frame settle."""

    discarded: list[int] = field(default_factory=list)
    """``step`` of any chunk dropped as stale before this one. At most one
    chunk is ever in flight, so this stays empty unless a previous request
    timed out and its answer arrived late."""

    @property
    def joint_position(self) -> np.ndarray:
        """``(32, 7)`` absolute joint targets."""
        return self.actions[:, :7]

    @property
    def gripper(self) -> np.ndarray:
        """``(32,)`` gripper command."""
        return self.actions[:, 7]


class CosmosDroidClient:
    """Client for the served ``cosmos-nano-policy-droid`` policy."""

    def __init__(
        self,
        *,
        model: str = "cosmos-nano-policy-droid",
        session: ReactorSession | None = None,
        fps: int = 15,
        frame_size: tuple[int, int] = (180, 320),
        settle_s: float | None = None,
        timeout_s: float = 90.0,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self.session = session or ReactorSession(
            model, fps=fps, frame_size=frame_size
        )
        # Frames are sent over video while the commands are sent over the data
        # channel, and the model pairs whichever frame is newest with the
        # request. A few track periods let the fresh frame land first. This
        # reduces the risk but does not remove it: none of these models tags a
        # chunk with the frame it came from.
        self.settle_s = settle_s if settle_s is not None else max(3.0 / fps, 0.2)
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self._task: str | None = None
        self._last_step: int | None = None
        self._last_rows: np.ndarray | None = None
        #: Per-chunk latency in ms, in order.
        self.latencies_ms: list[float] = []
        #: Every chunk returned, in order.
        self.history: list[CosmosDroidPrediction] = []
        #: The exact ``proprio_json`` most recently sent.
        self.last_proprio_json: str | None = None

    # ------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        await self.session.connect(
            TRACKS,
            subscribe=("action_prediction",),
            ready_timeout_s=self.ready_timeout_s,
        )

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "CosmosDroidClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    @property
    def last_step(self) -> int | None:
        """The ``step`` of the last chunk received, i.e. what the next echo
        will report. ``None`` before the first chunk, which needs no echo."""
        return self._last_step

    # --------------------------------------------------------------- predict

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        joints: Sequence[float] | np.ndarray,
        gripper: float,
        task: str,
        *,
        executed: np.ndarray | None = None,
    ) -> CosmosDroidPrediction:
        """Push one observation and return the chunk it produced.

        Args:
            frames: ``{track_name: (H, W, 3) uint8}`` for all three of
                :data:`TRACKS`. Keyed by name, never positional.
            joints: 7 measured joint positions in radians.
            gripper: Measured gripper position.
            task: The instruction. Re-sent only when it changes; a change needs
                nothing else, because the model is stateless.
            executed: Rows to echo, defaulting to the previous chunk in full,
                i.e. what a client that executed the whole chunk reports.
                Ignored on the first chunk, which has nothing to echo.
        """
        if len(task) > TASK_MAX_LEN:
            log.error(
                "task_description is %d chars, max %d; truncating",
                len(task), TASK_MAX_LEN,
            )
            task = task[:TASK_MAX_LEN]
        if task != self._task:
            await self.session.send(
                "set_task_description", {"task_description": task}
            )
            self._task = task

        # Frames and proprio both have to be in place before the echo: the
        # model predicts only when it holds a full frame set AND parseable
        # proprio, and it snapshots both at the tick it predicts on.
        self.session.set_frames(frames)
        proprio_json = encode_proprio(joints, gripper)
        await self.session.send("set_proprio_json", {"proprio_json": proprio_json})
        self.last_proprio_json = proprio_json

        await asyncio.sleep(self.settle_s)

        # A late answer to a previous, timed-out request must not be served as
        # this request's chunk.
        discarded = [
            int(d.get("step", -1)) for d in self.session.drain("action_prediction")
        ]
        if discarded:
            log.warning("discarded stale chunk(s) step=%s", discarded)

        t0 = time.perf_counter()
        if self._last_step is not None:
            rows = executed if executed is not None else self._last_rows
            await self.session.send(
                "set_executed_step_json",
                {"executed_step_json": encode_executed_step(self._last_step, rows)},
            )

        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            # The wait itself raises on expiry, so catch it there too -- a bare
            # asyncio TimeoutError escaping here loses the diagnosis below.
            try:
                if remaining <= 0:
                    raise TimeoutError
                data = await self.session.next_message(
                    "action_prediction", timeout_s=remaining
                )
            except TimeoutError:
                raise TimeoutError(
                    f"no chunk within {self.timeout_s:.0f}s. The model stays "
                    "silent if the echoed step did not strictly increase, if "
                    "proprio_json did not parse, or if any of the three "
                    "tracks has no frame. If none of those apply, the "
                    "deployment itself is not answering -- report it."
                ) from None
            actions = self._decode(data)
            if actions is None:
                discarded.append(int(data.get("step", -1)))
                continue
            latency_ms = (time.perf_counter() - t0) * 1e3
            self.latencies_ms.append(latency_ms)
            pred = CosmosDroidPrediction(
                actions=actions,
                step=int(data.get("step", -1)),
                latency_ms=latency_ms,
                discarded=discarded,
            )
            self._last_step, self._last_rows = pred.step, pred.actions
            self.history.append(pred)
            return pred

    # -------------------------------------------------------------- internals

    @staticmethod
    def _decode(data: dict) -> np.ndarray | None:
        """``(32, 8)`` float array, or None if the payload is unusable."""
        raw = data.get("action")
        if raw is None:
            log.warning("action_prediction without an 'action' field; discarding")
            return None
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, ACTION_SHAPE[1])
        if arr.ndim != 2 or arr.shape[1] != ACTION_SHAPE[1]:
            log.warning("action has shape %s; discarding", arr.shape)
            return None
        if not np.isfinite(arr).all():
            log.warning("action contains non-finite values; discarding")
            return None
        if arr.shape[0] != ACTION_SHAPE[0]:
            log.warning(
                "chunk horizon is %d, expected %d", arr.shape[0], ACTION_SHAPE[0]
            )
        return arr
