# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""XR-1 RoboCasa365 client: lock-step, one chunk per executed-step echo.

Xiaomi-Robotics-1 (XR-1) fine-tuned for the RoboCasa365 kitchen benchmark: a
Qwen3-VL-4B backbone with a 604M DiT action expert, on the single-arm
PandaOmron embodiment. Three camera views plus a short robot-state history in,
a ``(16, 60)`` chunk out of which the first 12 columns drive this embodiment.

    client = Xr1Robocasa365Client()
    await client.connect()
    pred = await client.predict(frames, state_history, task="close the blender lid")
    pred.live            # (16, 12): the columns this embodiment executes
    await client.close()

## Lock-step, and the first chunk needs an echo too

| Step | Wire |
|---|---|
| on change | ``set_task_description {task_description}`` (<=300 chars) |
| per chunk | ``set_state_history_json {state_history_json}`` (<=24000 chars) |
| per chunk | ``set_executed_step_json {executed_step_json}`` (<=8000 chars) |
| per chunk | model -> ``action_prediction {action: [16,60], step}`` |

Every prediction, **including the first**, needs an echo whose ``step`` is
strictly greater than the last one the model advanced on. Anything else (a
repeat, a lower value, malformed JSON) produces no reply and no error.

Requiring the echo on the first chunk as well is deliberate: an
asymmetric first prediction races the first echo, and losing that race leaves
the echo unconsumed, which reopens the gate on the same observation and makes
every later chunk answer the previous one. That shows up as a fixed one-step
lag that never corrects itself and looks exactly like a bad policy.

The model also refuses to predict until it has received ``OBS_HISTORY``
complete observations per echo it has consumed. A client publishing frames
continuously, as :class:`~reactor_robotics.track.RepeatingFrameTrack` does,
satisfies that without doing anything special.

## An observation is a set of three frames

The three cameras are separate named video tracks, and the model pairs them
frame for frame in arrival order into complete observations before it samples
its history. Publish them as a set. A camera that skips a step shifts its whole
history against the other two, which again reads as a bad policy rather than as
a client bug.

## The state history is four rows, not one

``state_history_json`` carries EXACTLY ``OBS_HISTORY`` rows of
``STATE_ROW_DIM`` floats, oldest first, sampled ``OBS_INTERVAL`` environment
steps apart to pair with the four video frames. While an episode is younger
than that window, repeat the earliest observation to fill the missing rows;
that is upstream's own clamp-to-earliest sampling and
:func:`encode_state_history` does it for you.

## Actions are a packed 60-dim layout

The checkpoint emits 60 columns because XR-1 shares one action layout across
embodiments. For RoboCasa365/PandaOmron the first 12 are live and the rest are
padding. Values arrive already decoded (denormalized) with the checkpoint's own
``robocasa365`` statistics, so they are the same numbers upstream's eval client
works with after its ``decode_action`` call.

## Timing

| | Value |
|---|---|
| chunk horizon | 16 steps |
| benchmark replan windows | 16 (whole chunk) and 8 |
| per-prediction compute, default build | ~171 ms |
| per-prediction compute, gated lossless lever | ~71 ms |

The faster figure needs ``XR1_COMPILE_DIT`` enabled on the deployment, which is
off by default. Both are reference figures for this deployment, not guarantees.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .session import ReactorSession

#: Track names, in the checkpoint's declaration order. This is also the order
#: the views enter the prompt template (Left camera / Right camera / Wrist
#: camera), so the names are not interchangeable.
TRACKS: tuple[str, str, str] = ("left_agentview", "right_agentview", "wrist_view")

#: (action horizon, packed layout width).
ACTION_SHAPE: tuple[int, int] = (16, 60)

#: Columns of :data:`ACTION_SHAPE` that drive the PandaOmron embodiment. The
#: remaining 48 are the packed-layout padding.
LIVE_DIMS = 12

#: Observations per prediction, and the environment-step stride between them.
#: These mirror the deployment's ``obs_history`` / ``obs_interval`` config.
OBS_HISTORY = 4
OBS_INTERVAL = 2

#: Width of one state row: [0:3] left EE xyz, [3:6] left EE axis-angle,
#: [6] left gripper, [7:10] right EE xyz, [10:13] right EE axis-angle,
#: [13] right gripper. Padded to the model's 60-dim layout server-side.
STATE_ROW_DIM = 14

#: The benchmark's native camera size.
FRAME_SIZE: tuple[int, int] = (256, 256)

#: How many chunk steps the benchmark executes before asking again. The parity
#: grid was run at both.
REPLAN_STEPS = 16

#: Reference figures for this deployment, not guarantees. The first is the
#: default build; the second needs the gated ``XR1_COMPILE_DIT`` lever.
EXPECTED_COMPUTE_MS = 171.0
EXPECTED_COMPUTE_COMPILED_MS = 71.0

#: ``InputField(max_length=...)`` on the three state fields.
TASK_MAX_LEN = 300
STATE_HISTORY_MAX_LEN = 24000
EXECUTED_STEP_MAX_LEN = 8000

log = logging.getLogger("reactor_robotics.xr1_robocasa365")


def sample_history(
    items: list, length: int = OBS_HISTORY, interval: int = OBS_INTERVAL
) -> list:
    """Upstream's newest-anchored sampling, clamped to the oldest item.

    Picks ``length`` items ``interval`` apart ending at the newest. While the
    buffer is younger than that window the earliest item repeats, which is what
    the model expects at the start of an episode.
    """
    if not items:
        raise ValueError("observation history cannot be empty")
    return [
        items[max(0, len(items) - 1 - (length - 1 - i) * interval)]
        for i in range(length)
    ]


def encode_state_history(rows: np.ndarray) -> str:
    """Build ``state_history_json`` from a state history.

    Accepts either exactly :data:`OBS_HISTORY` rows, or a longer buffer that is
    sampled down with :func:`sample_history`. A single row is broadcast to the
    full window, which is the correct thing at the first step of an episode.

    The server treats anything it cannot parse (empty, bad JSON, the wrong row
    count, a non-finite value) as "no state yet" and simply does not predict,
    so this validates locally instead.
    """
    arr = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"state history must be 2-D, got shape {arr.shape}")
    if not 1 <= arr.shape[1] <= 60:
        raise ValueError(
            f"state rows are {arr.shape[1]} wide; expected {STATE_ROW_DIM} "
            "(1 to 60 accepted, zero-padded server-side)"
        )
    if not np.isfinite(arr).all():
        raise ValueError("state history contains non-finite values")
    if arr.shape[0] == 1:
        window = [arr[0]] * OBS_HISTORY
    elif arr.shape[0] == OBS_HISTORY:
        window = list(arr)
    else:
        window = sample_history(list(arr))
    out = json.dumps(
        {"state_history": [row.tolist() for row in window]}, separators=(",", ":")
    )
    if len(out) > STATE_HISTORY_MAX_LEN:
        raise ValueError(
            f"state_history_json is {len(out)} chars, over the field's "
            f"{STATE_HISTORY_MAX_LEN} limit"
        )
    return out


def encode_executed_step(step: int) -> str:
    """Build ``executed_step_json``.

    The echo carries the step alone; the model does not want the executed
    rows back. The only requirement is that ``step`` strictly increases
    across the session.
    """
    out = json.dumps({"step": int(step)}, separators=(",", ":"))
    if len(out) > EXECUTED_STEP_MAX_LEN:  # pragma: no cover - an int cannot
        raise ValueError(f"executed_step_json is {len(out)} chars")
    return out


@dataclass
class Xr1Robocasa365Prediction:
    """One chunk."""

    actions: np.ndarray
    """``(16, 60)``: the packed layout, already decoded with the checkpoint's
    ``robocasa365`` statistics."""

    step: int
    """The model's monotonic prediction counter for this session, from 0."""

    latency_ms: float
    """Wall-clock from the executed-step echo to this chunk arriving. Excludes
    the frame settle."""

    discarded: list[int] = field(default_factory=list)
    """``step`` of any chunk dropped as stale before this one. At most one
    chunk is ever in flight, so this stays empty unless a previous request
    timed out and its answer arrived late."""

    @property
    def live(self) -> np.ndarray:
        """``(16, 12)``: the columns this embodiment executes."""
        return self.actions[:, :LIVE_DIMS]

    @property
    def padding(self) -> np.ndarray:
        """``(16, 48)``: the packed-layout padding, which should stay unused."""
        return self.actions[:, LIVE_DIMS:]


class Xr1Robocasa365Client:
    """Client for the served ``xr1-robocasa365`` policy."""

    def __init__(
        self,
        *,
        model: str = "xr1-robocasa365",
        session: ReactorSession | None = None,
        fps: int = 15,
        frame_size: tuple[int, int] = FRAME_SIZE,
        replan_steps: int = REPLAN_STEPS,
        settle_s: float | None = None,
        timeout_s: float = 90.0,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self.session = session or ReactorSession(
            model, fps=fps, frame_size=frame_size
        )
        # Frames go over video while commands go over the data channel, and the
        # model pairs whichever frames are newest with the request. A few track
        # periods let the fresh set land first. This reduces the risk but does
        # not remove it: no chunk is tagged with the frames it came from.
        self.settle_s = settle_s if settle_s is not None else max(3.0 / fps, 0.2)
        self.replan_steps = replan_steps
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self._task: str | None = None
        #: Environment steps reported so far. This is the NEXT echo value, and
        #: the only contract on it is that it strictly increases.
        self._executed = 0
        #: The last echo the model actually advanced on. Distinct from
        #: :attr:`executed_steps`, which is already the next value: anything
        #: at or below this is what the model refuses.
        self._last_echo: int | None = None
        #: Per-chunk latency in ms, in order.
        self.latencies_ms: list[float] = []
        #: Every chunk returned, in order.
        self.history: list[Xr1Robocasa365Prediction] = []
        #: The exact ``state_history_json`` most recently sent.
        self.last_state_history_json: str | None = None

    # ------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        await self.session.connect(
            TRACKS,
            subscribe=("action_prediction",),
            ready_timeout_s=self.ready_timeout_s,
        )

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "Xr1Robocasa365Client":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def reset(self) -> None:
        """Start a new episode.

        Clears the model's observation history and its flow-control counter.
        The benchmark client never needs this (the history window evicts by
        construction), but a real episode boundary is what it is for. The local
        echo counter restarts too, because the model's does.
        """
        await self.session.send("reset", {})
        self._executed = 0
        self._last_echo = None
        self.session.drain("action_prediction")

    @property
    def executed_steps(self) -> int:
        """The value the NEXT echo will report."""
        return self._executed

    @property
    def last_echo(self) -> int | None:
        """The last echo the model advanced on, or None before the first chunk.

        Use this, not :attr:`executed_steps`, when you want a value the model
        must refuse: ``executed_steps`` has already moved on to the next
        window, so ``executed_steps - 1`` is usually still strictly greater
        than what the model last consumed and gets answered.
        """
        return self._last_echo

    # --------------------------------------------------------------- predict

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        state_history: np.ndarray,
        task: str,
        *,
        executed_step: int | None = None,
    ) -> Xr1Robocasa365Prediction:
        """Push one observation and return the chunk it produced.

        Args:
            frames: ``{track_name: (H, W, 3) uint8}`` for all three of
                :data:`TRACKS`. Keyed by name, never positional: the server
                accepts a wrist frame published on the left track, so a
                positional mistake would only show up as degraded behaviour.
            state_history: :data:`OBS_HISTORY` rows of :data:`STATE_ROW_DIM`
                floats, oldest first, or a longer buffer to sample down, or a
                single row to broadcast.
            task: The instruction. Re-sent only when it changes.
            executed_step: The value to echo. Defaults to a running count of
                environment steps, advanced by ``replan_steps`` per call. Any
                strictly increasing sequence works.
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

        # Frames and state both have to be in place before the echo. The model
        # predicts only when it holds a complete observation set AND a state
        # history it can parse, and it snapshots both on the turn it predicts.
        self.session.set_frames(frames)
        state_json = encode_state_history(state_history)
        await self.session.send(
            "set_state_history_json", {"state_history_json": state_json}
        )
        self.last_state_history_json = state_json

        await asyncio.sleep(self.settle_s)

        # A late answer to a previous, timed-out request must not be served as
        # this request's chunk.
        discarded = [
            int(d.get("step", -1)) for d in self.session.drain("action_prediction")
        ]
        if discarded:
            log.warning("discarded stale chunk(s) step=%s", discarded)

        step = self._executed if executed_step is None else int(executed_step)
        t0 = time.perf_counter()
        await self.session.send(
            "set_executed_step_json",
            {"executed_step_json": encode_executed_step(step)},
        )

        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no chunk within {self.timeout_s:.0f}s. The model stays "
                    "silent if the echoed step did not strictly increase, if "
                    "state_history_json did not parse, or if any of the three "
                    "camera tracks has not contributed a complete observation."
                )
            data = await self.session.next_message(
                "action_prediction", timeout_s=remaining
            )
            actions = self._decode(data)
            if actions is None:
                discarded.append(int(data.get("step", -1)))
                continue
            latency_ms = (time.perf_counter() - t0) * 1e3
            self.latencies_ms.append(latency_ms)
            pred = Xr1Robocasa365Prediction(
                actions=actions,
                step=int(data.get("step", -1)),
                latency_ms=latency_ms,
                discarded=discarded,
            )
            self._last_echo = step
            self._executed = max(self._executed, step) + self.replan_steps
            self.history.append(pred)
            return pred

    # -------------------------------------------------------------- internals

    @staticmethod
    def _decode(data: dict) -> np.ndarray | None:
        """``(16, 60)`` float array, or None if the payload is unusable."""
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
