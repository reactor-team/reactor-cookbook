# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""GR00T N1.7 client: free-running, paired by engine ordering.

GR00T N1.7 (NVIDIA Isaac-GR00T) is a vision-language-action model on the
DROID/Franka embodiment. Two camera views plus proprioception in, a 40-step
chunk out.

    client = GrootN17Client()
    await client.connect()
    pred = await client.predict(frames, state, task="pick up the cup")
    pred.joint_position    # (40, 7) ABSOLUTE joint targets, radians
    pred.eef_9d            # (40, 9) xyz + rot6d
    pred.gripper_position  # (40, 1)
    await client.close()

## Three ways it departs from the generic contract

| Generic contract | GR00T N1.7 |
|---|---|
| ``state_json`` is ``{proprio: [N floats], chunk_id}`` | a **dict of named state vectors**, no ``chunk_id`` at all |
| one ``actions`` array ``[K, A]`` | actions **split across three named fields** |
| ``step`` echoes your ``chunk_id`` | ``step`` is the model's own **inference counter** (0 at reset) |
| one reply per request | **free-running**: it predicts every engine tick |

There is nothing to echo, so there is no way to name the chunk you want.
:meth:`GrootN17Client.predict` pairs an observation with a chunk by engine
ordering instead; see "Pairing an observation with a chunk" below.

## Malformed state becomes zeros

The generic contract says a malformed request is dropped. This model instead
zeroes the affected key, warns once per session, and keeps predicting, so a
bad ``state_json`` leaves the policy acting on zeros with no error. This
client therefore validates locally: shapes, finiteness, nothing else.

## state_json layout

```json
{"eef_9d": [<9 floats>], "gripper_position": [<1>], "joint_position": [<7>]}
```

17 floats across three keys, the ``proprio N = 17`` in the contract table.
``eef_9d`` is xyz + a 6D rotation. Send all three every observation.

## Actions are ABSOLUTE joint targets

The deployed checkpoint predicts relatively and the **server** converts to
absolute using the ``state_json`` of that tick. So ``joint_position`` rows are
poses, not deltas, and their first row sits near the state you sent. Lose the
streamed state and the rows become relative deltas, with no error.

## Pairing an observation with a chunk

The model conditions on a **2-frame window strided** :data:`STRIDE` **= 15
engine ticks** (~1 s at 15 fps, the DROID training-time temporal window). The
window persists across chunks by design, so no synchronous client can get a
chunk whose whole window is a single fresh observation *unless* it holds that
observation for the full window. :meth:`predict` does exactly that, then
discards one chunk:

1. Send the state, then the frames.
2. Hold the observation for :attr:`GrootN17Client.window_s` = ``(STRIDE + 1 +
   margin) / fps``. Every tick appends the latest frame, so after this the
   whole 16-slot buffer is copies of this observation.
3. Drain every queued chunk, all of which predate the wait.
4. Discard the **next** chunk, and return the one after it.

Step 4 is the ordering argument, and it is exact because the engine is serial:
the shared serve loop drains conditioning, runs one inference, emits, then
loops. So chunk *B*'s conditioning was drained after chunk *A* was emitted.
If *A* arrived after the wait deadline, *B*'s conditioning is strictly later
than the deadline, hence entirely this observation.

The one assumption is that the wait exceeds a round trip, so that "arrived
after the deadline" implies "emitted after the deadline". None of these models
tags a chunk with the frame it came from, so ordering is all any client has;
the same caveat applies to X-WAM's frame settle.
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

#: Track names, in the checkpoint's declaration order. The server maps them to
#: its own video keys (``exterior_image_1_left``, ``wrist_image_left``).
VIEWS: tuple[str, str] = ("exterior_view", "wrist_view")

#: Action-chunk horizon.
ACTION_HORIZON = 40

#: Per-field action dims, summing to the contract table's ``action dim A = 17``.
ACTION_DIMS: dict[str, int] = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}

#: State keys and dims, summing to ``proprio N = 17``.
STATE_DIMS: dict[str, int] = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}

#: Engine ticks between the two frames the model samples. At 15 fps that is
#: ~1 s apart, which is the spacing in the DROID training data.
STRIDE = 15

#: The resolution the server resizes frames to. Publishing at this size skips a
#: resize; anything else works and is resized for you.
FRAME_HW: tuple[int, int] = (256, 256)

#: Publishing rate that reproduces the training-time window spacing.
CONTROL_HZ = 15

#: ``InputField(max_length=...)`` on the two state fields.
STATE_JSON_MAX_LEN = 2000
TASK_MAX_LEN = 300

#: Franka Research 3 joint limits (radians), from the FR3 datasheet. Used by
#: the script to confirm the predicted joint targets are commandable. A Panda
#: is wider on q1 but tighter on q2, q6 (upper) and q7, and marginally tighter
#: on q3, so a chunk inside these is not automatically inside a Panda's range.
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

#: DROID's reset pose (radians): a plausible, in-range starting state.
DROID_RESET_JOINTS = np.array(
    [0.0, -0.2 * np.pi, 0.0, -0.8 * np.pi, 0.0, 0.6 * np.pi, 0.0],
    dtype=np.float64,
)

log = logging.getLogger("reactor_robotics.groot_n17")


def encode_state(
    joint_position: Sequence[float] | np.ndarray,
    eef_9d: Sequence[float] | np.ndarray,
    gripper_position: float,
) -> str:
    """Build ``state_json``. Validates locally, because the server zeroes.

    A key the server cannot parse as a finite vector of the right length
    becomes zeros and warns once per session, so a malformed state shows up as
    a policy acting on defaults rather than as an error.
    """
    payload = {
        "eef_9d": [float(v) for v in np.asarray(eef_9d, dtype=np.float64).reshape(-1)],
        "gripper_position": [float(gripper_position)],
        "joint_position": [
            float(v) for v in np.asarray(joint_position, dtype=np.float64).reshape(-1)
        ],
    }
    for key, dim in STATE_DIMS.items():
        vec = np.asarray(payload[key], dtype=np.float64)
        if vec.shape != (dim,):
            raise ValueError(f"state key {key!r} must be {dim} floats, got {vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError(f"state key {key!r} is not finite: {payload[key]}")
    out = json.dumps(payload, separators=(",", ":"))
    if len(out) > STATE_JSON_MAX_LEN:
        raise ValueError(
            f"state_json is {len(out)} chars, over the field's "
            f"{STATE_JSON_MAX_LEN} limit"
        )
    return out


@dataclass
class GrootN17Prediction:
    """One action chunk, unpacked from its three named fields."""

    joint_position: np.ndarray
    """``(40, 7)`` **absolute** joint targets in radians: what a robot runs."""

    eef_9d: np.ndarray
    """``(40, 9)`` end-effector xyz + rot6d."""

    gripper_position: np.ndarray
    """``(40, 1)`` gripper command."""

    step: int
    """The model's inference counter for this session, from 0, reset by
    ``reset``. **Not** an echo; this model has no request id."""

    latency_ms: float
    """Wall-clock from the end of the frame-window wait to this chunk arriving.
    Excludes :attr:`GrootN17Client.window_s`."""

    window_s: float
    """The frame-window wait that preceded this chunk, in seconds."""

    discarded: list[int] = field(default_factory=list)
    """``step`` of every chunk :meth:`GrootN17Client.predict` discarded on
    the way here. Non-empty is normal: this model free-runs, so chunks arrive
    whether or not anyone asked."""

    @property
    def packed(self) -> np.ndarray:
        """``(40, 17)``: the three fields concatenated in
        :data:`ACTION_DIMS` order, which is the order the server packed them
        in. Convenient for shape and finiteness checks."""
        return np.concatenate(
            [self.eef_9d, self.gripper_position, self.joint_position], axis=1
        )


class GrootN17Client:
    """Free-running client for the served ``groot-n17`` policy."""

    def __init__(
        self,
        *,
        model: str = "groot-n17",
        session: ReactorSession | None = None,
        fps: int = CONTROL_HZ,
        settle_frames: int = 5,
        discard_chunks: int = 1,
        timeout_s: float = 120.0,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self.session = session or ReactorSession(model, fps=fps, frame_size=FRAME_HW)
        self.fps = fps
        #: How long one observation is held so the strided window fills with
        #: it. ``settle_frames`` is margin covering the round trip and the
        #: video jitter buffer.
        self.window_s = (STRIDE + 1 + settle_frames) / float(fps)
        #: Chunks discarded after the drain. 1 is the ordering argument in the
        #: module docstring; more only costs time.
        self.discard_chunks = discard_chunks
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self._task: str | None = None
        #: Per-chunk latency in ms, in order.
        self.latencies_ms: list[float] = []
        #: Every chunk returned, in order.
        self.history: list[GrootN17Prediction] = []
        #: The exact ``state_json`` most recently sent. Kept because the
        #: script's negative test needs the real bytes.
        self.last_state_json: str | None = None

    # ------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        await self.session.connect(
            VIEWS,
            subscribe=("action_prediction",),
            ready_timeout_s=self.ready_timeout_s,
        )

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "GrootN17Client":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def reset(self) -> None:
        """Clear the frame buffers and restart ``step`` at 0.

        Cheap: the model holds no cross-chunk state beyond the frame window,
        so this cannot corrupt anything in flight.
        """
        await self.session.send("reset", {})
        self.session.drain("action_prediction")
        log.info("reset sent (frame buffers cleared, step restarts at 0)")

    async def observe_period_s(self, samples: int = 3) -> float:
        """Measure the deployment's chunk period from chunk arrivals.

        The engine ticks at up to its configured rate but is really paced by
        inference, so the period is a property of the deployment rather than a
        constant. The script reports the number instead of asserting one.

        **Call this only once the model is already predicting**, i.e. after at
        least one :meth:`predict`. Before the task and the first frames are in,
        the model emits nothing and this would just time out.
        """
        stamps: list[float] = []
        for _ in range(samples + 1):
            try:
                await self.session.next_message(
                    "action_prediction", timeout_s=self.timeout_s
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "no chunks to time. observe_period_s() needs the model to be "
                    "predicting already: call it after a predict(), not before."
                ) from exc
            stamps.append(time.perf_counter())
        gaps = np.diff(stamps)
        return float(np.median(gaps))

    # --------------------------------------------------------------- predict

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        state_json: str,
        task: str,
    ) -> GrootN17Prediction:
        """Push one observation and return a chunk that saw only it.

        Args:
            frames: ``{track_name: (H, W, 3) uint8}`` for both of
                :data:`VIEWS`. Keyed by name, never positional.
            state_json: Built with :func:`encode_state`. Sent before the
                frames, because the engine drains conditioning at a tick
                boundary and the state must already be there when the frames
                arrive.
            task: The instruction. Re-sent only when it changes.
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

        await self.session.send("set_state_json", {"state_json": state_json})
        self.last_state_json = state_json

        self.session.set_frames(frames)

        # Fill the strided window with this observation: every tick appends the
        # latest frame, so after STRIDE+1 frame periods the whole buffer is it.
        await asyncio.sleep(self.window_s)
        t0 = time.perf_counter()

        # Everything queued was computed before the wait finished.
        discarded = [int(d.get("step", -1)) for d in self.session.drain("action_prediction")]

        # Then discard `discard_chunks` more. The first arrival may have been
        # mid-inference when the wait ended; the one after it necessarily
        # drained its conditioning later. See the module docstring.
        deadline = time.monotonic() + self.timeout_s
        to_skip = self.discard_chunks
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no chunk within {self.timeout_s:.0f}s (discarded "
                    f"{len(discarded)}). This model free-runs, so silence means "
                    "it is not predicting at all: check that both tracks carry "
                    "frames and that the task is set."
                )
            data = await self.session.next_message(
                "action_prediction", timeout_s=remaining
            )
            if to_skip > 0:
                to_skip -= 1
                discarded.append(int(data.get("step", -1)))
                continue
            fields = self._decode(data)
            if fields is None:
                discarded.append(int(data.get("step", -1)))
                continue
            latency_ms = (time.perf_counter() - t0) * 1e3
            self.latencies_ms.append(latency_ms)
            pred = GrootN17Prediction(
                joint_position=fields["joint_position"],
                eef_9d=fields["eef_9d"],
                gripper_position=fields["gripper_position"],
                step=int(data.get("step", -1)),
                latency_ms=latency_ms,
                window_s=self.window_s,
                discarded=discarded,
            )
            self.history.append(pred)
            return pred

    # -------------------------------------------------------------- internals

    @staticmethod
    def _decode(data: dict) -> dict[str, np.ndarray] | None:
        """The three named action fields as arrays, or None if unusable."""
        out: dict[str, np.ndarray] = {}
        for key, dim in ACTION_DIMS.items():
            raw = data.get(key)
            if raw is None:
                log.warning("action_prediction missing %r; discarding", key)
                return None
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, dim)
            if arr.ndim != 2 or arr.shape[1] != dim:
                log.warning("%s has shape %s, want (H, %d)", key, arr.shape, dim)
                return None
            out[key] = arr
        rows = {k: v.shape[0] for k, v in out.items()}
        if len(set(rows.values())) != 1:
            log.warning("ragged chunk: %s; discarding", rows)
            return None
        if next(iter(rows.values())) != ACTION_HORIZON:
            log.warning(
                "chunk horizon is %d, expected %d",
                next(iter(rows.values())), ACTION_HORIZON,
            )
        return out
