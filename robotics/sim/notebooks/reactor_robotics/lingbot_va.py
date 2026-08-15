# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""LingBot-VA client: lock-step, driven by the executed-action echo.

LingBot-VA is a LIBERO-embodiment manipulation policy. Two camera views in,
one ``(16, 7)`` chunk of OSC_POSE deltas out.

    client = LingbotVaClient()
    await client.connect()
    pred = await client.predict(frames, task="put the bowl on the plate")
    pred.actions        # (16, 7): (dx, dy, dz, droll, dpitch, dyaw, gripper)
    pred.executable     # rows a client executes: 12 on chunk 0, 16 after
    await client.close()

The working reference for this protocol is in this repo:
[`../libero/`](../libero) drives a real LIBERO env against the same
model. This module is that bridge reduced to one ``predict()``.

## What advances the episode

Nothing else. The model emits one chunk, then stays silent until the client
reports what it executed:

| Step | Wire |
|---|---|
| once per episode | ``set_task_description {task_description}`` (≤300 chars) |
| once per episode | ``set_executed_action_json {executed_action_json: ""}`` |
| once per episode | ``reset {}``: re-attach; clears the KV cache, ``step`` → 0 |
| seed chunk | model → ``action_prediction {action: [16,7], step}``, unprompted |
| per chunk after | ``set_executed_action_json`` with the rows just executed |
| per chunk after | model → ``action_prediction {action: [16,7], step}`` |

Five properties of this deployment's wire matter, and breaking any of them
produces no error:

1. **There is no proprioception on this wire.** The state carries only the
   task and the echo. The observation is the two video tracks.
2. **The echo signals by changing value.** An identical repeat reads as
   "nothing new" and produces no chunk at all. :meth:`predict` keeps full
   float precision and counts repeats in :attr:`echo_duplicates`.
3. **``reset`` takes an empty payload.** With an unknown field
   (``reset {"sampling_seed": 0}``) nothing happens: ``step`` keeps climbing
   and the episode is not re-anchored. ``reset {}`` restarts ``step`` at 0
   and re-emits the pinned seed row.
4. **Clear the echo before ``reset``, not after.** ``reset`` rebuilds the KV
   cache while the echo is ordinary conditioning; the other order lets the
   previous episode's last echo land as this episode's first executed chunk.
5. **From the second echo on, it must be exactly 16 rows.** The server
   reshapes it to ``(4, 4, 7)`` and drops the exception, so a wrong length
   ends the episode without an error.

Also: **do not drain before the seed chunk.** After ``reset`` the model emits
that chunk unprompted, so a client that clears its queue first throws away the
only chunk it will ever get and then waits forever for an echo it cannot send.

## The seed chunk skips 4 rows

At the start of an episode the server pins the leading action latent to
normalized zero as a **conditioning slot** rather than predicting it, and the
authors' own client skips it: one latent frame, i.e.
:data:`ACTION_PER_FRAME` = 4 actions, so **12 of 16 execute**. Executing it
commands the quantile midpoint, which is real motion (``dx`` +0.12 four times
over). :attr:`LingbotVaPrediction.executable` applies the skip.

## Frame budget: hold each observation for the commit window

The server commits a window of video frames per chunk. Nominal size is
:data:`COMMIT_FRAMES` = 16 (``frame_chunk_size`` 4 × ``action_per_frame`` 4),
12 on the seed chunk, and the committed window ranges over
**14-19 frames** in practice because nothing ties the data channel's
echo to the video track. A replayed observation therefore has to be held on the
tracks long enough for that window to fill with it, which is what
:attr:`LingbotVaClient.window_s` waits out before echoing: 20 frame periods,
covering the jitter. Ordering is the only thing pairing an observation with its
chunk, because none of these models tags a chunk with the frame it came from.

## Action units

Rows arrive in **raw LIBERO action units**: the server already un-normalized
the model's ``[-1, 1]`` output through the training quantiles
(:data:`NORM_Q01` / :data:`NORM_Q99`). They are **deltas**, not absolute
poses. One raw unit is not a metre: LIBERO drives robosuite's ``OSC_POSE``
controller, which rescales ``[-1, 1]`` to its own ``output_max`` internally,
and that scale is not part of the served contract. Executing these on a real
arm needs a site calibration plus IK; see the guide's Physical deployment
section.
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

#: Track names, in the checkpoint's declaration order. The server maps them
#: **positionally** onto its own camera keys (``agentview_rgb``,
#: ``eye_in_hand_rgb``), and accepts a wrist view published on ``agentview``
#: without an error.
VIEWS: tuple[str, str] = ("agentview", "eye_in_hand")

#: (action horizon, DoF). 7 = 6 end-effector deltas + gripper.
ACTION_SHAPE: tuple[int, int] = (16, 7)

#: Channel order of a chunk row.
ACTION_CHANNELS: tuple[str, ...] = (
    "dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper",
)

#: Actions per latent frame, and the unit the seed skip is counted in.
ACTION_PER_FRAME = 4

#: Rows of the FIRST chunk of an episode that must not be executed.
SEED_SKIP_STEPS = ACTION_PER_FRAME

#: Video frames per view the server commits per chunk (12 on the seed chunk).
COMMIT_FRAMES = 16

#: The training set's native camera resolution.
CAM_SIZE = 128

#: LIBERO/robosuite's control rate. One chunk = 16 actions = 0.8 s of LIBERO
#: time, which is also 16 frames at this rate, so publishing at 20 fps makes
#: the committed window one chunk of LIBERO time.
CONTROL_HZ = 20

#: ``InputField(max_length=...)`` on the two state fields.
TASK_MAX_LEN = 300
ECHO_MAX_LEN = 8000

#: Per-channel training quantiles the server un-normalizes through. Used here
#: only as the range check that says a row is a delta rather than a pose.
NORM_Q01 = np.array(
    [-0.6589285731315613, -0.84375, -0.9375,
     -0.12107142806053162, -0.15964286029338837, -0.26571428775787354, -1.0],
    dtype=np.float64,
)
NORM_Q99 = np.array(
    [0.8999999761581421, 0.8544642925262451, 0.9375,
     0.17142857611179352, 0.1842857152223587, 0.34392857551574707, 1.0],
    dtype=np.float64,
)

log = logging.getLogger("reactor_robotics.lingbot_va")


def encode_executed(rows: np.ndarray) -> str:
    """Encode executed rows as ``executed_action_json``.

    Full float precision: the echo signals by *changing*, and rounding raises
    the chance that two chunks echo identically.
    """
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_SHAPE[1]:
        raise ValueError(
            f"executed rows have shape {arr.shape}, want (n, {ACTION_SHAPE[1]})"
        )
    payload = json.dumps(arr.tolist(), separators=(",", ":"))
    if len(payload) > ECHO_MAX_LEN:
        raise ValueError(
            f"executed_action_json is {len(payload)} chars, over the field's "
            f"{ECHO_MAX_LEN} limit ({arr.shape[0]} rows)"
        )
    return payload


@dataclass
class LingbotVaPrediction:
    """One chunk."""

    actions: np.ndarray
    """``(16, 7)`` raw-LIBERO-unit deltas, exactly as the model returned
    them."""

    executable: np.ndarray
    """The rows a client executes: ``actions`` with the seed skip applied, so
    ``(12, 7)`` on an episode's first chunk and ``(16, 7)`` after it."""

    step: int
    """The model's own prediction counter. **Not** an echo of anything the
    client sent; this model has no request id."""

    chunk_in_episode: int
    """0-based index of this chunk within the current episode, client-side."""

    latency_ms: float
    """Wall-clock to the chunk arriving.

    For a steady-state chunk this is measured from the echo: round trip plus
    inference. For the **seed chunk** it is measured from ``reset``, so the
    number also contains the time the server spent gathering its first 12
    frames. The two are not comparable; report them apart.
    """

    window_s: float
    """The frame-window hold that preceded this chunk, in seconds. 0 for the
    seed chunk, which needs no hold."""

    discarded: list[int] = field(default_factory=list)
    """``step`` of any chunk discarded before this one. Should stay empty: the
    model emits one chunk per echo, so a non-empty list means an extra chunk
    arrived and the protocol assumption is wrong somewhere."""


class LingbotVaClient:
    """Lock-step client for the served ``lingbot-va`` policy."""

    def __init__(
        self,
        *,
        model: str = "lingbot-va",
        session: ReactorSession | None = None,
        fps: int = CONTROL_HZ,
        settle_frames: int = 4,
        timeout_s: float = 120.0,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self.session = session or ReactorSession(
            model, fps=fps, frame_size=(CAM_SIZE, CAM_SIZE)
        )
        self.fps = fps
        #: How long one observation is held on the tracks before the echo, so
        #: the server's commit window fills with it. ``settle_frames`` is margin
        #: for the variable window the server commits: 14-19 vs the nominal 16.
        self.window_s = (COMMIT_FRAMES + settle_frames) / float(fps)
        #: Shorter wait used only at episode start, so the real frames replace
        #: the black placeholder the track opens with before ``reset``
        #: re-anchors the episode on them.
        self.episode_settle_s = max(4.0 / fps, 0.3)
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self._task: str | None = None
        self._episode_open = False
        self._chunk_in_episode = 0
        self._last_pred: LingbotVaPrediction | None = None
        self._last_echo = ""
        #: Per-chunk latency in ms, in order.
        self.latencies_ms: list[float] = []
        #: Echoes that were identical to their predecessor. Every one is a
        #: chunk the model read as "nothing new".
        self.echo_duplicates = 0
        #: Row counts echoed, in order. Expect ``[12, 16, 16, ...]``.
        self.echo_rows_sent: list[int] = []

    # ------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        await self.session.connect(
            VIEWS,
            subscribe=("action_prediction",),
            ready_timeout_s=self.ready_timeout_s,
        )

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "LingbotVaClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    @property
    def chunk_in_episode(self) -> int:
        """Chunks returned in the current episode. 0 before the first."""
        return self._chunk_in_episode

    @property
    def last_echo(self) -> str:
        """The exact ``executed_action_json`` string most recently sent.

        Public because deduplication is defined on this *string*: re-sending it
        identically stalls the episode, and the script's negative test
        needs the real bytes to show that.
        """
        return self._last_echo

    async def start_episode(self, task: str) -> None:
        """Begin an episode: set the task, clear the echo, then ``reset {}``.

        The order is the contract. ``reset`` re-attaches the session and
        rebuilds the KV cache; ``executed_action_json`` is ordinary
        conditioning sampled per tick. Clearing it first closes the window in
        which the previous episode's final echo is committed as this
        episode's first executed chunk.

        Push the frames **before** calling this, so the fresh episode anchors
        on a real observation instead of the black frame the track opens with.
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
        await self.session.send("set_executed_action_json", {"executed_action_json": ""})
        self._last_echo = ""
        # Empty payload: an unknown field turns `reset` into a no-op.
        await self.session.send("reset", {})
        # Drop anything from the previous episode. Safe here: the fresh
        # episode's seed chunk cannot have been emitted yet, because `reset`
        # was sent on this same ordered data channel a moment ago.
        self.session.drain("action_prediction")
        self._episode_open = True
        self._chunk_in_episode = 0
        self._last_pred = None
        log.info("episode started (task set, echo cleared, reset sent)")

    # --------------------------------------------------------------- predict

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        task: str,
        *,
        executed: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> LingbotVaPrediction:
        """Push one observation and return the chunk it produced.

        Args:
            frames: ``{track_name: (H, W, 3) uint8}`` for both of
                :data:`VIEWS`. Keyed by name, never positional.
            task: The episode's instruction. A change starts a new episode,
                because the prompt is embedded at attach time only.
            executed: Rows to echo, defaulting to the previous chunk's
                :attr:`LingbotVaPrediction.executable`, i.e. what a client
                that ran the whole chunk would report. Ignored on an episode's
                first chunk, which has nothing to echo.
        """
        self.session.set_frames(frames)

        if task != self._task or not self._episode_open:
            # Frames first, then a short settle, then start the episode: the
            # seed chunk should anchor on this observation, not on the black
            # placeholder frame the track opens with.
            await asyncio.sleep(self.episode_settle_s)
            await self.start_episode(task)

        if self._chunk_in_episode == 0:
            # The seed chunk needs no echo and no window hold: `reset` cleared
            # the server's frame buffers, so everything it gathers from here is
            # already this observation, and it emits as soon as it has enough.
            # Do NOT drain: the chunk may already be queued, and draining it
            # away leaves the model waiting for an echo that can never come,
            # which looks exactly like a hang.
            t0 = time.perf_counter()
        else:
            # Hold the observation for the server's whole commit window, so the
            # frames it consumes are all this observation rather than a mix
            # with the previous one.
            await asyncio.sleep(self.window_s)
            # The model emits nothing until echoed, so anything queued here
            # contradicts the protocol; log it.
            unclaimed = self.session.drain("action_prediction")
            if unclaimed:
                log.warning(
                    "%d unclaimed chunk(s) before this echo; the model emitted "
                    "without being asked", len(unclaimed),
                )
            rows = executed
            if rows is None:
                assert self._last_pred is not None
                rows = self._last_pred.executable
            t0 = time.perf_counter()
            await self._send_echo(np.asarray(rows, dtype=np.float64))

        discarded: list[int] = []
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no chunk within {self.timeout_s:.0f}s. The usual cause is "
                    "an echo the model read as 'nothing new' (identical to "
                    "the last) or the wrong row count (16 required after the "
                    "first echo); the server reports neither."
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
            seed_chunk = self._chunk_in_episode == 0
            skip = SEED_SKIP_STEPS if seed_chunk else 0
            pred = LingbotVaPrediction(
                actions=actions,
                executable=actions[skip:],
                step=int(data.get("step", -1)),
                chunk_in_episode=self._chunk_in_episode,
                latency_ms=latency_ms,
                window_s=0.0 if seed_chunk else self.window_s,
                discarded=discarded,
            )
            self._chunk_in_episode += 1
            self._last_pred = pred
            return pred

    # -------------------------------------------------------------- internals

    async def _send_echo(self, rows: np.ndarray) -> None:
        payload = encode_executed(rows)
        if payload == self._last_echo:
            # The value is the model's own output; suppressing the send would
            # stall the episode just as surely as sending it. Send and log.
            self.echo_duplicates += 1
            log.error(
                "executed-action echo is identical to the previous one; "
                "the model reads that as 'nothing new' and stops advancing"
            )
        await self.session.send(
            "set_executed_action_json", {"executed_action_json": payload}
        )
        self._last_echo = payload
        self.echo_rows_sent.append(int(rows.shape[0]))

    @staticmethod
    def _decode(data: dict) -> np.ndarray | None:
        """``(16, 7)`` float array, or None if the payload is unusable."""
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
        if arr.shape[0] != ACTION_SHAPE[0]:
            # Not fatal (the horizon is read off the config on both sides),
            # but the seed skip is expressed in frame groups of this horizon.
            log.warning(
                "chunk horizon is %d, expected %d", arr.shape[0], ACTION_SHAPE[0]
            )
        return arr
