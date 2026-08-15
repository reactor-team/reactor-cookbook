# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""DreamZero client: a synchronous predict() over a free-running stream.

DreamZero does **not** wait to be asked. Once a prompt is set and every
camera has a frame, it infers a chunk whenever the cameras give it fresh
frames and broadcasts it. A robot just executes the newest chunk. A script
that pushes one observation and waits for "its" chunk has more to do, because
the obvious implementation is wrong:

    push frames; chunk = await next_chunk()      # WRONG

The chunk that arrives next was **already in flight** when you pushed,
computed from the *previous* observation. It has the right shape, finite
values and a plausible trajectory, so nothing about it looks wrong. The fix:

    Record the largest obs_seq seen so far, push the new frames, then ignore
    every arriving chunk until one carries a strictly larger obs_seq; that one
    saw the new frames.

This is sound because the model waits for *every* camera to deliver a new
frame before inferring, so the first chunk whose ``obs_seq`` advances past the
pre-push mark necessarily consumed the pushed frame on all three cameras.
:meth:`DreamZeroClient.predict` implements exactly that.

**Camera naming.** ``exterior_1`` is the *real* left exterior view.
RoboLab calls it ``exterior_image_0_left``; the checkpoint numbers its video
keys from 1 while RoboLab numbers cameras from 0. ``exterior_2`` is RoboLab's
``exterior_image_1_left``, which under the default ``--cam2-source black`` is
an all-black frame matching the checkpoint's training-time camera dropout.
Getting these backwards feeds the model a black primary view and does not
error. ``../dreamzero/`` implements the same mapping.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .session import ReactorSession

#: Track names, in the order the pipeline declares them. ``exterior_1`` is the
#: real left exterior camera; see the module docstring.
TRACKS: tuple[str, str, str] = ("exterior_1", "exterior_2", "wrist")

#: (action horizon, dim). 7 joint targets + 1 gripper.
ACTION_SHAPE: tuple[int, int] = (24, 8)

#: The DROID checkpoint's eval transform geometry. The model resizes anything
#: else itself; this is only the natural frame size.
FRAME_HW: tuple[int, int] = (180, 320)

#: Franka Panda joint limits (radians), used by the script's invariant
#: checks to confirm the predicted joint targets are physically commandable.
FRANKA_JOINT_LIMITS = np.array(
    [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ],
    dtype=np.float64,
)

log = logging.getLogger("reactor_robotics.dreamzero")


@dataclass
class DreamZeroPrediction:
    """One action chunk that provably saw the observation you pushed."""

    actions: np.ndarray
    """``(24, 8)``: 24 steps of 7 absolute joint targets + 1 gripper."""

    chunk_index: int
    """The model's own chunk counter for this episode."""

    inference_seconds: float
    """Model-reported inference time: the replan cost, excluding transport."""

    obs_seq: int
    """Highest camera-frame count present in this chunk's snapshot."""

    latency_ms: float
    """Wall-clock from frame push to this chunk arriving, client-side."""

    discarded: list[int] = field(default_factory=list)
    """``obs_seq`` of the in-flight chunks ignored on the way here because
    they were computed from old frames. Non-empty is normal."""


class DreamZeroClient:
    """Synchronous wrapper over DreamZero's free-running chunk stream."""

    def __init__(
        self,
        *,
        model: str = "dreamzero",
        session: ReactorSession | None = None,
        fps: int = 15,
        settle_s: float | None = None,
        timeout_s: float = 120.0,
        ready_timeout_s: float = 900.0,
    ) -> None:
        self.session = session or ReactorSession(
            model, fps=fps, frame_size=FRAME_HW
        )
        # DreamZero is 14B across TWO B200s. A cold session has to schedule
        # both GPUs, stage ~60 GB of weights and warm torch.compile before it
        # reports READY, which can take many minutes, far longer than the
        # SDK-ish 120 s that suffices for a warm single-GPU model. Waiting is
        # correct here; timing out early just loses the pod you started.
        self.ready_timeout_s = ready_timeout_s
        # Only used at episode start, where there is no pre-push obs_seq mark
        # to compare against; see predict().
        self.settle_s = settle_s if settle_s is not None else max(4.0 / fps, 0.3)
        self.timeout_s = timeout_s
        self._prompt: str | None = None
        #: Highest obs_seq seen on any chunk this episode.
        self._obs_seq_high = -1
        #: Every chunk returned, in order. The script checks monotonicity.
        self.history: list[DreamZeroPrediction] = []

    # ------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        await self.session.connect(
            TRACKS,
            subscribe=(
                "action_chunk",
                "episode_started",
                "prompt_accepted",
                "episode_reset",
                "command_error",
            ),
            ready_timeout_s=self.ready_timeout_s,
        )

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "DreamZeroClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    @property
    def obs_seq_high(self) -> int:
        """Highest ``obs_seq`` seen this episode; ``-1`` before any chunk.

        Exposed because it is the client's whole synchronisation state: after
        :meth:`reset` it must go back to ``-1``, or :meth:`predict` would wait
        forever for an ``obs_seq`` to climb past a mark from an episode that
        no longer exists.
        """
        return self._obs_seq_high

    async def reset(self) -> None:
        """End the episode: clears the prompt and the causal cache.

        ``obs_seq`` restarts at 0 for the next episode, so the high-water mark
        has to reset with it; that is the reason this is a method and not
        just a ``send("reset")``.
        """
        await self.session.send("reset", {})
        try:
            await self.session.next_message("episode_reset", timeout_s=30.0)
        except asyncio.TimeoutError:
            log.warning("no episode_reset ack; resetting client state anyway")
        self._prompt = None
        self._obs_seq_high = -1
        self.session.drain("action_chunk")

    # --------------------------------------------------------------- predict

    def _absorb(self, chunks: list[dict]) -> None:
        for data in chunks:
            self._obs_seq_high = max(
                self._obs_seq_high, int(data.get("obs_seq", -1))
            )

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        joints: Sequence[float] | np.ndarray,
        gripper: float,
        task: str,
    ) -> DreamZeroPrediction:
        """Push one observation and return the chunk that saw it.

        Args:
            frames: ``{track_name: (H, W, 3) uint8}`` for all three of
                :data:`TRACKS`. Keyed by name; ``exterior_1`` is the real
                left view, see the module docstring.
            joints: 7 measured joint positions in radians. Streaming the real
                state is what makes the predicted joint targets **absolute**;
                with zeros the model emits relative deltas instead, without
                an error.
            gripper: 0 = open, 1 = closed.
            task: Task instruction. Re-sent only when it changes (a change
                mid-episode re-anchors the causal cache on the latest
                observation).
        """
        joints_arr = np.asarray(joints, dtype=np.float64).reshape(-1)
        if joints_arr.shape != (7,):
            raise ValueError(f"joints must be 7 floats, got {joints_arr.shape}")
        if not np.isfinite(joints_arr).all():
            raise ValueError("joints contains non-finite values")
        if not 0.0 <= float(gripper) <= 1.0:
            raise ValueError(f"gripper must be in [0, 1], got {gripper}")

        # State first: the model snapshots the latest value together with the
        # frames it consumes, so it must be in place before the frames land.
        await self.session.send(
            "set_joint_position", {"joint_position": joints_arr.tolist()}
        )
        await self.session.send(
            "set_gripper_position", {"gripper_position": float(gripper)}
        )

        # Pre-push high-water mark: everything already emitted was computed
        # without the observation we are about to push.
        self._absorb(self.session.drain("action_chunk"))
        pre_push_high = self._obs_seq_high

        self.session.set_frames(frames)
        t0 = time.perf_counter()

        starting_episode = task != self._prompt
        if starting_episode:
            # obs_seq restarts at 0 with each episode, so across an episode
            # boundary there is no earlier mark to compare against. Give
            # the real frames time to land before the prompt starts the
            # episode, so chunk 0's warmup anchors on them rather than on the
            # black placeholder frames the track opens with.
            await asyncio.sleep(self.settle_s)
            await self.session.send("set_prompt", {"prompt": task})
            self._prompt = task
            pre_push_high = -1

        discarded: list[int] = []
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no chunk with obs_seq > {pre_push_high} within "
                    f"{self.timeout_s:.0f}s (discarded {len(discarded)} "
                    "in-flight chunks)"
                )
            try:
                data = await self.session.next_message(
                    "action_chunk", timeout_s=remaining
                )
            except asyncio.TimeoutError:
                continue
            obs_seq = int(data.get("obs_seq", -1))
            self._obs_seq_high = max(self._obs_seq_high, obs_seq)
            if obs_seq <= pre_push_high:
                # In flight when we pushed => computed from the PREVIOUS
                # observation. Plausible-looking and wrong.
                log.info(
                    "discarding in-flight chunk obs_seq=%d (need > %d)",
                    obs_seq,
                    pre_push_high,
                )
                discarded.append(obs_seq)
                continue
            pred = DreamZeroPrediction(
                actions=np.asarray(data["actions"], dtype=np.float64),
                chunk_index=int(data["chunk_index"]),
                inference_seconds=float(data["inference_seconds"]),
                obs_seq=obs_seq,
                latency_ms=(time.perf_counter() - t0) * 1e3,
                discarded=discarded,
            )
            self.history.append(pred)
            return pred
