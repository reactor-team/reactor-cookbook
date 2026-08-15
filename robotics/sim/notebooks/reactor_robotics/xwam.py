# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""X-WAM client: the lock-step robot-policy contract, one method.

X-WAM is the reference implementation of
``robot-policy-client-contract.md``: three named video tracks in, one
``(32, 14)`` action chunk out, one request outstanding at a time.

    client = XwamClient()
    await client.connect()
    pred = await client.predict(frames, proprio, task="pick up the bottle")
    pred.actions            # (32, 14) delta joint actions, bimanual
    await client.close()

This is the client that produced the published evaluation numbers. Two
details in it matter:

**Frame settle.** None of these models tags a chunk with the frame it came
from, so ordering is the only thing pairing an observation with its reply:
push frames, let them clear the encoder, *then* send the request. Send the
request too early and the model answers using the tail of the encoder queue
(the previous observation), and the reply looks plausible.

**Retry must change a byte.** The model answers once per *distinct*
``state_json`` value; an identical re-send is indistinguishable from the
continuous re-delivery of unchanged state and is deduplicated, so it produces
no reply at all. Retry therefore keeps the same ``chunk_id`` and bumps a
``retry`` counter. Because the noise seed is a pure function of the seed
fields, the retried answer is identical to the lost one when frames are fed
directly; over the video transport the re-encoded frames make it equal within
tolerance instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .session import ReactorSession

#: Track names, in the checkpoint's training-time camera order. The server
#: accepts a wrist frame published on the head track, so key frames by name.
VIEWS: tuple[str, str, str] = ("head_view", "left_wrist_view", "right_wrist_view")

#: (control steps, action dim). 14 = bimanual delta joint actions.
ACTION_SHAPE: tuple[int, int] = (32, 14)

#: Proprioception layout width.
PROPRIO_DIM = 16

log = logging.getLogger("reactor_robotics.xwam")


@dataclass
class XwamPrediction:
    """One answered request."""

    actions: np.ndarray
    """``(32, 14)`` delta joint actions: what the robot executes."""

    proprios: np.ndarray
    """``(9, 16)`` predicted future robot states. Diagnostic; nothing
    executes these."""

    step: int
    """The model's echo of the request's ``chunk_id``."""

    latency_ms: float
    """Wall-clock from request send to reply, excluding the frame settle."""

    retries: int = 0
    """How many re-sends this answer needed (0 on the happy path)."""


class XwamClient:
    """Lock-step client for the served ``xwam`` policy."""

    def __init__(
        self,
        *,
        model: str = "xwam",
        session: ReactorSession | None = None,
        fps: int = 15,
        settle_s: float | None = None,
        timeout_s: float = 30.0,
        retries: int = 2,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self.session = session or ReactorSession(
            model, fps=fps, frame_size=(240, 320)
        )
        # A few track periods is enough for the swapped observation to clear
        # the encoder; never less than 0.2 s even at a high fps.
        self.settle_s = settle_s if settle_s is not None else max(3.0 / fps, 0.2)
        self.timeout_s = timeout_s
        self.retries = retries
        # A warm deployment reports READY in seconds; a cold one has to
        # schedule a B200 and stage weights first.
        self.ready_timeout_s = ready_timeout_s
        self._task: str | None = None
        self._chunk_id = 0
        #: Per-request latency in ms, in order. The script reports these.
        self.latencies_ms: list[float] = []
        #: Replies discarded for a stale ``step`` echo.
        self.stale_replies: list[int] = []
        #: The exact ``state_json`` string of the most recent request. Kept
        #: because deduplication is defined on this *string*: re-sending it
        #: identically produces no reply, while changing one byte (the
        #: ``retry`` convention) re-answers with an identical chunk. The
        #: statelessness demo in the script needs the real bytes to show
        #: either behaviour.
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

    async def __aenter__(self) -> "XwamClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def reset(self) -> None:
        """Clear the model's episode state. Cheap: the model is stateless
        across chunks, so this cannot corrupt anything in flight."""
        await self.session.send("reset", {})
        self._task = None

    # --------------------------------------------------------------- predict

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        proprio: Sequence[float] | np.ndarray,
        task: str,
        *,
        cfg: float = 0.0,
        seed: tuple[int, int, int] | None = None,
    ) -> XwamPrediction:
        """Run one prediction.

        Args:
            frames: ``{track_name: (H, W, 3) uint8}`` for all three of
                :data:`VIEWS`. Keyed by name, never positional.
            proprio: 16 finite floats, the current robot state.
            task: The episode's language instruction. Re-sent only when it
                changes.
            cfg: Guidance scale.
            seed: Optional ``(env_rank, rollout_id, step_id)`` triple that
                pins the sampling noise exactly. Pass it **only** to replay a
                recorded evaluation request exactly; a robot client omits
                it and gets a seed derived from ``chunk_id``.
        """
        proprio_arr = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio_arr.shape != (PROPRIO_DIM,):
            raise ValueError(
                f"proprio must be {PROPRIO_DIM} floats, got {proprio_arr.shape}"
            )
        if not np.isfinite(proprio_arr).all():
            # The model drops a malformed request rather than zero-filling it,
            # because fabricated state would command a real arm. Fail here
            # instead of waiting out the timeout.
            raise ValueError("proprio contains non-finite values")

        if task != self._task:
            await self.session.send(
                "set_task_description", {"task_description": task}
            )
            self._task = task

        self.session.set_frames(frames)
        # Let the swapped observation clear the encoder before the request.
        await asyncio.sleep(self.settle_s)

        self._chunk_id += 1
        request: dict[str, object] = {
            "proprio": proprio_arr.tolist(),
            "chunk_id": self._chunk_id,
            "cfg": float(cfg),
        }
        if seed is not None:
            env_rank, rollout_id, step_id = (int(v) for v in seed)
            request.update(
                env_rank=env_rank, rollout_id=rollout_id, step_id=step_id
            )

        for attempt in range(self.retries + 1):
            if attempt:
                # Same chunk_id, one byte different: a distinct request string
                # (so it is answered) with identical seeds (so the answer is
                # identical). An identical resend gets deduplicated and
                # never replies.
                request["retry"] = attempt
            state_json = json.dumps(request)
            self.last_state_json = state_json
            t0 = time.perf_counter()
            await self.session.send("set_state_json", {"state_json": state_json})
            try:
                while True:
                    reply = await self.session.next_message(
                        "action_prediction", timeout_s=self.timeout_s
                    )
                    step = int(reply.get("step", -1))
                    if step != self._chunk_id:
                        # A stale reply crossing an episode reset is the
                        # classic harness bug. Drop it.
                        log.warning(
                            "discarding stale reply step=%s (want %d)",
                            step,
                            self._chunk_id,
                        )
                        self.stale_replies.append(step)
                        continue
                    latency_ms = (time.perf_counter() - t0) * 1e3
                    self.latencies_ms.append(latency_ms)
                    return XwamPrediction(
                        actions=np.asarray(reply["actions"], dtype=np.float64),
                        proprios=np.asarray(reply["proprios"], dtype=np.float64),
                        step=step,
                        latency_ms=latency_ms,
                        retries=attempt,
                    )
            except asyncio.TimeoutError:
                log.warning(
                    "timeout waiting for chunk %d (attempt %d/%d)",
                    self._chunk_id,
                    attempt + 1,
                    self.retries + 1,
                )
        raise TimeoutError(
            f"no reply for chunk {self._chunk_id} after {self.retries + 1} attempts"
        )
