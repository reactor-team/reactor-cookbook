# ──────────────────────────────────────────────────────────────────────────
# The gateway: this example's analog of the other examples' RolloutState,
# and the reason there is no env.py here at all.
#
# RoboLab is not a library this example wraps: Isaac Sim owns its process, its
# python and its episode loop, and the seam it exposes for a remote policy is
# an openpi WebSocket port. So this example IS that port. RoboLab connects to
# it exactly as it would to a local policy server, and every query is relayed
# to a dreamzero model served on Reactor. The simulator stays completely
# unmodified, which is what makes a success rate measured here comparable
# to RoboLab's own published evaluations.
#
# Threading: none, unlike cosmos_droid_sim. This package includes the
# protocol server (policy_server.py), which allows an async infer(), so the
# WebSocket server and the WebRTC session share one event loop and there is
# no cross-thread hand-off to guard.
#
# Episode boundaries come from RoboLab two ways, and both have to reset the
# model: a `reset` endpoint call, and a changed `session_id` on an infer
# request. Missing either one would carry a causal cache across episodes.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging

from .bridge import Bridge
from .contract import (
    OBS_JOINTS,
    OBS_PROMPT,
    OBS_SESSION_ID,
    action_reply,
    extract_frames,
    extract_state,
)

log = logging.getLogger("dreamzero_sim.gateway")


class RoboLabPolicy:
    """The openpi policy interface, backed by a Reactor session."""

    def __init__(self, bridge: Bridge) -> None:
        self._bridge = bridge
        self._session_id: str | None = None
        self._warned_no_state = False

    @property
    def diag(self):
        return self._bridge.diag

    async def infer(self, obs: dict) -> dict:
        """One observation in, one action chunk out."""
        session_id = obs.get(OBS_SESSION_ID)
        if session_id is not None and session_id != self._session_id:
            if self._session_id is not None:
                await self._bridge.reset_episode(
                    f"session id changed to {session_id!r}"
                )
            else:
                log.info("session started: %r", session_id)
            self._session_id = session_id

        frames = extract_frames(obs)
        joints, gripper = extract_state(obs)
        if OBS_JOINTS not in obs and not self._warned_no_state:
            # Worth one loud line: with zero joints the model's joint outputs
            # are relative deltas rather than absolute targets, which changes
            # what the executed numbers mean without raising anything.
            log.warning(
                "request carried no %s; sending zeros, which makes the "
                "predicted joint values RELATIVE deltas rather than absolute "
                "targets",
                OBS_JOINTS,
            )
            self._warned_no_state = True

        prompt = str(obs.get(OBS_PROMPT, "") or "")
        actions = await self._bridge.predict(frames, joints, gripper, prompt)
        return action_reply(actions)

    async def reset(self, reset_info: dict) -> None:
        """RoboLab's reset endpoint: end the episode."""
        session_ids = reset_info.get("session_ids")
        await self._bridge.reset_episode(
            f"reset endpoint (session_ids={session_ids})"
        )
        self._session_id = None
