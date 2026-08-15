# ──────────────────────────────────────────────────────────────────────────
# The gateway state: this example's analog of the other examples'
# RolloutState, and the reason there is no env.py here at all.
#
# RoboLab/Isaac is not a library you wrap: it owns its own process, its own
# python, and its own rollout loop, and its supported seam for a remote
# policy is an openpi WebSocket port. So instead of re-implementing the env
# loop, this example IS that port: RoboLab connects to the gateway exactly
# as it would to a local policy server, and the gateway relays each request
# to a Reactor-served model. The sim stays completely unmodified, which
# is also what makes results comparable to RoboLab's own eval numbers.
#
# Consumption semantics follow for free: RoboLab requests a chunk,
# executes all 32 steps open-loop, then requests again. That whole-chunk
# cadence is the measured optimum for this policy: replanning mid-chunk
# strictly hurts; see the README's pointer to the latency study.
#
# Threading: the openpi server calls infer() synchronously on its own
# thread; reactor-sdk lives on the bridge thread's asyncio loop. infer()
# hands one GatewayRequest across and blocks on its result, so there is one
# request in flight at a time, which matches both sides' contracts (RoboLab
# is lock-step around its request; the model is single-inference-in-flight).
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from .contract import (
    OBS_IMAGE,
    OBS_PROMPT,
    TRACKS,
    encode_proprio,
    split_composite,
)

log = logging.getLogger("cosmos_droid.gateway")


@dataclass
class GatewayDiagnostics:
    """Enough to tell a stalled rollout from a failing one afterwards."""

    requests: int = 0
    chunks_returned: int = 0
    timeouts: int = 0
    last_step: int = -1


@dataclass
class GatewayRequest:
    """One relayed inference request, resolved by the bridge."""

    frames: dict[str, np.ndarray]
    proprio_json: str
    task: str
    done = None  # threading.Event, set when resolved
    step: int = -1
    action: np.ndarray | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        self.done = threading.Event()


class GatewayState:
    """Thread-safe hand-off between the openpi server thread (produces
    requests) and the bridge thread (produces chunks)."""

    def __init__(self, *, chunk_timeout_s: float = 90.0):
        self._chunk_timeout_s = chunk_timeout_s
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray | None] = {n: None for n in TRACKS}
        self._frame_seq = 0
        self._pending: GatewayRequest | None = None
        self._pending_evt = threading.Event()
        self.diag = GatewayDiagnostics()

    # ── reads for the bridge / tracks ───────────────────────────────────────
    def frame_reader(self, name: str):
        """Latest frame for a view + the request sequence it arrived with.

        Frames change once per RoboLab REQUEST (every ~2.1 s), not per sim
        step: RoboLab only sends observations when it asks for a chunk. The
        tracks publish one frame per request and heartbeat in between
        (tracks.CameraTrack), so the engine always pairs the newest
        observation with the proprio of the same request.
        """

        def _get() -> tuple[np.ndarray | None, int]:
            with self._lock:
                return self._frames.get(name), self._frame_seq

        return _get

    def take_pending(self, timeout: float) -> GatewayRequest | None:
        """Bridge side: wait for the next relayed request."""
        if not self._pending_evt.wait(timeout):
            return None
        with self._lock:
            req, self._pending = self._pending, None
            self._pending_evt.clear()
        return req

    # ── the openpi server side (sync, its own thread) ───────────────────────
    def infer(self, obs: dict) -> dict:
        """openpi Policy interface: one observation in, one chunk out.

        Called by openpi's WebsocketPolicyServer for every RoboLab request.
        Blocks until the bridge resolves the relayed request or the timeout
        lapses. A lapse raises, which surfaces in RoboLab as a failed
        episode rather than a silent stall.
        """
        if OBS_IMAGE not in obs:
            raise ValueError(f"observation missing {OBS_IMAGE!r}; is this RoboLab's Cosmos3Client?")
        req = GatewayRequest(
            frames=split_composite(obs[OBS_IMAGE]),
            proprio_json=encode_proprio(obs),
            task=str(obs.get(OBS_PROMPT, "")),
        )
        with self._lock:
            self._frames.update(req.frames)
            self._frame_seq += 1
            self._pending = req  # newest wins; RoboLab never overlaps requests
            self.diag.requests += 1
        self._pending_evt.set()

        if not req.done.wait(self._chunk_timeout_s):
            self.diag.timeouts += 1
            raise TimeoutError(
                f"no chunk within {self._chunk_timeout_s:.0f}s: model not READY, "
                "or the flow gate never opened (check the bridge log)"
            )
        if req.error is not None:
            raise RuntimeError(req.error)
        assert req.action is not None
        with self._lock:
            self.diag.chunks_returned += 1
            self.diag.last_step = req.step
        return {"action": req.action}
