# ──────────────────────────────────────────────────────────────────────────
# Inference bridge (Reactor Python SDK transport).
#
#   sim -> model:  publish_track(head_view / left_wrist_view / right_wrist_view)
#                  send_command("set_task_description", {...})   [on change]
#                  send_command("set_state_json", {...})         [per request]
#   model -> sim:  @on_message -> {"type": "action_prediction",
#                  "data": {actions, proprios, step}}
#
# Three properties of this path are the difference between working and
# subtly wrong, and all three are invisible at runtime:
#
# 1. ORDER AT CONNECT. Register handlers before connect() (READY can arrive
#    before the first await after it returns), then wait for READY before
#    publish_track. Publishing early does nothing at all, no error and no
#    track, and the model then waits forever for frames.
# 2. KEEPALIVE. The runtime kills a client that goes quiet for 20 s and
#    reactor-sdk 0.8.0 has no ping of its own. A lock-step eval is quiet by
#    construction while the simulator steps physics, so the bridge pings on
#    its own 10 s loop for the life of the session.
# 3. RETRY MUST CHANGE A BYTE. See contract.encode_state_json.
#
# reactor-sdk mints the session JWT from the API key in-process; the key
# never leaves the machine.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .contract import (
    CMD_RESET,
    CMD_SET_STATE,
    CMD_SET_TASK,
    FIELD_STATE,
    FIELD_TASK,
    FRAME_HW,
    MESSAGE_ACTION_PREDICTION,
    VIEWS,
    SimRequest,
    decode_prediction,
    encode_state_json,
)
from .tracks import RepeatingFrameTrack

log = logging.getLogger("robotwin_sim.bridge")

DEFAULT_MODEL = "xwam"
#: PROD, where xwam is served. Overridable for a different deployment.
DEFAULT_API_URL = "https://api.reactor.inc"
#: The runtime's watchdog fires at 20 s of client silence; ping at half that.
PING_INTERVAL_S = 10.0


@dataclass
class BridgeDiagnostics:
    """Enough to tell a stalled rollout from a failing one afterwards."""

    requests: int = 0
    replies: int = 0
    retries: int = 0
    stale_replies: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lat = np.asarray(self.latencies_ms, dtype=np.float64)
        p50 = f"{np.median(lat):.0f} ms" if lat.size else "n/a"
        return (
            f"{self.replies}/{self.requests} answered, p50 {p50}, "
            f"{self.retries} retried, {len(self.stale_replies)} stale replies "
            "discarded"
        )


class Bridge:
    """One lock-step Reactor session serving the RoboTwin gateway."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        model_name: str = DEFAULT_MODEL,
        fps: int = 15,
        settle_s: float | None = None,
        timeout_s: float = 30.0,
        retries: int = 2,
        ready_timeout_s: float = 300.0,
        ping_interval_s: float = PING_INTERVAL_S,
    ) -> None:
        from reactor_sdk import Reactor

        self.model_name = model_name
        self.api_url = api_url
        self.fps = fps
        # A few track periods is enough for a swapped observation to clear the
        # encoder; never less than 0.2 s even at a high fps.
        self.settle_s = settle_s if settle_s is not None else max(3.0 / fps, 0.2)
        self.timeout_s = timeout_s
        self.retries = retries
        # A warm deployment reports READY in seconds; a cold one has to
        # schedule a B200 and stage weights first.
        self.ready_timeout_s = ready_timeout_s
        self.diag = BridgeDiagnostics()

        self._reactor = Reactor(model_name, api_key=api_key, api_url=api_url)
        self._tracks = {
            view: RepeatingFrameTrack(view, fps=fps, size=FRAME_HW)
            for view in VIEWS
        }
        self._replies: asyncio.Queue = asyncio.Queue()
        self._ready = asyncio.Event()
        self._dropped = asyncio.Event()
        self._keepalive_task: asyncio.Task | None = None
        self._connected = False
        self._task: str | None = None
        self._chunk_id = 0
        self._ping_interval_s = ping_interval_s

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Register handlers, connect, await READY, publish tracks, start ping.

        The order of those steps is the whole point of this method; see the
        module header.
        """
        from reactor_sdk import ReactorStatus

        @self._reactor.on_status
        def _on_status(status) -> None:  # pragma: no cover - network callback
            log.info("status: %s", getattr(status, "name", status))
            if status == ReactorStatus.READY:
                self._ready.set()
            elif status == ReactorStatus.DISCONNECTED:
                self._dropped.set()

        @self._reactor.on_message
        def _on_message(msg) -> None:  # pragma: no cover - network callback
            if isinstance(msg, dict) and msg.get("type") == MESSAGE_ACTION_PREDICTION:
                self._replies.put_nowait(msg.get("data") or {})

        @self._reactor.on_error
        def _on_error(err) -> None:  # pragma: no cover - network callback
            log.error("session error: %s", err)

        await self._reactor.connect()
        self._connected = True
        await self._wait_ready()
        for view in VIEWS:
            await self._reactor.publish_track(view, self._tracks[view])
        log.info(
            "connected to %s at %s; tracks published: %s",
            self.model_name,
            self.api_url,
            ", ".join(VIEWS),
        )
        self._keepalive_task = asyncio.create_task(self._keepalive())

    async def _wait_ready(self) -> None:
        """Wait for READY, but give up the moment the session is dropped.

        There is no FAILED status in the SDK, and a session that cannot be
        placed goes CONNECTING -> WAITING -> DISCONNECTED, so a plain wait for
        READY sits out the entire ready timeout on a session that is already
        dead. Race the two instead.
        """
        ready = asyncio.ensure_future(self._ready.wait())
        dropped = asyncio.ensure_future(self._dropped.wait())
        try:
            await asyncio.wait(
                {ready, dropped},
                timeout=self.ready_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (ready, dropped):
                task.cancel()
        if self._ready.is_set():
            return
        if self._dropped.is_set():
            raise ConnectionError(
                f"the {self.model_name} session was dropped before it reached "
                "READY. Either the deployment is not currently serving, or the "
                "cluster is at capacity: session creation answers a busy "
                "cluster with HTTP 429 rather than queueing. Retry after a "
                "short wait."
            )
        raise TimeoutError(
            f"the {self.model_name} session did not reach READY within "
            f"{self.ready_timeout_s:.0f}s"
        )

    async def _keepalive(self) -> None:
        from reactor_sdk.types import MessageScope

        while True:
            try:
                await self._reactor.send_command(
                    "ping", {}, scope=MessageScope.RUNTIME
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - transport hiccup
                log.warning("keepalive ping failed", exc_info=True)
            await asyncio.sleep(self._ping_interval_s)

    async def close(self) -> None:
        """Stop the keepalive and disconnect. Safe to call twice.

        Always call this: a live session holds a real GPU worker.
        """
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive_task = None
        if self._connected:
            try:
                await self._reactor.disconnect()
            except Exception:  # pragma: no cover - best-effort teardown
                log.warning("disconnect() failed during close", exc_info=True)
            self._connected = False
        log.info("session closed: %s", self.diag.summary())

    async def __aenter__(self) -> "Bridge":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def reset(self) -> None:
        """Clear the model's episode state and any reply still in flight."""
        await self._reactor.send_command(CMD_RESET, {})
        while not self._replies.empty():
            self._replies.get_nowait()
        self._task = None

    # ── the one operation: request in, chunk out ─────────────────────────────

    async def predict(self, request: SimRequest) -> tuple[np.ndarray, np.ndarray]:
        """Answer one relayed client request. Returns ``(actions, proprios)``."""
        if request.task and request.task != self._task:
            await self._reactor.send_command(
                CMD_SET_TASK, {FIELD_TASK: request.task}
            )
            self._task = request.task
            log.info("task: %r", request.task)

        for view, frame in request.frames.items():
            self._tracks[view].set_frame(frame)
        # Let the swapped observation clear the encoder before the request. The
        # model pairs the request with the next frames to ARRIVE, which must
        # carry the new content and not the tail of the encoder queue.
        await asyncio.sleep(self.settle_s)

        self._chunk_id += 1
        self.diag.requests += 1

        for attempt in range(self.retries + 1):
            if attempt:
                self.diag.retries += 1
            state_json = encode_state_json(request, self._chunk_id, retry=attempt)
            t0 = time.perf_counter()
            await self._reactor.send_command(
                CMD_SET_STATE, {FIELD_STATE: state_json}
            )
            try:
                while True:
                    data = await asyncio.wait_for(
                        self._replies.get(), timeout=self.timeout_s
                    )
                    step, actions, proprios = decode_prediction(data)
                    if step != self._chunk_id:
                        # A reply crossing an episode reset. Drop it.
                        log.warning(
                            "discarding stale reply step=%s (want %d)",
                            step,
                            self._chunk_id,
                        )
                        self.diag.stale_replies.append(step)
                        continue
                    latency_ms = (time.perf_counter() - t0) * 1e3
                    self.diag.latencies_ms.append(latency_ms)
                    self.diag.replies += 1
                    log.info(
                        "chunk %d (rollout %d step %d): %.0f ms%s",
                        self._chunk_id,
                        request.seed[1],
                        request.seed[2],
                        latency_ms,
                        f" after {attempt} retr{'y' if attempt == 1 else 'ies'}"
                        if attempt
                        else "",
                    )
                    return actions, proprios
            except asyncio.TimeoutError:
                log.warning(
                    "timeout waiting for chunk %d (attempt %d/%d)",
                    self._chunk_id,
                    attempt + 1,
                    self.retries + 1,
                )
        raise TimeoutError(
            f"no reply for chunk {self._chunk_id} after {self.retries + 1} "
            "attempts. Is the model READY and are all three tracks published?"
        )
