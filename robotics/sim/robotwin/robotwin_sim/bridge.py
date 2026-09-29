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
# 1. ORDER AT CONNECT. SDK 1.6 connect() resolves at READY; publish named
#    tracks after it returns, then push paced RGB observations into them.
# 2. KEEPALIVE. The SDK owns client keepalive, including while the simulator
#    is stepping physics and the gateway has no new requests.
# 3. RETRY MUST CHANGE A BYTE. See contract.encode_state_json.
#
# reactor-sdk exchanges the API key with the configured Reactor API for
# a session JWT.
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
        self._pump_tasks: list[asyncio.Task] = []
        self._connected = False
        self._task: str | None = None
        self._chunk_id = 0

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Register handlers, connect at READY, then publish and pump frames."""
        @self._reactor.on_status
        def _on_status(status) -> None:  # pragma: no cover - network callback
            log.info("status: %s", getattr(status, "name", status))

        @self._reactor.on_message
        def _on_message(msg) -> None:  # pragma: no cover - network callback
            self._accept_reply(msg)

        @self._reactor.on_error
        def _on_error(err) -> None:  # pragma: no cover - network callback
            log.error("session error: %s", err)

        try:
            await asyncio.wait_for(self._reactor.connect(), self.ready_timeout_s)
            self._connected = True
            for view in VIEWS:
                track = await self._reactor.publish_track(view)
                self._pump_tasks.append(
                    asyncio.create_task(self._tracks[view].pump(track))
                )
        except BaseException:
            await self.close()
            raise
        log.info(
            "connected to %s at %s; tracks published: %s",
            self.model_name,
            self.api_url,
            ", ".join(VIEWS),
        )

    def _accept_reply(self, msg) -> None:
        if isinstance(msg, dict) and msg.get("type") == MESSAGE_ACTION_PREDICTION:
            self._replies.put_nowait(msg.get("data") or {})

    async def close(self) -> None:
        """Stop frame pumps, end the session, and release the native handle."""
        for task in self._pump_tasks:
            task.cancel()
        await asyncio.gather(*self._pump_tasks, return_exceptions=True)
        self._pump_tasks.clear()
        try:
            # Also clean up a partially completed connect().
            await self._reactor.disconnect()
        except Exception:  # pragma: no cover - best-effort teardown
            log.warning("disconnect() failed during close", exc_info=True)
        finally:
            self._reactor.close()
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
            try:
                # SDK 1.6 also dispatches correlated replies to on_message;
                # ignore the returned envelope to avoid enqueueing it twice.
                # Bound ACK and action waits by the same attempt deadline.
                deadline = asyncio.get_running_loop().time() + self.timeout_s
                await asyncio.wait_for(
                    self._reactor.send_command(CMD_SET_STATE, {FIELD_STATE: state_json}),
                    timeout=self.timeout_s,
                )
                while True:
                    data = await asyncio.wait_for(
                        self._replies.get(),
                        timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
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
