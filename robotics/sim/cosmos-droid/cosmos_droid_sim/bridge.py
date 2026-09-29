# ──────────────────────────────────────────────────────────────────────────
# Inference bridge (Reactor Python SDK transport).
#
#   sim -> model:  publish_track(wrist_view / exterior_view_1 / _2)   [video]
#                  send_command("set_task_description", {...})        [on change]
#                  send_command("set_proprio_json", {...})            [per request]
#                  send_command("set_executed_step_json", {...})      [per chunk]
#   model -> sim:  @on_message -> {action: [32, 8], step} -> resolves the
#                  pending GatewayRequest
#
# The executed-step echo is this model's flow-control gate, and the request
# loop leans on a property the other examples' models don't have: the model
# is STATELESS per request. There is no reset event and no KV cache (the
# prompt and proprio are sent with every prediction), so a new episode or a
# task change needs no wire ceremony at all. The gate simply will not emit
# chunk N+1 until step N is echoed, which maps 1:1 onto RoboLab's
# request/execute cadence:
#
#   request arrives -> frames+proprio out -> echo chunk N -> await N+1 -> reply
#
# reactor-sdk exchanges the API key with the configured Reactor API for
# a session JWT.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import threading

from reactor_sdk import Reactor, ReactorStatus

from .contract import (
    CMD_SET_EXECUTED_STEP,
    CMD_SET_PROPRIO,
    CMD_SET_TASK,
    FIELD_EXECUTED_STEP,
    FIELD_PROPRIO,
    FIELD_TASK,
    TRACKS,
    decode_chunk,
    encode_executed_step,
)
from .gateway import GatewayState
from .tracks import CameraTrack

log = logging.getLogger("cosmos_droid.bridge")

DEFAULT_MODEL = "cosmos-nano-policy-droid"
# cosmos-nano-policy-droid serves from the production API. The wrong host
# answers with an opaque 429, which reads as missing capacity, so check the
# URL before anything else.
DEFAULT_API_URL = "https://api.reactor.inc"

# How often the relay loop polls for a pending gateway request.
POLL_S = 0.01


class Bridge:
    def __init__(
        self,
        gateway: GatewayState,
        *,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        model_name: str = DEFAULT_MODEL,
        settle_s: float = 0.1,
    ):
        self._gateway = gateway
        # Settle window between pushing frames and opening the flow gate:
        # frames are sent over WebRTC video while commands are sent over the
        # data channel, and the engine pairs whatever frame is newest with
        # the request. The pause gives the fresh frame time to land first.
        # This reduces the risk but does not remove it; the other examples'
        # echo delay carries the same caveat.
        self._settle_s = settle_s
        self._chunks: asyncio.Queue = asyncio.Queue()
        self._pump_tasks: list[asyncio.Task] = []
        self._relay_task: asyncio.Task | None = None
        self._last_task: str | None = None
        self._last_step: int | None = None
        self._last_rows = None
        self.reactor = Reactor(model_name=model_name, api_key=api_key, api_url=api_url)
        self._register()

    # ── event wiring (register BEFORE connect, per the SDK contract) ─────────
    def _register(self) -> None:
        reactor = self.reactor

        @reactor.on_message
        def _message(message, *_scope):
            # SDK 1.6 also dispatches correlated message replies here. The
            # send_command return must not enqueue the same action again.
            self._accept_chunk(message)

        @reactor.on_error
        def _error(err):
            log.error(
                "[%s:%s] %s",
                getattr(err, "component", "?"),
                getattr(err, "code", "?"),
                getattr(err, "message", err),
            )

    def _accept_chunk(self, message) -> None:
        try:
            decoded = decode_chunk(message)
        except ValueError as exc:
            log.warning("dropping malformed chunk: %s", exc)
            return
        if decoded is not None:
            self._chunks.put_nowait(decoded)

    async def _publish_tracks(self) -> None:
        for name in TRACKS:
            track = await self.reactor.publish_track(name)
            source = CameraTrack(name, self._gateway.frame_reader(name))
            self._pump_tasks.append(asyncio.create_task(source.pump(track)))
            log.info("published track %s", name)
        self._relay_task = asyncio.create_task(self._relay())

    # ── the relay: one gateway request -> one chunk ──────────────────────────
    async def _relay(self) -> None:
        log.info("relay running")
        while True:
            req = await asyncio.to_thread(self._gateway.take_pending, POLL_S * 10)
            if req is None:
                continue
            try:
                await self._serve_one(req)
            except asyncio.CancelledError:
                req.error = "bridge closed before the request completed"
                req.done.set()
                raise
            except Exception as exc:  # surface in RoboLab, keep relaying
                log.exception("request failed")
                req.error = f"{type(exc).__name__}: {exc}"
                req.done.set()

    async def _serve_one(self, req) -> None:
        if self.reactor.get_status() != ReactorStatus.READY:
            raise RuntimeError("session is not READY")
        # Drain before any command can trigger a reply. In particular the
        # first proprio can produce an action before send_command returns.
        while not self._chunks.empty():
            stale = self._chunks.get_nowait()
            log.warning("discarding stale chunk step=%d", stale[0])

        # Give the camera pumps time to send this request's new observation
        # before task/proprio can trigger the first prediction.
        if self._settle_s > 0:
            await asyncio.sleep(self._settle_s)
        if req.task and req.task != self._last_task:
            await self.reactor.send_command(CMD_SET_TASK, {FIELD_TASK: req.task})
            self._last_task = req.task
            log.info("task: %r", req.task)
        await self.reactor.send_command(CMD_SET_PROPRIO, {FIELD_PROPRIO: req.proprio_json})

        # Open the flow gate. The first request of a session has nothing to
        # echo; the engine emits the first chunk once task+proprio are in.
        if self._last_step is not None:
            await self.reactor.send_command(
                CMD_SET_EXECUTED_STEP,
                {FIELD_EXECUTED_STEP: encode_executed_step(self._last_step, self._last_rows)},
            )

        step, action = await self._chunks.get()
        self._last_step, self._last_rows = step, action
        req.step, req.action = step, action
        req.done.set()

    # ── lifecycle ────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self.reactor.get_status() == ReactorStatus.READY

    async def __aenter__(self) -> "Bridge":
        try:
            # SDK 1.6 connect() resolves at READY. Publish here so callback
            # scheduling cannot race startup or hide publication failures.
            await self.reactor.connect()
            await self._publish_tracks()
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        tasks = [*self._pump_tasks]
        if self._relay_task is not None:
            tasks.append(self._relay_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._pump_tasks.clear()
        self._relay_task = None
        try:
            await self.reactor.disconnect()
        except Exception:
            log.warning("disconnect failed during close", exc_info=True)
        finally:
            self.reactor.close()

    async def __aexit__(self, *exc) -> None:
        await self.close()


class BridgeThread(threading.Thread):
    """Runs the Bridge on its own event loop in its own thread. The openpi
    WebsocketPolicyServer owns the main thread (its serve_forever blocks),
    so asyncio for the Reactor side live here. Nothing crosses the
    boundary except GatewayState, which is lock-guarded."""

    def __init__(self, **bridge_kwargs):
        super().__init__(name="cosmos-droid-bridge", daemon=True)
        self._kwargs = bridge_kwargs
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing: asyncio.Event | None = None
        self.bridge: Bridge | None = None
        self.ready = threading.Event()  # connected, or gave up trying
        self.failed: BaseException | None = None

    @property
    def is_ready(self) -> bool:
        return self.bridge is not None and self.bridge.is_ready

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        finally:
            loop.close()

    async def _serve(self) -> None:
        self._closing = asyncio.Event()
        try:
            async with Bridge(**self._kwargs) as bridge:
                self.bridge = bridge
                self.ready.set()
                await self._closing.wait()
        except BaseException as exc:  # noqa: BLE001 (reported to the main thread)
            self.failed = exc
            log.error("bridge failed: %s", exc)
        finally:
            self.ready.set()  # never leave the main thread waiting on a dead bridge

    def stop(self) -> None:
        if self._loop is not None and not self._loop.is_closed() and self._closing is not None:
            self._loop.call_soon_threadsafe(self._closing.set)
