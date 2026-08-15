# ──────────────────────────────────────────────────────────────────────────
# Inference bridge (Reactor Python SDK transport).
#
#   sim -> model:  publish_track(agentview / eye_in_hand)          [video]
#                  send_command("set_task_description", {...})     [once per episode]
#                  send_command("set_executed_action_json", {...}) [once per chunk]
#   model -> sim:  @on_message -> {type:"action_prediction", data:{action, step}}
#                  -> RolloutState.submit_chunk(data)
#
# reactor-sdk exchanges the API key for a session JWT through the API's
# /tokens endpoint over HTTPS. This bridge never prints or logs the key.
#
# TIMING CAVEAT (the thing most likely to bite): the echo goes over the data
# channel while the frames go over WebRTC video, and the engine pairs
# whatever frame is latest with the state it reads. Sending the echo the
# instant the chunk finishes can therefore pair it with pre-execution frames.
# --echo-delay inserts a settle window before the echo to make that pairing
# right. This reduces the risk but does not remove it: the protocol has no
# explicit barrier, so nothing tags an echo with the frames it belongs to and
# timing is the only thing pairing them.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import threading

from reactor_sdk import Reactor, ReactorStatus

from .contract import TASK_MAX_LEN, VIEWS
from .loop import RolloutState
from .tracks import CameraTrack

log = logging.getLogger("libero.bridge")

DEFAULT_MODEL = "reactor/lingbot-va"

# How often the echo pump checks for a completed chunk.
PUMP_HZ = 100.0


def _msg_field(msg, key):
    """ActionPrediction may arrive as a dict or a dataclass-ish object."""
    if isinstance(msg, dict):
        return msg.get(key)
    return getattr(msg, key, None)


class Bridge:
    def __init__(
        self,
        rollout: RolloutState,
        *,
        api_key: str,
        api_url: str,
        task: str,
        model_name: str = DEFAULT_MODEL,
        echo_delay: float = 0.1,
    ):
        self._rollout = rollout
        self._task = task[:TASK_MAX_LEN]
        if len(task) > TASK_MAX_LEN:
            log.warning("task truncated to the model's %d-char limit", TASK_MAX_LEN)
        self._echo_delay = echo_delay
        self._last_task: str | None = None
        self._pump_task: asyncio.Task | None = None
        self._kicked = False
        self.reactor = Reactor(model_name=model_name, api_key=api_key, api_url=api_url)
        self._register()

    # ── event wiring (register BEFORE connect, per the SDK contract) ─────────
    def _register(self) -> None:
        reactor = self.reactor

        @reactor.on_status(ReactorStatus.READY)
        async def _ready(_status):
            await self._on_ready()

        @reactor.on_message
        def _message(message):
            if _msg_field(message, "type") == "action_prediction":
                data = _msg_field(message, "data") or {}
                self._rollout.submit_chunk(data)

        @reactor.on_error
        def _error(err):
            log.error(
                "[%s:%s] %s",
                getattr(err, "component", "?"),
                getattr(err, "code", "?"),
                getattr(err, "message", err),
            )

        @reactor.on_status(ReactorStatus.DISCONNECTED)
        def _disconnected(_status):
            self._stop_pump()

    async def _on_ready(self) -> None:
        # VIEWS order is the contract (see contract.py), so publish in it.
        for name in VIEWS:
            track = CameraTrack(name, self._rollout.frame_reader(name))
            await self.reactor.publish_track(name, track)
            log.info("published track %s", name)
        await self._push_task(self._task)
        self._start_pump()

    async def _reattach(self) -> None:
        """Re-attach the engine so it starts an episode from the current task.

        The prompt is embedded at ATTACH time only (every subsequent tick
        conditions on it without re-encoding), and the session attaches the
        moment it connects, necessarily before our set_task_description
        lands. Without this re-attach, every tick after that first empty
        attach fails server-side and the client just sees silence.

        This is also what starts each later episode: attach is where the
        policy clears its KV cache, so the reset that re-encodes the prompt
        is the same reset that stops the new episode from being predicted
        off the old one's history.

        The payload must be empty. The model ignores a reset carrying any
        unknown field, silently and as a no-op: ``reset {"sampling_seed":
        0}`` left ``step`` climbing and the episode anchored to the old KV
        cache (verified against the live deployment, 2026-08-11).
        """
        await self.reactor.send_command("reset", {})
        log.info("re-attached engine with task set")

    # ── echo pump: one send per completed chunk ──────────────────────────────
    def _start_pump(self) -> None:
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self._pump())

    def _stop_pump(self) -> None:
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()

    async def _pump(self) -> None:
        period = 1.0 / PUMP_HZ
        try:
            while True:
                try:
                    if self.reactor.get_status() == ReactorStatus.READY:
                        # Start each episode, including the first: re-attach
                        # so the policy drops the previous episode's KV cache
                        # and re-encodes the prompt, then send an empty echo
                        # (which the model reads as "nothing executed yet")
                        # to draw the first chunk. Once per episode; a
                        # repeat mid-episode would look like a restart.
                        if self._rollout.is_episode_start():
                            if not self._kicked:
                                # Latched only once both sends land, so a
                                # failed kick is retried on the next poll.
                                await self._reattach()
                                await self._send_echo("")
                                self._kicked = True
                        else:
                            self._kicked = False  # armed for the next reset
                        echo = self._rollout.take_pending_echo()
                        if echo is not None:
                            if self._echo_delay > 0:
                                await asyncio.sleep(self._echo_delay)
                            await self._send_echo(echo)
                            self._rollout.diag.echoes_sent += 1
                except Exception:
                    # One failed send must not take the pump with it.
                    log.exception("echo pump iteration failed")
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass

    async def _send_echo(self, echo: str) -> None:
        try:
            await self.reactor.send_command(
                "set_executed_action_json", {"executed_action_json": echo}
            )
        except Exception as exc:  # transient; keep pumping
            log.warning("set_executed_action_json failed: %s", exc)

    async def _push_task(self, task: str) -> None:
        if task == self._last_task:
            return
        self._last_task = task
        await self.reactor.send_command("set_task_description", {"task_description": task})
        log.info("task: %r", task)

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self.reactor.get_status() == ReactorStatus.READY

    async def set_task(self, task: str) -> None:
        # Re-attach after the change: a new task is only honoured at attach
        # (see _reattach), so setting it alone would leave the old prompt in
        # place.
        if task[:TASK_MAX_LEN] != self._last_task:
            await self._push_task(task[:TASK_MAX_LEN])
            await self._reattach()

    async def reset(self) -> None:
        # Only asks the sim to reset. The pump sees the episode-start edge
        # that follows and does the re-attach and kick.
        self._rollout.request_reset()

    async def __aenter__(self) -> "Bridge":
        await self.reactor.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        self._stop_pump()
        try:
            await self.reactor.disconnect(recoverable=False)
        except Exception:
            pass


class BridgeThread(threading.Thread):
    """Runs the Bridge on its own event loop in its own thread.

    The main thread belongs to MuJoCo (loop.SimDriver), so asyncio and aiortc
    live here instead. Nothing crosses the boundary except RolloutState,
    which is lock-guarded, so no other synchronisation is needed.
    """

    def __init__(self, **bridge_kwargs):
        super().__init__(name="libero-bridge", daemon=True)
        self._kwargs = bridge_kwargs
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = asyncio.Event()
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
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._closing.set)
