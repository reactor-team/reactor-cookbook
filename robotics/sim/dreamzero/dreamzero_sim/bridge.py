# ──────────────────────────────────────────────────────────────────────────
# Inference bridge (Reactor Python SDK transport).
#
#   sim -> model:  publish_track(exterior_1 / exterior_2 / wrist)   [video]
#                  send_command("set_joint_position", {...})        [per request]
#                  send_command("set_gripper_position", {...})      [per request]
#                  send_command("set_prompt", {...})                [on change]
#   model -> sim:  @on_message -> action_chunk {actions, obs_seq, ...}
#
# Four things here are the difference between working and subtly wrong.
#
# 1. THE obs_seq GATE. The model is free-running: it does not answer
#    requests, it broadcasts chunks. The chunk that arrives right after a
#    push was already in flight and was computed from the PREVIOUS
#    observation (right shape, finite, plausible). So: note the highest
#    obs_seq seen, push, then discard arriving chunks until one carries a
#    strictly larger obs_seq. Because the model waits for every camera to go
#    fresh before inferring, that chunk necessarily saw the pushed frames on
#    all three cameras.
#
# 2. CONNECT LAZILY. The session comes up on the first request, not at
#    startup. The tracks are queue-fed (see tracks.py), so between requests no
#    RTP flows at all, and a peer connection that is brought up and then left
#    idle for tens of seconds has been observed to leave a serving runtime
#    mapping inbound video to the wrong track names, after which the model
#    silently never satisfies its every-camera-fresh gate. RoboLab takes
#    minutes to boot its simulator, so connecting at startup means exactly
#    that idle gap. Connecting on the first request keeps it under a second.
#
# 3. PRIME THE SENDERS ONE AT A TIME. For the same reason, the first
#    observation's frames go out one camera at a time with a gap between them,
#    so the order the three video streams first appear on the wire is
#    deterministic rather than a race. It costs a few seconds once per
#    session, and the priming frames are the first real observation's own
#    frames; nothing synthetic enters the model.
#
# 4. THE STANDARD SDK ORDER. Handlers before connect(), READY before
#    publish_track (publishing early does nothing at all: no error, no
#    track), and a client-side keepalive, because the runtime drops a client
#    that is quiet for 20 s and reactor-sdk 0.8.0 has no ping of its own. An
#    evaluation is quiet by construction while the simulator executes a chunk.
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
    CMD_SET_GRIPPER,
    CMD_SET_JOINTS,
    CMD_SET_PROMPT,
    FIELD_GRIPPER,
    FIELD_JOINTS,
    FIELD_PROMPT,
    MESSAGE_ACTION_CHUNK,
    MESSAGE_COMMAND_ERROR,
    MESSAGE_EPISODE_RESET,
    MESSAGE_EPISODE_STARTED,
    MESSAGE_PROMPT_ACCEPTED,
    TRACKS,
    decode_chunk,
)
from .tracks import QueueVideoTrack

log = logging.getLogger("dreamzero_sim.bridge")

DEFAULT_MODEL = "dreamzero"
#: PROD, where dreamzero is served. Overridable for a different deployment.
DEFAULT_API_URL = "https://api.reactor.inc"
#: The runtime's watchdog fires at 20 s of client silence; ping at half that.
PING_INTERVAL_S = 10.0
#: A cold session for this model has to schedule TWO GPUs, stage ~60 GB of
#: weights and warm compilation before it reports READY. Waiting is correct;
#: timing out early just loses the workers you started.
DEFAULT_READY_TIMEOUT_S = 900.0


@dataclass
class BridgeDiagnostics:
    """Enough to tell a stalled evaluation from a failing one afterwards."""

    requests: int = 0
    chunks_returned: int = 0
    stale_discarded: int = 0
    stale_returned: int = 0
    resets: int = 0
    inference_seconds: list[float] = field(default_factory=list)

    def summary(self) -> str:
        secs = np.asarray(self.inference_seconds, dtype=np.float64)
        p50 = f"{np.median(secs) * 1e3:.0f} ms" if secs.size else "n/a"
        return (
            f"{self.chunks_returned}/{self.requests} answered, model p50 {p50}, "
            f"{self.stale_discarded} in-flight chunks discarded by the gate, "
            f"{self.stale_returned} stale chunks returned on timeout, "
            f"{self.resets} episode resets"
        )


class Bridge:
    """One Reactor session, reconciling free-running chunks with a
    synchronous client."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        model_name: str = DEFAULT_MODEL,
        chunk_timeout_s: float = 300.0,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        prime_stagger_s: float = 2.0,
        ping_interval_s: float = PING_INTERVAL_S,
    ) -> None:
        from reactor_sdk import Reactor

        self.model_name = model_name
        self.api_url = api_url
        self.chunk_timeout_s = chunk_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.prime_stagger_s = prime_stagger_s
        self.diag = BridgeDiagnostics()

        self._reactor = Reactor(model_name, api_key=api_key, api_url=api_url)
        self._tracks = {name: QueueVideoTrack(name) for name in TRACKS}
        self._chunks: asyncio.Queue = asyncio.Queue()
        self._ready = asyncio.Event()
        self._dropped = asyncio.Event()
        self._episode_started = asyncio.Event()
        self._keepalive_task: asyncio.Task | None = None
        self._ping_interval_s = ping_interval_s
        self._connected = False
        self._connecting: asyncio.Lock = asyncio.Lock()
        self._primed = False
        self._prompt: str | None = None
        #: Highest obs_seq seen this episode; -1 before any chunk.
        self._obs_seq_high = -1

    # ── lifecycle ───────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def ensure_connected(self) -> None:
        """Connect on first use. See point 2 in the module header."""
        async with self._connecting:
            if self._connected:
                return
            await self._connect()

    async def _connect(self) -> None:
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
            if not isinstance(msg, dict):
                return
            self._handle_message(str(msg.get("type")), msg.get("data") or {})

        @self._reactor.on_error
        def _on_error(err) -> None:  # pragma: no cover - network callback
            log.error("session error: %s", err)

        log.info("connecting to %s at %s", self.model_name, self.api_url)
        await self._reactor.connect()
        self._connected = True
        await self._wait_ready()
        for name in TRACKS:
            await self._reactor.publish_track(name, self._tracks[name])
        log.info("tracks published: %s", ", ".join(TRACKS))
        self._keepalive_task = asyncio.create_task(self._keepalive())

    async def _wait_ready(self) -> None:
        """Wait for READY, but give up the moment the session is dropped.

        There is no FAILED status in the SDK. A session that cannot be placed
        goes CONNECTING -> WAITING -> DISCONNECTED, so a plain wait for READY
        sits out the entire ready timeout on a session that is already dead.
        That matters more here than anywhere else in this repo, because this
        model's legitimate cold start is minutes long, so the timeout is
        generous. Race the two instead.
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
                "cluster could not place a two-GPU session: session creation "
                "answers a busy cluster with HTTP 429 rather than queueing. "
                "Retry after a short wait."
            )
        raise TimeoutError(
            f"the {self.model_name} session did not reach READY within "
            f"{self.ready_timeout_s:.0f}s"
        )

    def _handle_message(self, msg_type: str, data: dict) -> None:
        if msg_type == MESSAGE_ACTION_CHUNK:
            self._chunks.put_nowait(data)
        elif msg_type == MESSAGE_EPISODE_STARTED:
            self._episode_started.set()
            log.info(
                "episode started (prompt=%r frames_per_chunk=%s action_horizon=%s)",
                str(data.get("prompt", ""))[:60],
                data.get("frames_per_chunk"),
                data.get("action_horizon"),
            )
        elif msg_type == MESSAGE_EPISODE_RESET:
            log.info("model confirmed episode reset")
        elif msg_type == MESSAGE_PROMPT_ACCEPTED:
            log.info("prompt accepted: %r", str(data.get("prompt", ""))[:60])
        elif msg_type == MESSAGE_COMMAND_ERROR:
            log.error(
                "model rejected %s: %s", data.get("command"), data.get("reason")
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

        Always call this: a live session holds two real GPU workers.
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

    # ── episodes ────────────────────────────────────────────────────────────

    async def reset_episode(self, reason: str) -> None:
        """End the episode: clears the model's prompt and causal cache.

        ``obs_seq`` restarts at 0 for the next episode, so the high-water mark
        has to reset with it. Keeping the old mark would make every fresh
        chunk look stale and the gate would wait forever.
        """
        if not self._connected:
            return
        await self._reactor.send_command(CMD_RESET, {})
        dropped = 0
        while not self._chunks.empty():
            self._chunks.get_nowait()
            dropped += 1
        self._prompt = None
        self._obs_seq_high = -1
        self._episode_started.clear()
        self.diag.resets += 1
        log.info("episode reset (%s); dropped %d queued chunk(s)", reason, dropped)
        # Give the model a moment to unwind before the next observation
        # restarts the episode.
        await asyncio.sleep(0.5)

    # ── one request ─────────────────────────────────────────────────────────

    async def predict(
        self,
        frames: dict[str, np.ndarray],
        joints: list[float],
        gripper: float,
        prompt: str,
    ) -> np.ndarray:
        """Push one observation and return the chunk that provably saw it."""
        await self.ensure_connected()
        self.diag.requests += 1

        # State first: the model snapshots the latest value together with the
        # frames it consumes, so it has to be in place before the frames land.
        await self._reactor.send_command(CMD_SET_JOINTS, {FIELD_JOINTS: joints})
        await self._reactor.send_command(
            CMD_SET_GRIPPER, {FIELD_GRIPPER: float(gripper)}
        )
        starting_episode = bool(prompt) and prompt != self._prompt
        if starting_episode:
            await self._reactor.send_command(CMD_SET_PROMPT, {FIELD_PROMPT: prompt})
            self._prompt = prompt

        # Everything already emitted was computed without this observation.
        self._absorb_queued()
        seq_floor = self._obs_seq_high
        if starting_episode:
            # A new prompt starts a new episode, and obs_seq restarts at 0 with
            # it. Carrying the previous episode's high-water mark forward would
            # make every chunk of the new episode look stale, and the gate
            # would wait out the whole chunk timeout for a number that is never
            # coming. Normally reset_episode() has already cleared the mark;
            # this covers a prompt that changes without one. The stored mark has
            # to go back too, or the NEXT request in this episode would inherit
            # the old episode's number and stall instead.
            seq_floor = -1
            self._obs_seq_high = -1
            self._episode_started.clear()

        t0 = time.perf_counter()
        await self._push(frames)

        if not self._episode_started.is_set():
            await self._require_episode_started()
        return await self._await_fresh_chunk(seq_floor, t0)

    def _absorb_queued(self) -> None:
        """Raise the high-water mark past every chunk already queued."""
        while not self._chunks.empty():
            data = self._chunks.get_nowait()
            self._obs_seq_high = max(self._obs_seq_high, int(data.get("obs_seq", -1)))

    async def _push(self, frames: dict[str, np.ndarray]) -> None:
        """Publish one frame per camera. See point 3 in the module header."""
        if not self._primed and self.prime_stagger_s > 0:
            log.info(
                "priming senders one at a time (%.1fs apart) so the order the "
                "three video streams appear on the wire is deterministic",
                self.prime_stagger_s,
            )
            for i, name in enumerate(TRACKS):
                await self._tracks[name].push(frames[name])
                log.info("  primed %s", name)
                if i < len(TRACKS) - 1:
                    await asyncio.sleep(self.prime_stagger_s)
            self._primed = True
            return
        self._primed = True
        for name in TRACKS:
            await self._tracks[name].push(frames[name])

    async def _require_episode_started(self, timeout: float = 45.0) -> None:
        """Fail fast if the model never saw all three cameras.

        The model emits ``episode_started`` as soon as it has a prompt plus one
        frame on *every* camera. If that has not arrived shortly after the
        first observation, the frames are not reaching it, and no amount of
        further waiting will help, so raise here rather than burn the whole
        chunk timeout and then hand RoboLab a stale plan.
        """
        try:
            await asyncio.wait_for(self._episode_started.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            if not self._chunks.empty():
                return  # chunks are arriving; we simply missed the event
            raise RuntimeError(
                f"the model reported no episode_started within {timeout:.0f}s of "
                "the first observation, so it is not receiving all three "
                "cameras. Check that a prompt was set, that all three tracks "
                f"({', '.join(TRACKS)}) were published after READY, and that "
                "the frames are uint8 RGB."
            ) from None

    async def _await_fresh_chunk(self, seq_floor: int, t0: float) -> np.ndarray:
        """Return the first chunk whose ``obs_seq`` beats *seq_floor*.

        On timeout, fall back to the newest stale chunk and say so loudly: a
        lagging plan beats a dead episode, but a run containing these warnings
        is not a valid data point.
        """
        deadline = time.monotonic() + self.chunk_timeout_s
        stale: np.ndarray | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                data = await asyncio.wait_for(
                    self._chunks.get(), timeout=min(remaining, 5.0)
                )
            except asyncio.TimeoutError:
                continue
            actions, obs_seq, chunk_index, seconds = decode_chunk(data)
            self._obs_seq_high = max(self._obs_seq_high, obs_seq)
            if obs_seq <= seq_floor:
                # In flight when we pushed => computed from the PREVIOUS
                # observation. Plausible-looking and wrong.
                self.diag.stale_discarded += 1
                stale = actions
                log.debug(
                    "discarding in-flight chunk %d (obs_seq=%d <= floor %d)",
                    chunk_index,
                    obs_seq,
                    seq_floor,
                )
                continue
            self.diag.chunks_returned += 1
            self.diag.inference_seconds.append(seconds)
            log.info(
                "request %d -> chunk %d: %s obs_seq=%d (floor %d), model %.0f ms, "
                "round trip %.0f ms",
                self.diag.requests,
                chunk_index,
                actions.shape,
                obs_seq,
                seq_floor,
                seconds * 1e3,
                (time.perf_counter() - t0) * 1e3,
            )
            return actions

        if stale is not None:
            self.diag.stale_returned += 1
            log.warning(
                "no chunk with obs_seq > %d within %.0fs; returning a stale "
                "chunk. The plan lags the observation and this run is not a "
                "valid data point.",
                seq_floor,
                self.chunk_timeout_s,
            )
            return stale
        raise TimeoutError(
            f"the model produced no action_chunk within {self.chunk_timeout_s:.0f}s "
            f"(obs_seq floor {seq_floor}). Is a prompt set and are all three "
            "camera tracks flowing?"
        )
