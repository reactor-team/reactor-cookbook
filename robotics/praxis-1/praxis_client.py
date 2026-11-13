# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""A closed-loop client for the Praxis-1 robot policies on Reactor.

Two hosted models share one protocol:

- ``reactor/praxis-1-droid``: one Franka arm, cameras ``exterior`` and ``wrist``.
- ``reactor/praxis-1-yam``: two YAM arms, cameras ``top``, ``wrist_left`` and
  ``wrist_right``.

You implement :class:`Robot` (read the cameras, read the arm state, execute one
command row). :class:`PraxisClient` runs the loop at 15 Hz:

1. Every tick, read ``t = reactor_sdk.time_micros()`` once, capture every camera
   and the arm state, push each frame with ``capture_time_us=t``, and keep the
   state row with ``time_us = t``.
2. After the last row of a chunk, send the last 25 state rows
   (``set_robot_state_json``), then echo the chunk's ``step``
   (``set_executed_step``). The model then computes the next chunk.

Frames and state reach the model on different channels. The shared stamp is the
only thing that pairs them; see the README.

:class:`SteppedPraxisClient` drives the same robot in a stepped session
instead. Nothing streams and nothing is stamped: after each chunk, one
``Reactor.step`` carries the chunk's frames of every camera (H.264, one stream
per camera) together with the state and the step echo, so frames and state
arrive as one unit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

import numpy as np
from reactor_sdk import EncodedFrame, Reactor, ReactorStatus, time_micros

log = logging.getLogger("praxis")

#: Control rate of both models, and of every list in the protocol.
FPS = 15
#: State rows (and frames per camera) per policy call: the previous call's
#: current row, then the 24 rows executed since.
STACK = 25


@dataclass(frozen=True)
class Embodiment:
    """What differs between the two models."""

    model: str
    cameras: tuple[str, ...]
    #: Arm names. A single-arm robot has one arm, and its state is sent flat.
    arms: tuple[str, ...]
    #: Largest frame as (width, height), or ``None`` to bound the longer side.
    max_size: tuple[int, int] | None
    max_side: int | None
    #: ``max_length`` of ``robot_state_json`` in the model's contract.
    max_state_chars: int

    def check_frame(self, name: str, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name}: expected an (H, W, 3) uint8 RGB frame")
        height, width = frame.shape[:2]
        if self.max_size and (width > self.max_size[0] or height > self.max_size[1]):
            raise ValueError(
                f"{name}: {width}x{height} is over {self.max_size[0]}x"
                f"{self.max_size[1]}; scale it down with the aspect kept"
            )
        if self.max_side and max(width, height) > self.max_side:
            raise ValueError(
                f"{name}: {width}x{height} is over {self.max_side} px on its longer "
                "side; scale it down with the aspect kept"
            )


DROID = Embodiment(
    model="reactor/praxis-1-droid",
    cameras=("exterior", "wrist"),
    arms=("arm",),
    max_size=(512, 288),
    max_side=None,
    max_state_chars=16_000,
)
YAM = Embodiment(
    model="reactor/praxis-1-yam",
    cameras=("top", "wrist_left", "wrist_right"),
    arms=("left", "right"),
    max_size=None,
    max_side=416,
    max_state_chars=32_000,
)
EMBODIMENTS = {"droid": DROID, "yam": YAM}


@dataclass(frozen=True)
class ArmState:
    """One arm's measured state at one tick.

    ``ee_pose`` is ``[x, y, z, qw, qx, qy, qz]``: metres, quaternion scalar
    first. DROID reports it in the frame named by ``pose_frame``; YAM in the
    rig's world frame, which both arms share. ``grip`` is absolute, 0 = open.
    """

    ee_pose: Sequence[float]
    grip: float


@dataclass(frozen=True)
class ArmTarget:
    """One arm's target for one command row."""

    position: np.ndarray  # (3,) metres
    rotation: np.ndarray  # (3, 3) rotation matrix
    grip: float  # absolute, 0 = open


@dataclass(frozen=True)
class CommandRow:
    """One row of a chunk, to execute at one 15 Hz tick."""

    step: int  # the chunk's step
    index: int  # the row's index in the chunk
    time_s: float  # offset from the chunk's first row
    targets: Mapping[str, ArmTarget]  # by arm name, as Embodiment.arms
    #: DROID only: the control law (``{"type": "impedance", "kq": ..., ...}``),
    #: or ``None`` when the arm's own control law applies.
    control: Mapping[str, Any] | None


@dataclass
class Chunk:
    step: int
    rows: list[CommandRow]
    #: From the request that triggered the chunk to its arrival.
    latency_ms: float = 0.0
    #: How the call paired frames and state (``session_status.join``), when
    #: polled for this chunk; see JOIN_MEANINGS.
    join: str | None = None
    #: Stepped sessions only: the encoded video the call carried, in bytes.
    sent_bytes: int = 0


class Robot(Protocol):
    """What the client needs from your robot. Every method must return quickly
    (well inside one 66 ms tick); hand slow work to your own threads."""

    def read_cameras(self) -> Mapping[str, np.ndarray]:
        """The newest RGB frame of every camera, by camera name."""
        ...

    def read_arm_state(self) -> Mapping[str, ArmState]:
        """The measured state of every arm, by arm name."""
        ...

    def execute(self, row: CommandRow) -> None:
        """Start moving toward this row's targets. Called once per tick."""
        ...


@dataclass(frozen=True)
class _Row:
    time_us: int | None  # None in a stepped session, which sends no stamps
    state: Mapping[str, ArmState]


class PraxisClient:
    """One session with a Praxis-1 model, driving one robot.

    Use it as an async context manager, then call :meth:`run_episode`.
    """

    #: The ``join`` every call should report; anything else is logged.
    expected_join: str | None = "capture_time"

    def __init__(
        self,
        robot: Robot,
        embodiment: Embodiment,
        *,
        model: str | None = None,
        api_url: str | None = None,
        pose_frame: str | None = None,
        stream_while_waiting: bool = True,
        status_every: int = 10,
        tick_s: float = 1 / FPS,
        connect_timeout_s: float = 900,
        chunk_timeout_s: float = 60,
    ) -> None:
        key = os.environ.get("REACTOR_API_KEY")
        if not key:
            raise RuntimeError("Set REACTOR_API_KEY to a key with access to this model")
        if pose_frame is not None and embodiment is not DROID:
            raise ValueError("pose_frame applies to praxis-1-droid only")
        model = model or embodiment.model
        if "/" not in model:
            raise ValueError(f"Name the model as owner/name, not {model!r}")
        self.robot = robot
        self.embodiment = embodiment
        self.pose_frame = pose_frame
        self.stream_while_waiting = stream_while_waiting
        self.status_every = status_every
        self.tick_s = tick_s
        self.connect_timeout_s = connect_timeout_s
        self.chunk_timeout_s = chunk_timeout_s
        self.reactor = Reactor(
            model,
            api_key=key,
            api_url=api_url
            or os.environ.get("REACTOR_API_URL")
            or "https://api.reactor.inc",
            **self._session_options(),
        )
        self.tracks: dict[str, Any] = {}
        self.history: deque[_Row] = deque(maxlen=STACK)
        #: The last ``session_status`` reply (``waiting_for``, ``join``, ...).
        self.status: dict[str, Any] = {}
        self._messages: asyncio.Queue[dict] = asyncio.Queue()
        self._next_step = 0  # the step the next action_prediction must carry
        self._chunks = 0
        self._episodes = 0
        self._deadline = 0.0

    # -- session ------------------------------------------------------------------

    def _session_options(self) -> dict[str, Any]:
        """Extra ``Reactor`` arguments; a streamed session needs none."""
        return {}

    async def __aenter__(self) -> PraxisClient:  # noqa: PYI034 - Python 3.10
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()

        @self.reactor.on_status
        def on_status(status: str) -> None:
            if status == ReactorStatus.READY:
                loop.call_soon_threadsafe(ready.set)

        @self.reactor.on_message
        def on_message(message: Any) -> None:
            if isinstance(message, dict) and message.get("type") in (
                "action_prediction",
                "prediction_failed",
            ):
                loop.call_soon_threadsafe(self._messages.put_nowait, message)

        try:
            await asyncio.wait_for(self.reactor.connect(), self.connect_timeout_s)
            await asyncio.wait_for(ready.wait(), self.connect_timeout_s)
            for name in self.embodiment.cameras:
                self.tracks[name] = await self.reactor.publish_track(name)
            if self.pose_frame is not None:
                await self.reactor.send_command(
                    "set_pose_frame", {"pose_frame": self.pose_frame}
                )
            return self
        except BaseException:
            await self.reactor.disconnect()
            raise

    async def __aexit__(self, *exc_info: object) -> None:
        await self.reactor.disconnect()

    # -- the loop -----------------------------------------------------------------

    async def run_episode(self, task: str, chunks: int) -> list[Chunk]:
        """Run one episode of ``chunks`` chunks and return them.

        The robot holds still for the first 25 ticks, so the first call has 25
        real frames and rows. Each chunk is executed row by row at 15 Hz. The last
        chunk of the episode is executed but not echoed; the next episode starts
        with ``reset_context``.
        """
        if chunks < 1:
            raise ValueError("chunks must be positive")
        await self.reactor.send_command("set_task", {"task": task})
        self.history.clear()
        self._deadline = time.monotonic()
        for _ in range(STACK):  # episode start: 25 ticks of the start pose
            await self._tick()
            self._observe(record=True)
        await self._send_state()
        sent = time.perf_counter()
        if self._episodes > 0:
            # The session is still waiting for the last chunk's echo; a reset drops
            # that wait and starts a new context from the state just sent.
            reply = await self.reactor.send_command("reset_context", {})
            data = (reply or {}).get("data") or {}
            self._next_step = int(data.get("next_step", self._next_step))
        self._episodes += 1

        done: list[Chunk] = []
        for n in range(chunks):
            chunk = await self._next_chunk(sent)
            done.append(chunk)
            poll = None
            if self.status_every and (self._chunks - 1) % self.status_every == 0:
                # Runs while the chunk executes; reports this chunk's call.
                poll = asyncio.create_task(self._poll_status(chunk))
            for row in chunk.rows:
                self.robot.execute(row)
                await self._tick()  # the row has had one tick to act...
                self._observe(record=True)  # ...then this is the state it reached
            if n < chunks - 1:
                await self._send_state()
                await self.reactor.send_command(
                    "set_executed_step", {"executed_step": chunk.step}
                )
                sent = time.perf_counter()
            if poll is not None:
                await poll
        return done

    async def _tick(self) -> None:
        """Sleep until the next 15 Hz tick. Scheduling uses ``time.monotonic``;
        stamps use ``time_micros`` (they never need to agree)."""
        self._deadline += self.tick_s
        delay = self._deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -self.tick_s:
            log.warning("control loop %.0f ms late; resyncing", -delay * 1000)
            self._deadline = time.monotonic()

    def _observe(self, *, record: bool) -> None:
        """One tick's observation: one stamp for every camera and the state."""
        t = time_micros()  # read ONCE, on the engine's clock (not time.time())
        frames = self.robot.read_cameras()
        state = dict(self.robot.read_arm_state()) if record else None
        for name in self.embodiment.cameras:
            frame = np.ascontiguousarray(frames[name])
            self.embodiment.check_frame(name, frame)
            self.tracks[name].push_frame(frame, capture_time_us=t)
        if state is not None:
            if set(state) != set(self.embodiment.arms):
                raise ValueError(f"read_arm_state must return {self.embodiment.arms}")
            self.history.append(_Row(t, state))

    async def _send_state(self) -> None:
        """Send the last 25 rows: the previous call's current row, then the rows
        executed since (or, at episode start, the 25 still ticks)."""
        text = state_json(self.embodiment, list(self.history))
        await self.reactor.send_command(
            "set_robot_state_json", {"robot_state_json": text}
        )

    async def _next_chunk(self, sent: float) -> Chunk:
        """Wait for the next ``action_prediction``. Cameras keep streaming while
        waiting (stamped, not recorded): the model pairs frames to state rows by
        stamp, so the extra frames are skipped."""
        self._deadline = time.monotonic()
        give_up = sent + self.chunk_timeout_s
        while True:
            if time.perf_counter() > give_up:
                raise TimeoutError(f"No action_prediction for step {self._next_step}")
            next_tick = self._deadline + self.tick_s
            timeout = max(0.0, next_tick - time.monotonic())
            try:
                message = await asyncio.wait_for(self._messages.get(), timeout)
            except asyncio.TimeoutError:
                self._deadline = max(next_tick, time.monotonic() - self.tick_s)
                if self.stream_while_waiting:
                    self._observe(record=False)
                continue
            data = message.get("data") or {}
            if message["type"] == "prediction_failed":
                # The session retries by itself; its next chunk opens a new episode.
                log.warning(
                    "prediction_failed step=%s code=%s: %s",
                    data.get("step"),
                    data.get("code"),
                    data.get("reason"),
                )
                continue
            if int(data.get("step", -1)) < self._next_step:
                continue  # a stale chunk from before a reset
            chunk = parse_chunk(self.embodiment, data)
            self._next_step = chunk.step + 1
            self._chunks += 1
            self._deadline = time.monotonic()  # the chunk's row 0 starts now
            chunk.latency_ms = (time.perf_counter() - sent) * 1000
            return chunk

    async def _poll_status(self, chunk: Chunk) -> None:
        """``get_status`` reports how the last call (``chunk``'s) paired frames
        and state."""
        try:
            reply = await self.reactor.send_command("get_status", {})
        except Exception as exc:  # noqa: BLE001 - a status poll must not stop the robot
            log.warning("get_status failed: %s", exc)
            return
        self.status = dict((reply or {}).get("data") or {})
        join = str(self.status.get("join", ""))
        chunk.join = join
        if join and self.expected_join and join != self.expected_join:
            log.warning(
                "join=%r, not %r: %s",
                join,
                self.expected_join,
                JOIN_MEANINGS.get(join, "unknown value"),
            )


class PredictionFailed(RuntimeError):
    """A stepped call that produced no chunk. Start a new episode to go on."""

    def __init__(self, step: int, code: str, reason: str) -> None:
        super().__init__(f"step {step}: {code}: {reason}")
        self.step, self.code, self.reason = step, code, reason


class SteppedPraxisClient(PraxisClient):
    """The same loop in a stepped session: one ``Reactor.step`` per chunk.

    The robot still executes one row per 15 Hz tick, and every tick is still
    observed, but nothing streams. The client keeps each tick's frames; after
    the chunk it encodes them, one H.264 stream per camera, and sends them in
    the same call as the state and the step echo. Frames and state arrive
    together, so no stamps are needed: the call sends no ``time_us``, and the
    model pairs the last 25 state rows with the 25 newest frames.

    The call carries 25 frames per camera at episode start, then the frames of
    the rows executed since the previous call: 21 after an episode's first
    chunk, 24 after the others.
    """

    expected_join = None  # "newest" is normal here: each call carries its frames

    def __init__(self, robot: Robot, embodiment: Embodiment, **kwargs: Any) -> None:
        super().__init__(robot, embodiment, **kwargs)
        self.frames: dict[str, list[np.ndarray]] = {}

    def _session_options(self) -> dict[str, Any]:
        return {"generation": "stepped", "step_timeout": self.chunk_timeout_s}

    async def run_episode(self, task: str, chunks: int) -> list[Chunk]:
        """Run one episode of ``chunks`` chunks and return them.

        As in :meth:`PraxisClient.run_episode`: the robot holds still for 25
        ticks, then each chunk is executed row by row at 15 Hz. The last chunk is
        executed but not echoed; the next episode starts with ``reset_context``.
        """
        if chunks < 1:
            raise ValueError("chunks must be positive")
        await self.reactor.send_command("set_task", {"task": task})
        self.history.clear()
        self.frames = {name: [] for name in self.embodiment.cameras}
        self._deadline = time.monotonic()
        for _ in range(STACK):  # episode start: 25 ticks of the start pose
            await self._tick()
            self._observe(record=True)
        if self._episodes > 0:
            # As in a streamed session: the reset drops the wait for the last
            # chunk's echo, and the next call starts a new context.
            reply = await self.reactor.send_command("reset_context", {})
            data = (reply or {}).get("data") or {}
            self._next_step = int(data.get("next_step", self._next_step))
        self._episodes += 1

        done: list[Chunk] = []
        executed: int | None = None
        for _ in range(chunks):
            chunk = await self._step(executed)
            done.append(chunk)
            poll = None
            if self.status_every and (self._chunks - 1) % self.status_every == 0:
                poll = asyncio.create_task(self._poll_status(chunk))
            self._deadline = time.monotonic()  # the chunk's row 0 starts now
            for row in chunk.rows:
                self.robot.execute(row)
                await self._tick()
                self._observe(record=True)
            executed = chunk.step
            if poll is not None:
                await poll
        return done

    def _observe(self, *, record: bool = True) -> None:
        """One tick's observation: keep every camera's frame and the state row.
        Nothing is sent and nothing is stamped."""
        if not record:
            return
        frames = self.robot.read_cameras()
        state = dict(self.robot.read_arm_state())
        if set(state) != set(self.embodiment.arms):
            raise ValueError(f"read_arm_state must return {self.embodiment.arms}")
        for name in self.embodiment.cameras:
            frame = np.array(frames[name], order="C")  # a copy: drivers reuse buffers
            self.embodiment.check_frame(name, frame)
            self.frames[name].append(frame)
        self.history.append(_Row(None, state))

    async def _step(self, executed: int | None) -> Chunk:
        """Send the frames kept since the previous call, the last 25 state rows
        and (after the first chunk) the step echo, in one call; return the chunk
        it produces."""
        started = time.perf_counter()
        state: dict[str, Any] = {
            "robot_state_json": state_json(self.embodiment, list(self.history))
        }
        if executed is not None:
            state["executed_step"] = executed
        frames = self.frames
        self.frames = {name: [] for name in self.embodiment.cameras}
        media = await asyncio.to_thread(
            lambda: {name: encode_h264(f) for name, f in frames.items()}
        )
        result = await self.reactor.step(state=state, media=media)
        if result.error is not None:
            raise PredictionFailed(self._next_step, "step_failed", str(result.error))
        give_up = started + self.chunk_timeout_s
        while True:
            timeout = give_up - time.perf_counter()
            try:
                message = await asyncio.wait_for(self._messages.get(), max(0, timeout))
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"No action_prediction for step {self._next_step}"
                ) from None
            data = message.get("data") or {}
            if message["type"] == "prediction_failed":
                # Nothing retries a stepped call by itself: stop the episode.
                raise PredictionFailed(
                    int(data.get("step", self._next_step)),
                    str(data.get("code", "")),
                    str(data.get("reason", "")),
                )
            if int(data.get("step", -1)) < self._next_step:
                continue  # a stale chunk from before a reset
            chunk = parse_chunk(self.embodiment, data)
            self._next_step = chunk.step + 1
            self._chunks += 1
            chunk.latency_ms = (time.perf_counter() - started) * 1000
            chunk.sent_bytes = sum(len(f.data) for v in media.values() for f in v)
            return chunk


def encode_h264(frames: Sequence[np.ndarray]) -> list[EncodedFrame]:
    """One camera's frames of one call as an H.264 stream that starts at a
    keyframe: one ``EncodedFrame`` per frame, in order.

    Raw frames would make a call about 22 MB; encoded, it is about 60-130 KB.
    Needs PyAV (``av``), which bundles the x264 encoder.
    """
    try:
        import av
    except ImportError as exc:  # streamed sessions do not need it
        raise ImportError(
            "Stepped sessions encode video with PyAV: pip install av"
        ) from exc
    if not frames:
        raise ValueError("no frames to encode")
    height, width = frames[0].shape[:2]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 needs an even width and height, not {width}x{height}")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("every frame of one camera in one call must be the same size")
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "yuv420p"
    # No B-frames and no lookahead, so each frame comes out as one packet, in
    # order; one GOP longer than the call, so the first frame is the only
    # keyframe.
    codec.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
        "bf": "0",
    }
    codec.gop_size = len(frames) + 1
    packets = []
    for i, frame in enumerate(frames):
        video = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(
            format="yuv420p"
        )
        video.pts = i
        packets += codec.encode(video)
    packets += codec.encode(None)
    if len(packets) != len(frames):
        raise RuntimeError(f"{len(frames)} frames encoded to {len(packets)} packets")
    return [EncodedFrame(bytes(p), width=width, height=height) for p in packets]


JOIN_MEANINGS = {
    "capture_time": "every state row was paired with the frames stamped with it",
    "cameras": "no time_us was sent; the newest moments all cameras share were used",
    "nearest": "some row had no frame within 33 ms after 0.5 s: dropped or late "
    "frames, or stamps slightly off; the nearest frames stood in",
    "newest": "no frame within 1 s of time_us: the stamps are not on the "
    "time_micros() clock (or a frame was pushed without capture_time_us)",
}
#: The same values in a stepped session, where the call sends no ``time_us``.
STEPPED_JOIN_MEANINGS = {
    "newest": "expected in a stepped session: each call carries its own frames and "
    "state, and the last 25 rows were paired with the call's 25 newest frames",
}


def state_json(embodiment: Embodiment, rows: Sequence[_Row]) -> str:
    """``robot_state_json`` for the given rows, oldest first. It carries
    ``time_us`` when the rows are stamped (streamed), and none when no row is
    (stepped)."""
    if len(rows) < STACK:
        raise ValueError(f"need {STACK} rows, have {len(rows)}")
    times = [row.time_us for row in rows]
    stamped = [t for t in times if t is not None]
    if stamped and len(stamped) != len(times):
        raise ValueError("stamp every row or none")
    if any(b <= a for a, b in pairwise(stamped)):
        raise ValueError("time_us must be strictly ascending")

    def arm(name: str) -> dict[str, list]:
        poses = [[float(v) for v in row.state[name].ee_pose] for row in rows]
        if any(len(p) != 7 or not all(map(math.isfinite, p)) for p in poses):
            raise ValueError(f"{name}: ee_pose must be 7 finite numbers")
        return {
            "ee_pose": poses,
            "grip": [float(row.state[name].grip) for row in rows],
        }

    if len(embodiment.arms) == 1:
        doc: dict[str, Any] = arm(embodiment.arms[0])
    else:
        doc = {name: arm(name) for name in embodiment.arms}
    if stamped:
        doc["time_us"] = stamped
    text = json.dumps(doc, separators=(",", ":"))
    if len(text) > embodiment.max_state_chars:
        raise ValueError(f"robot_state_json is {len(text)} characters, over the limit")
    return text


def parse_chunk(embodiment: Embodiment, data: Mapping[str, Any]) -> Chunk:
    """An ``action_prediction`` payload as command rows."""
    step = int(data["step"])
    times = [float(t) for t in data["timestamps"]]
    if len(embodiment.arms) == 1:
        fields = {embodiment.arms[0]: ("poses", "grips")}
    else:
        fields = {name: (f"{name}_poses", f"{name}_grips") for name in embodiment.arms}
    arms = {}
    for name, (poses_key, grips_key) in fields.items():
        poses = np.asarray(data[poses_key], dtype=np.float64)
        grips = np.asarray(data[grips_key], dtype=np.float64)
        if poses.shape != (len(times), 12) or grips.shape != (len(times),):
            raise ValueError(f"step {step}: malformed {poses_key}/{grips_key}")
        if not (np.isfinite(poses).all() and np.isfinite(grips).all()):
            raise ValueError(f"step {step}: non-finite targets")
        arms[name] = (poses, grips)
    control_text = data.get("control_json") or ""
    control = json.loads(control_text) if control_text else None
    rows = [
        CommandRow(
            step=step,
            index=i,
            time_s=t,
            targets={
                name: ArmTarget(
                    position=poses[i, :3],
                    rotation=poses[i, 3:].reshape(3, 3),
                    grip=float(grips[i]),
                )
                for name, (poses, grips) in arms.items()
            },
            control=control,
        )
        for i, t in enumerate(times)
    ]
    return Chunk(step, rows)
