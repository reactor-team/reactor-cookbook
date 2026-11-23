"""A talking avatar on LiveKit's avatar protocol, the application half.

:class:`AvatarApp` is the :class:`ReactorApp` for any model that renders a
block of video from a block of speech (:class:`avatar_contract.AvatarModel`).
It declares the client contract and drives the model with its own loop; a
subclass only builds the model in :meth:`AvatarApp.create_model`.

**The contract.** One input track, ``agent_audio``: the speech the avatar says,
one utterance at a time. Two output tracks, ``main_video`` and ``main_audio``:
the avatar's frames and the speech as it plays with them. Two messages,
``playback_started`` and ``playback_finished``, and three commands,
``clear_buffer``, ``user_speaking_state`` and ``set_avatar``. The manifest maps
them onto LiveKit's avatar protocol: speech as byte streams on
``lk.audio_stream``, an interrupt as the ``lk.clear_buffer`` RPC, and the
playback reports as RPC calls back to the agent.

**The loop.** :meth:`AvatarApp.run` repeats four steps while a session is live:
plan the next block, render it on the render thread, pair its frames with the
audio they play with, and emit it. The render runs off the event loop, so
commands, hooks and the session's two tasks run during a render: the speech
pump, which moves arriving speech into utterances, and the playback reports.

**Utterances.** Speech arrives faster than real time, as one byte stream per
utterance. Each frame's metadata names its stream, and a frame with no samples
marks the stream's end. A block carries speech from one utterance only: a full
block as soon as one is in, and the utterance's last, shorter part once its
end has arrived. With no speech to render, the avatar renders silent blocks,
and listens after ``listen_after_blocks`` of them or while the user speaks.

**The playout clock.** Every emitted block is scheduled on a clock that mirrors
the runtime's pacer: it starts when the block before it ends, or now if nothing
is playing. The clock paces the blocks (a silent block starts when the video
queued would run out while it renders; speech renders up to ``speech_lead_s``
ahead) and dates the playback reports. When an utterance's first block lands
behind idle frames only, the idle frames are dropped so the speech plays at
once.

**Interrupts.** ``clear_buffer`` lands at once, mid-render included: the sound
stops, the last frame holds until the next block, the speech still queued is
dropped, and ``playback_finished`` reports how far the utterance played. A
block rendered for the cut speech is dropped when it lands.

**Failures.** A session that cannot run does not start: ``set_avatar``
refuses a portrait that is not a decodable image. When the model raises while
a session runs, that session renders nothing more: the last frame holds,
``avatar_failed`` tells its clients why, and the model is reset when the
session ends, as after every session. The runtime stays up. Each portrait gets
an id unique for the app's life, and the app tracks the one the model has
bound, so a block carries the portrait whenever the model does not hold it.
"""

from __future__ import annotations

import asyncio
import io
import itertools
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from reactor_runtime import (
    Audio,
    ClientInfo,
    CommandError,
    InputBuffer,
    InputField,
    InputFrame,
    MediaInput,
    MessageField,
    ModelMessage,
    Output,
    ReactorApp,
    ReadMode,
    StepCompleted,
    UploadedFile,
    Video,
    event,
    get_logger,
    session_ended,
    session_started,
)

from avatar_contract import AvatarInput, AvatarModel, AvatarResult

logger = get_logger(__name__)

#: Rate of the speech on ``agent_audio`` and of the sound on ``main_audio``.
SAMPLE_RATE = 24_000
#: How often the playback reports are checked against the clock.
_REPORT_INTERVAL_S = 0.02
#: How often the speech pump moves arrived speech into its utterances.
_PUMP_INTERVAL_S = 0.01
#: The longest the loop sleeps before looking at the clock again when nothing wakes it.
_MAX_WAIT_S = 0.1
#: How long a finished utterance is kept, so late frames of it are recognised and dropped.
_UTTERANCE_TTL_S = 30.0


class AgentAudio(Audio):
    """Speech to lip-sync to, mono at 24 kHz. Holds two minutes, more than one utterance needs."""

    sample_rate = SAMPLE_RATE
    buffer_seconds = 120


class AvatarAudio(Audio):
    """The speech as it plays with the avatar's frames, mono at 24 kHz."""

    sample_rate = SAMPLE_RATE


class AvatarMedia(MediaInput):
    """The speech the avatar says."""

    agent_audio: AgentAudio


class AvatarOutput(Output):
    """The avatar's frames and the speech that plays with them."""

    main_video: Video
    main_audio: AvatarAudio


class PlaybackStarted(ModelMessage):
    """Sent when the first frame of an utterance received on `agent_audio` plays on `main_video`."""


class AvatarFailed(ModelMessage):
    """Sent when the avatar cannot go on in this session; no more frames follow until the session ends."""

    reason: str = MessageField(description="Why the avatar stopped.")


class PlaybackFinished(ModelMessage):
    """Sent when an utterance received on `agent_audio` has played to its end, or `clear_buffer` cut it."""

    playback_position: float = MessageField(
        description="Seconds of the utterance that played."
    )
    interrupted: bool = MessageField(
        description="Whether `clear_buffer` cut the utterance short."
    )


@dataclass(frozen=True)
class AvatarSettings:
    """How the application paces and shapes the blocks; ``config.yml`` sets each key.

    Attributes:
        listen_after_blocks: Silent blocks after an utterance before the avatar listens.
        silence_rms: A speech block quieter than this (int16 RMS) is marked silent.
        idle_margin_s: Video still queued, in seconds, when the next silent block lands.
        speech_lead_s: How far ahead of playout speech renders, in seconds. More rides out a slow
            block; less drops less rendered speech on an interrupt.
        cut_idle_for_speech: Drop the idle frames queued ahead of an utterance's first block, so the
            speech starts at once. The picture holds its last frame over the cut.
    """

    listen_after_blocks: int = 2
    silence_rms: float = 100.0
    idle_margin_s: float = 0.1
    speech_lead_s: float = 1.5
    cut_idle_for_speech: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AvatarSettings:
        names = cls.__dataclass_fields__
        return cls(
            **{
                k: type(getattr(cls, k))(v)
                for k, v in config.items()
                if k in names and v is not None
            }
        )


@dataclass
class _Block:
    """A block on the playout clock that carries speech."""

    start: float
    speech_seconds: float


@dataclass
class _Utterance:
    """One utterance: the speech still to render, and the blocks that carry what was rendered."""

    id: str
    chunks: deque[np.ndarray] = field(default_factory=deque)
    pending: int = 0
    #: Samples taken for a block that has not landed yet.
    rendering: int = 0
    received: int = 0
    ended: bool = False
    ended_at: float | None = None
    cleared: bool = False
    started_reported: bool = False
    finished_reported: bool = False
    blocks: list[_Block] = field(default_factory=list)

    def push(self, samples: np.ndarray) -> None:
        if samples.size:
            self.chunks.append(samples)
            self.pending += samples.size
            self.received += samples.size

    def take(self, n: int) -> np.ndarray:
        out: list[np.ndarray] = []
        need = n
        while need and self.chunks:
            head = self.chunks.popleft()
            if head.size > need:
                self.chunks.appendleft(head[need:])
                head = head[:need]
            out.append(head)
            need -= head.size
        taken = np.concatenate(out) if out else np.zeros(0, dtype=np.int16)
        self.pending -= taken.size
        return taken

    def give_back(self, samples: np.ndarray) -> None:
        if samples.size:
            self.chunks.appendleft(samples)
            self.pending += samples.size

    def drop(self) -> None:
        self.chunks.clear()
        self.pending = 0

    def position(self, now: float) -> float:
        """Seconds of the utterance played by ``now``."""
        return sum(min(max(now - b.start, 0.0), b.speech_seconds) for b in self.blocks)

    def finish_at(self) -> float | None:
        """When the last of the utterance plays, once its end has arrived and all of it is scheduled."""
        if not self.ended or self.pending or self.rendering or not self.blocks:
            return None
        return max(b.start + b.speech_seconds for b in self.blocks)


class _Playout:
    """The playout clock: when the last emitted frame finishes playing."""

    def __init__(self) -> None:
        self.end = 0.0

    def lead(self, now: float) -> float:
        return max(self.end - now, 0.0)

    def schedule(self, now: float, seconds: float) -> float:
        start = max(now, self.end)
        self.end = start + seconds
        return start

    def cut(self, now: float) -> None:
        self.end = now


class _Session:
    """Everything one session keeps."""

    def __init__(self, avatar: bytes, avatar_id: int) -> None:
        self.active = False
        self.utterances: dict[str, _Utterance] = {}
        self.playout = _Playout()
        self.silent_run = 0
        self.user_speaking = False
        self.avatar = avatar
        #: Unique across the app's sessions, so a portrait is never mistaken for another one the model bound.
        self.avatar_id = avatar_id
        #: The model failed in this session: nothing more renders until it ends.
        self.failed = False
        #: Seconds a block takes to render, smoothed; paces the silent blocks.
        self.render_s = 0.5
        #: When the last block that carries speech ends on the playout clock.
        self.speech_until = 0.0
        #: The block on the render thread, if any.
        self.rendering: _Plan | None = None

    def next_speech(self) -> _Utterance | None:
        """The utterance whose speech plays next, oldest first."""
        for u in self.utterances.values():
            if not u.cleared and u.pending:
                return u
        return None


@dataclass
class _Plan:
    """One block the loop renders: the model's input, and what the loop needs to land it."""

    input: AvatarInput
    #: Samples of the utterance's speech the block carries; the rest of ``input.audio`` is padding.
    speech_samples: int
    #: The utterance the block speaks; ``None`` for a silent block.
    utterance: _Utterance | None
    #: The session the block was planned in; a block that lands in another one is dropped.
    session: _Session
    #: The interrupt count when the block was planned: an interrupt since then drops a speech block.
    epoch: int
    #: Seconds the render took.
    elapsed: float = 0.0


class AvatarApp(ReactorApp):
    """A talking avatar that lip-syncs to streamed speech. Subclasses build the model."""

    media: AvatarMedia
    fps = 25
    #: Frames the pacer may hold, so ``emit()`` hands a block over without waiting: speech renders at
    #: most ``speech_lead_s`` ahead, and silent blocks stop ``idle_margin_s`` plus one block ahead.
    buffer_size = 100

    def create_model(
        self, config: dict[str, Any], config_dir: Path
    ) -> tuple[AvatarModel, bytes]:
        """Load the model and return it with the portrait a session starts from (PNG or JPEG bytes).

        Args:
            config: The parsed ``config.yml``, with the application's own keys included.
            config_dir: The folder of ``config.yml``, for paths relative to it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_model()"
        )

    def load(self, config_path: Path | None) -> None:
        """Read the config, build the model, and check it speaks this application's rates."""
        config: dict[str, Any] = {}
        if config_path is not None:
            config = yaml.safe_load(config_path.read_text()) or {}
        self.settings = AvatarSettings.from_config(config)
        model, avatar = self.create_model(
            config, config_path.parent if config_path else Path.cwd()
        )
        if model.sample_rate != SAMPLE_RATE or model.fps != self.fps:
            raise ValueError(
                f"the model must take {SAMPLE_RATE} Hz speech and render at {self.fps} FPS"
            )
        self.model = model
        self.block_frames = model.block_frames
        self.samples_per_frame = SAMPLE_RATE // self.fps
        self.block_samples = self.block_frames * self.samples_per_frame
        self.default_avatar = avatar
        self._init_loop_state()

    def _init_loop_state(self) -> None:
        self.clock = time.monotonic
        self._avatar_ids = itertools.count(1)
        #: The portrait the model has bound, or ``None`` after a reset; only the render loop changes it.
        self._bound_avatar_id: int | None = None
        self.session = _Session(self.default_avatar, next(self._avatar_ids))
        self._epoch = 0
        self._wake = asyncio.Event()
        self._render_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="avatar-render"
        )
        self._session_tasks: list[asyncio.Task[None]] = []
        self._reset_model = False

    # -- the loop -------------------------------------------------------------------

    async def run(self) -> None:
        """Plan a block, render it on the render thread, land it; for as long as the model runs."""
        while True:
            await self._until_live()
            plan = await self._next_plan()
            if plan is None:
                continue
            try:
                result = await self._render(plan)
            except Exception as error:
                await self._fail(plan.session, error)
                continue
            await self._land(plan, result)

    async def _until_live(self) -> None:
        """Wait for a started session with a client connected. Reset the model after a session ends."""
        if self._reset_model:
            self._reset_model = False
            await asyncio.get_running_loop().run_in_executor(
                self._render_thread, self.model.reset
            )
            self._bound_avatar_id = None
        while not self._session_live():
            await self._sleep_until_woken(_MAX_WAIT_S)

    async def _next_plan(self) -> _Plan | None:
        """The next block to render, once there is one; ``None`` if the session stops being live first."""
        while self._session_live():
            now = self.clock()
            plan = self._plan(now)
            if plan is not None:
                return plan
            await self._sleep_until_woken(self._wait_for(now))
        return None

    async def _render(self, plan: _Plan) -> AvatarResult:
        """Run the model on the render thread; the event loop stays free until it returns."""
        s = plan.session
        s.rendering = plan
        started = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._render_thread, self.model.generate, plan.input
            )
        finally:
            s.rendering = None
        plan.elapsed = time.perf_counter() - started
        s.render_s = 0.8 * s.render_s + 0.2 * plan.elapsed
        return result

    async def _land(self, plan: _Plan, result: AvatarResult) -> AvatarOutput | None:
        """Emit a rendered block, unless an interrupt or the session's end overtook it, and report it."""
        s = plan.session
        self._bound_avatar_id = result.avatar_id
        if plan.utterance is not None:
            plan.utterance.rendering -= plan.speech_samples
        speaks = plan.utterance is not None
        if s is not self.session or (speaks and plan.epoch != self._epoch):
            return None
        out = self._pair(plan, result, self.clock())
        await self.emit(out)
        if speaks and plan.epoch != self._epoch:
            # An interrupt landed while the block was handed to the pacer; its flush can miss a block
            # still on its way there.
            self.output.flush(hold=True)
        await self.complete_step(StepCompleted(output=out, elapsed=plan.elapsed))
        return out

    def _session_live(self) -> bool:
        return (
            self.session.active and not self.session.failed and self.connected.is_set()
        )

    async def _fail(self, session: _Session, error: Exception) -> None:
        """Stop rendering for ``session`` after a model failure and tell its clients why.

        The runtime stays up: the session renders nothing more, holding its last frame, until its caller
        ends it; the model is reset then, as after every session.
        """
        logger.error(
            "the avatar model failed; this session renders nothing more", exc_info=error
        )
        if session is not self.session:
            return
        session.failed = True
        self.output.flush(hold=True)
        await self.send(AvatarFailed(reason=f"{type(error).__name__}: {error}"))

    async def _sleep_until_woken(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except TimeoutError:
            pass
        self._wake.clear()

    def _wait_for(self, now: float) -> float:
        """How long until the queued video runs low enough for the next block, bounded."""
        s = self.session
        due = s.playout.lead(now) - (s.render_s + self.settings.idle_margin_s)
        return min(max(due, 0.005), _MAX_WAIT_S)

    # -- planning and landing a block -------------------------------------------------

    def _plan(self, now: float) -> _Plan | None:
        """What to render next, or ``None`` to wait.

        A speech block once a block of the next utterance is in, or its end has arrived, while playout
        is less than ``speech_lead_s`` ahead. Otherwise a silent block once the queued video would be
        down to ``idle_margin_s`` as it lands.
        """
        s = self.session
        self._drain(now)
        lead = s.playout.lead(now)
        u = s.next_speech()
        if u is not None and (u.pending >= self.block_samples or u.ended):
            if lead > self.settings.speech_lead_s:
                return None
            speech = u.take(self.block_samples)
            u.rendering += speech.size
            audio = np.zeros(self.block_samples, dtype=np.int16)
            audio[: speech.size] = speech
            s.silent_run = 0
            return self._make_plan(
                audio, speech.size, u, silent=_rms(speech) < self.settings.silence_rms
            )
        if lead > s.render_s + self.settings.idle_margin_s:
            return None
        if u is None:
            s.silent_run += 1
        listening = u is None and (
            s.user_speaking or s.silent_run > self.settings.listen_after_blocks
        )
        audio = np.zeros(self.block_samples, dtype=np.int16)
        return self._make_plan(audio, 0, None, silent=True, listening=listening)

    def _make_plan(
        self,
        audio: np.ndarray,
        speech_samples: int,
        utterance: _Utterance | None,
        *,
        silent: bool,
        listening: bool = False,
    ) -> _Plan:
        s = self.session
        new_avatar = s.avatar_id != self._bound_avatar_id
        model_input = AvatarInput(
            audio=audio,
            silent=silent,
            listening=listening,
            avatar=s.avatar if new_avatar else None,
            avatar_id=s.avatar_id,
        )
        return _Plan(model_input, speech_samples, utterance, s, self._epoch)

    def _pair(self, plan: _Plan, result: AvatarResult, now: float) -> AvatarOutput:
        """Pair a block's frames with the audio they play with, and put the block on the playout clock."""
        s = self.session
        u = plan.utterance
        audio = plan.input.audio
        n = int(result.frames.shape[0])
        used = min(n * self.samples_per_frame, self.block_samples)
        speech_used = min(plan.speech_samples, used)
        if u is not None and plan.speech_samples > used:
            # A clip's first block is shorter than the speech it took; the rest plays in the next block.
            u.give_back(audio[used : plan.speech_samples])
        if (
            u is not None
            and not u.blocks
            and self.settings.cut_idle_for_speech
            and s.speech_until <= now < s.playout.end
        ):
            # Only idle frames are queued ahead of the utterance's first block: drop them, holding the
            # picture, so the speech plays now rather than after them.
            self.output.flush(hold=True)
            s.playout.cut(now)
        start = s.playout.schedule(now, n / self.fps)
        if u is not None and speech_used and not u.cleared:
            u.blocks.append(_Block(start, speech_used / SAMPLE_RATE))
            s.speech_until = s.playout.end
        return AvatarOutput(
            main_video=result.frames, main_audio=audio[:used].reshape(1, -1)
        )

    # -- speech in --------------------------------------------------------------------

    async def _pump_speech(self) -> None:
        """Move arriving speech into its utterances while the session lasts."""
        while True:
            if self._drain(self.clock()):
                self._wake.set()
            await asyncio.sleep(_PUMP_INTERVAL_S)

    def _drain(self, now: float) -> int:
        """Move the speech that arrived on `agent_audio` into its utterances. Returns the frames moved."""
        s = self.session
        track = cast(InputBuffer, self.media.agent_audio)
        moved = 0
        while (frames := track.try_read(1, mode=ReadMode.FIFO)) is not None:
            for frame in frames:
                self._ingest(frame, now)
                moved += 1
        stale = [
            key
            for key, u in s.utterances.items()
            if (u.cleared or u.finished_reported)
            and u.ended_at is not None
            and now - u.ended_at > _UTTERANCE_TTL_S
        ]
        for key in stale:
            del s.utterances[key]
        return moved

    def _ingest(self, frame: InputFrame, now: float) -> None:
        info = _metadata(frame.metadata)
        key = str(info.get("stream_id", ""))
        s = self.session
        u = s.utterances.get(key)
        if u is None:
            u = s.utterances[key] = _Utterance(key)
        if info.get("ended"):
            u.ended = True
            u.ended_at = now
            return
        if u.cleared:
            return
        u.push(np.ascontiguousarray(frame.data, dtype=np.int16).ravel())

    # -- playback reports -------------------------------------------------------------

    async def report_playback(self) -> None:
        """Send the playback reports the clock has passed."""
        now = self.clock()
        due: list[ModelMessage] = []
        for u in self.session.utterances.values():
            if u.cleared or u.finished_reported or not u.blocks:
                continue
            if not u.started_reported and now >= u.blocks[0].start:
                u.started_reported = True
                due.append(PlaybackStarted())
            done = u.finish_at()
            if done is not None and now >= done:
                u.finished_reported = True
                due.append(
                    PlaybackFinished(
                        playback_position=round(u.received / SAMPLE_RATE, 3),
                        interrupted=False,
                    )
                )
        for message in due:
            await self.send(message)

    async def _report_loop(self) -> None:
        while True:
            try:
                await self.report_playback()
            except Exception:
                logger.exception("playback report failed")
            await asyncio.sleep(_REPORT_INTERVAL_S)

    # -- commands and hooks -----------------------------------------------------------

    @event(
        name="clear_buffer",
        description=(
            "Stop the current utterance and return to idle. The sound stops and the last frame holds "
            "until the next block; `playback_finished` follows with `interrupted` true and how far the "
            "utterance played. Speech of that utterance still arriving on `agent_audio` is dropped."
        ),
    )
    async def clear_buffer(self) -> None:
        """Cut what plays, drop the speech still queued or rendering, and report the interrupted utterance."""
        s = self.session
        now = self.clock()
        self._drain(now)
        active = [
            u
            for u in s.utterances.values()
            if not u.cleared and not u.finished_reported and u.received
        ]
        playing = next((u for u in active if u.blocks), active[0] if active else None)
        speech_rendering = s.rendering is not None and s.rendering.utterance is not None
        if playing is None and not speech_rendering:
            return
        position = playing.position(now) if playing is not None else 0.0
        for u in s.utterances.values():
            if not u.finished_reported:
                u.cleared = True
                u.drop()
        self._epoch += 1
        self.output.flush(hold=True)
        s.playout.cut(now)
        s.speech_until = now
        s.silent_run = self.settings.listen_after_blocks
        self._wake.set()
        if playing is not None:
            for u in active:
                u.finished_reported = True
            await self.send(
                PlaybackFinished(playback_position=round(position, 3), interrupted=True)
            )

    @event(
        name="user_speaking_state",
        description="Whether the user is speaking. While `speaking`, the avatar listens between utterances.",
    )
    def user_speaking_state(
        self,
        state: str = InputField(
            choices=["speaking", "idle"], description="`speaking` or `idle`."
        ),
    ) -> None:
        """Keep the user's speaking state for the next silent block."""
        self.session.user_speaking = state == "speaking"
        self._wake.set()

    @event(
        name="set_avatar",
        description=(
            "Replace the portrait. One face, frontal, PNG or JPEG; it is scaled and cropped to the "
            "model's output size. The next block starts a fresh clip from it. Only the session's starting "
            "input can send it; fails with `not_allowed` from any other client, `unsupported_media` when "
            "the upload is not an image, and `undecodable_image` when it cannot be decoded."
        ),
    )
    async def set_avatar(
        self,
        image: UploadedFile = InputField(description="The portrait."),  # noqa: B008 (the wire schema)
        client: ClientInfo | None = None,
    ) -> None:
        """Check the upload and make it the portrait the next block binds."""
        if client is None or not client.system:
            raise CommandError(
                "not_allowed", "Only the session's starting input can set the portrait."
            )
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} is not an image.")
        try:
            await asyncio.to_thread(_check_image, image.data)
        except UNDECODABLE as exc:
            raise CommandError(
                "undecodable_image", "The file is not a decodable image."
            ) from exc
        s = self.session
        s.avatar = image.data
        s.avatar_id = next(self._avatar_ids)
        logger.info("portrait accepted", name=image.name, avatar_id=s.avatar_id)
        self._wake.set()

    @session_started
    async def on_session_started(self) -> None:
        """Start the session from the default portrait, with its speech pump and playback reports."""
        self._epoch += 1
        self.session = _Session(self.default_avatar, next(self._avatar_ids))
        self.session.active = True
        loop = asyncio.get_running_loop()
        self._session_tasks = [
            loop.create_task(self._pump_speech()),
            loop.create_task(self._report_loop()),
        ]
        self._wake.set()

    @session_ended
    async def on_session_ended(self) -> None:
        """Stop the session's tasks, drop its speech and media, and have the loop reset the model."""
        self._epoch += 1
        self.session.active = False
        for task in self._session_tasks:
            task.cancel()
        self._session_tasks = []
        self._reset_model = True
        self.output.flush()
        cast(InputBuffer, self.media.agent_audio).clear()
        self.session = _Session(self.default_avatar, next(self._avatar_ids))
        self._wake.set()


# What Pillow raises for bytes it cannot decode: not an image, truncated or corrupt data (verify() reports
# a broken PNG as SyntaxError), or an image too large to decode safely (a ValueError subclass).
UNDECODABLE = (OSError, ValueError, SyntaxError)


def _check_image(data: bytes) -> None:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.verify()


def _metadata(raw: bytes | str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        info = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {}
    return info if isinstance(info, dict) else {}


def _rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
