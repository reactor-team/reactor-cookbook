"""FastH3 as a Reactor model: a queue of prompt-driven video-and-audio clips.

FastH3 is MiniMax-H3 distilled to four transformer forwards, and on a few
Blackwell GPUs it builds video about as fast as the video plays. This model
puts a queue in front of that: clients `enqueue` generation requests — a prompt
plus opaque metadata, each answered with a UUID — builds run through the queue
in order, and playback is a separate, explicit step. `play` streams one built
clip on `main_video` and `main_audio`; when it ends (or `stop` cuts it) the
stream flushes to black and holds until the next `play`. Nothing plays on its
own.

The unit of work is a whole clip, not a frame, which is why this subclasses
``ReactorModel`` and owns its own ``run()`` loop rather than using
``ReactorPipeline``. Command handlers then run on their own coroutines
concurrent with ``run()``, so `enqueue` and `stop` answer immediately even
while a clip is being built or played.

Layout:
  * ``fasth3_types.py``         — everything a client sees (tracks, `ClipInfo`, messages).
  * ``fasth3_queue.py``         — the bounded clip queue and its entries.
  * ``fasth3_backend.py``       — the FastVideo engine and its worker thread.
  * ``fasth3_assets.py``        — config parsing and weights validation.
  * ``fasth3_clip_plan.py``     — clip geometry (lengths, frame counts, canvases).
  * ``fasth3_session_rules.py`` — which commands each state accepts.
  * ``fasth3.yaml``             — the generation recipe, queue size, and weight layout.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from reactor_runtime import (
    ClientInfo,
    InputField,
    ReactorModel,
    connected,
    event,
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

import fasth3_clip_plan as clip_plan
import fasth3_session_rules as session_rules
from fasth3_assets import load_config, require_weights, resolve_model_path
from fasth3_backend import OUTPUT_SAMPLE_RATE, ClipJob, FastH3Backend
from fasth3_queue import ClipEntry, ClipQueue
from fasth3_types import (
    MAX_METADATA_CHARS,
    MAX_PROMPT_CHARS,
    AutoplayAccepted,
    CanvasAccepted,
    ClipFailed,
    ClipFinished,
    ClipLengthAccepted,
    ClipPopped,
    ClipQueued,
    ClipStarted,
    ClipStopped,
    CommandError,
    FastH3Output,
    QueueUpdate,
    SeedAccepted,
    SessionReset,
    StateUpdate,
)

logger = get_logger(__name__)

FRAME_RATE = clip_plan.FPS

# The clip-length range, rendered once so the command text and the schema's own
# bounds can never disagree.
_CLIP_RANGE = f"{clip_plan.MIN_SECONDS_PUBLISHED:g} and {clip_plan.MAX_SECONDS_PUBLISHED:g}"

# Frames per emitted slice. The runtime recorder's feed queue cannot absorb
# one-second bursts, and the emitter is a metronome either way, so smaller
# slices cost nothing.
EMIT_FRAMES = 3

# How often the idle loop re-checks for a play request and a finished build.
# Runs on the event loop, so this is a scheduling granularity, not a busy-wait.
POLL_SECONDS = 0.05


class FastH3(ReactorModel):
    """Queue prompt-driven clip generations and play them back one at a time."""

    # Pinned: `_emit_clip` is a strict 24 fps metronome and every emit omits
    # `compute_time`, which is exactly the "unmeasured" path this rate tags.
    # Measuring instead re-estimates the rate from observed timing, whose wobble
    # both drops chunks while converging and drifts video against the
    # sample-clocked audio.
    fps = FRAME_RATE
    # Two seconds of transport-side tolerance at 24 fps, so a hiccup dents the
    # buffer instead of dropping frames.
    buffer_size = 48

    def __init__(self) -> None:
        """Create the model shell; everything session-scoped arrives in load()."""
        super().__init__()
        # The build in flight: its entry, its job handle, and when it was
        # submitted (monotonic), so readiness latency is a measured number.
        self._build: tuple[ClipEntry, ClipJob, float] | None = None

    # ------------------------------------------------------------------ load

    def load(self, config_path: Path | None) -> None:
        """Parse the config, validate the weights, and build the warm engine.

        Runs once at startup, before any session. The runtime marks the pod
        ready only when this returns, so the backend's warm-up means a deployed
        pod never builds a cold clip.

        Args:
            config_path: Path to ``fasth3.yaml``; its ``inference`` block is the
                generation recipe and the queue size, and its ``runtime`` block
                holds the weight layout and the engine shape.
        """
        self.config = load_config(config_path)
        weights = get_weights_path()
        model_path = resolve_model_path(self.config, weights)
        require_weights(weights, model_path)

        self.backend = FastH3Backend(self.config, model_path)
        # Session-scoped state exists before the first session, so a command
        # racing ahead of `@session_started` reads defaults, never garbage.
        self._reset_session_state()
        self.backend.load()
        logger.info("fasth3 loaded", queue_capacity=self.config.queue_size)

    # -------------------------------------------------------- session state

    def _reset_session_state(self) -> None:
        """Return every session-scoped field to its default.

        Called once at ``load()`` and at every ``@session_started``, which is
        what keeps one session from ever observing another's queue or
        conditions. A build still in flight for the old session is cancelled;
        its result, if it completes anyway, is discarded by ``_pump_builds``
        because its entry no longer lives in the queue.
        """
        if self._build is not None:
            _entry, job, _submitted = self._build
            job.cancelled = True
            self._build = None

        # Conditions newly enqueued clips snapshot.
        self._clip_frames: int = self.config.clip_frames
        self._seed: int = self.config.seed
        self._aspect: str = self.config.aspect
        # Off by default: playback waits for an explicit `play`.
        self._autoplay: bool = False

        # The queue, and the playout lifecycle around it. `_play_request` is a
        # clip taken off the queue and armed for the run loop; `_playing` is
        # the clip whose frames are on the wire; `_stop_playout` asks the
        # emitter to cut it.
        self._queue = ClipQueue(self.config.queue_size)
        self._play_request: ClipEntry | None = None
        self._playing: ClipEntry | None = None
        self._stop_playout: bool = False

        # Progress, mirrored so a `state_update` is a complete snapshot.
        self._clips_played: int = 0
        self._frames_sent: int = 0
        self._seconds_sent: float = 0.0

    def _canvas(self) -> tuple[int, int]:
        """The `(height, width)` this session generates at."""
        return clip_plan.canvas_for_choice(self._aspect)

    def _current_clip(self) -> ClipEntry | None:
        """The clip on (or headed for) the output tracks, if any."""
        return self._playing or self._play_request

    def _snapshot(self) -> StateUpdate:
        """Everything a client can observe, in one message.

        The single source of the snapshot: `state_update` broadcasts it, a
        joining client is greeted with it, and `get_state` answers with it.
        Built once here so those three can never disagree.
        """
        height, width = self._canvas()
        current = self._current_clip()
        return StateUpdate(
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            clip_seconds_min=clip_plan.MIN_SECONDS_PUBLISHED,
            clip_seconds_max=clip_plan.MAX_SECONDS_PUBLISHED,
            seed=self._seed,
            autoplay=self._autoplay,
            aspect=self._aspect,
            width=width,
            height=height,
            playing=current is not None,
            playing_clip_id=current.clip_id if current is not None else None,
            queued=len(self._queue),
            queue_capacity=self._queue.capacity,
            clips_played=self._clips_played,
            seconds_sent=round(self._seconds_sent, 2),
            valid_commands=session_rules.valid_commands(
                playing=current is not None,
                queued=len(self._queue),
                ready=self._queue.ready_count(),
                capacity=self._queue.capacity,
            ),
        )

    async def _send_state_update(self) -> None:
        """Broadcast the snapshot to every connected client."""
        await self.send(self._snapshot())

    async def _send_queue_update(self) -> None:
        """Broadcast the queue's contents to every connected client."""
        await self.send(QueueUpdate(clips=self._queue.snapshot()))

    async def _refuse(self, command: str, reason: str) -> None:
        """Reject a command: tell every client, and leave its reply bodyless.

        A handler returns only the message its annotation names, and reports
        a failure by broadcasting `command_error` and returning without a
        value. The runtime answers that with a correlated bodyless
        acknowledgement, so an awaiting client resolves rather than hanging —
        and unlike a raised runtime ``CommandError``, whose failure frame is
        withheld from v0 clients, the broadcast reaches every SDK generation.

        Logged as well, so refusals are visible server-side and not only in the
        client's message.
        """
        logger.info("command refused", command=command, reason=reason)
        await self.send(CommandError(command=command, reason=reason))

    # ------------------------------------------------------------ lifecycle

    @session_started
    async def on_session_started(self) -> None:
        """Clear the queue and every condition so a new session inherits nothing."""
        self._reset_session_state()

    @session_ended
    async def on_session_ended(self) -> None:
        """Drop the session's work; the only hook guaranteed to fire on every path."""
        self._stop_playout = True
        self._play_request = None
        if self._build is not None:
            _entry, job, _submitted = self._build
            job.cancelled = True
        self._queue.clear()

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        """Greet the joining client with the full state and the queue.

        Addressed rather than broadcast: the clients already watching have
        this, and a late joiner needs it without replaying every command.
        """
        await client.send(self._snapshot())
        await client.send(QueueUpdate(clips=self._queue.snapshot()))

    # ------------------------------------------------------------- commands

    @event(
        name="enqueue",
        description=(
            "Queue one clip generation. The prompt is what the clip will show; "
            "the metadata is an opaque string echoed back on every message that "
            "references the clip, for frontends to carry their own tracking "
            "data. The clip's canvas is the session's; its length is the "
            "`seconds` passed here (snapped to what the model can produce) or "
            "the session default, and its seed is the one passed here or the "
            "session's advancing default. Builds run through the queue in "
            "order; watch `queue_update` for the clip turning ready. Replies "
            "`clip_queued` with the clip's UUID and emits `queue_update` and "
            "`state_update`, or `command_error` when the queue is full or the "
            "prompt is empty."
        ),
    )
    async def enqueue(
        self,
        prompt: str = InputField(
            default="",
            max_length=MAX_PROMPT_CHARS,
            moderate=True,
            description=(
                "What the clip should show, up to 800 characters. Fixed once "
                "enqueued; a different scene is a new `enqueue`."
            ),
        ),
        metadata: str = InputField(
            default="",
            max_length=MAX_METADATA_CHARS,
            moderate=True,
            description=(
                "Free-form string stored with the clip and echoed back on every "
                "message that references it. The model never reads it; use it "
                "to correlate clips with your own records — who asked for it, "
                "which group it belongs to, display text."
            ),
        ),
        seed: int | None = InputField(
            default=None,
            ge=0,
            description=(
                "Seed for this clip. Omitted or null, the session's default is "
                "used and advances by one; passing a seed leaves the default "
                "untouched, so explicit and automatic seeding do not interfere."
            ),
        ),
        seconds: float | None = InputField(
            default=None,
            ge=clip_plan.MIN_SECONDS_PUBLISHED,
            le=clip_plan.MAX_SECONDS_PUBLISHED,
            description=(
                f"Length of this clip in seconds, between {_CLIP_RANGE}, "
                "snapped to the nearest length the model can produce; the "
                "clip's structure reports the effective value. Omitted or "
                "null, the session default applies. A length the deployment "
                "has not built before pays a one-off compile cost on its "
                "first build."
            ),
        ),
    ) -> ClipQueued:
        """Append one generation request to the queue."""
        prompt = prompt.strip()
        if not prompt:
            await self._refuse("enqueue", "The prompt is empty; a clip needs one.")
            return None
        if self._queue.full:
            await self._refuse(
                "enqueue",
                f"The queue is full ({self._queue.capacity} clips); play or `reset` first.",
            )
            return None
        if not isinstance(seed, int):
            # None on the wire; the InputField sentinel when called directly.
            seed = self._seed
            self._seed += 1
        frames = (
            clip_plan.frames_for_seconds(float(seconds))
            if isinstance(seconds, (int, float))
            else self._clip_frames
        )
        entry = self._queue.enqueue(
            prompt=prompt,
            metadata=metadata,
            frames=frames,
            seed=seed,
        )
        await self._send_queue_update()
        await self._send_state_update()
        return ClipQueued(clip=entry.snapshot())

    @event(
        name="play",
        description=(
            "Play one built clip from the queue. Blank `clip_id` plays the "
            "oldest ready clip; a UUID plays that specific clip. Playing "
            "consumes the entry: it leaves the queue, `clip_started` marks its "
            "first frames, and when it ends the stream holds on black until "
            "the next `play`. Emits `queue_update` and `state_update`, or "
            "`command_error` when a clip is already playing, the id is "
            "unknown, or the clip is not ready yet."
        ),
    )
    async def play(
        self,
        clip_id: str = InputField(
            default="",
            description=(
                "UUID of the clip to play, from `clip_queued` or `queue_update`. "
                "Blank plays the oldest clip that is ready."
            ),
        ),
    ) -> None:
        """Take one ready clip off the queue and hand it to the playout loop."""
        if self._current_clip() is not None:
            await self._refuse("play", "A clip is already playing; send `stop` first.")
            return
        if clip_id:
            entry = self._queue.get(clip_id)
            if entry is None:
                await self._refuse("play", f"No queued clip has id {clip_id!r}.")
                return
            if not entry.ready:
                await self._refuse(
                    "play",
                    "That clip is still generating; wait for `queue_update` to report it ready.",
                )
                return
        else:
            entry = self._queue.next_ready()
            if entry is None:
                await self._refuse(
                    "play",
                    "No clip is ready to play; `enqueue` one and wait for `queue_update`.",
                )
                return
        self._queue.remove(entry)
        self._play_request = entry
        await self._send_queue_update()
        await self._send_state_update()

    @event(
        name="pop",
        description=(
            "Remove one clip from the queue by its UUID, freeing its slot for "
            "something else. Works on built and still-generating clips alike; "
            "a build already running for it is discarded when it completes. "
            "The clip that is playing is not in the queue — `stop` is the "
            "command that cuts it. Emits `clip_popped`, `queue_update` and "
            "`state_update`, or `command_error` when no queued clip has that "
            "id."
        ),
    )
    async def pop(
        self,
        clip_id: str = InputField(
            default="",
            description="UUID of the queued clip to remove, from `clip_queued` or `queue_update`.",
        ),
    ) -> ClipPopped:
        """Take one clip out of the queue and free its slot."""
        entry = self._queue.get(clip_id) if clip_id else None
        if entry is None:
            reason = (
                f"No queued clip has id {clip_id!r}."
                if clip_id
                else "Pass the `clip_id` of the queued clip to remove."
            )
            await self._refuse("pop", reason)
            return None
        self._queue.remove(entry)
        if self._build is not None and self._build[0] is entry:
            _entry, job, _submitted = self._build
            job.cancelled = True
        await self._send_queue_update()
        await self._send_state_update()
        return ClipPopped(clip=entry.snapshot())

    @event(
        name="stop",
        description=(
            "Cut the clip that is playing. Whatever is queued on the output "
            "tracks is dropped, the picture goes to black within a fraction of "
            "a second, and the session is back where a finished clip leaves it "
            "— the queue is untouched and the next `play` starts clean. With "
            "autoplay on this acts as a skip: the next ready clip starts on "
            "its own, so send `set_autoplay` off first to hold the stream. "
            "Emits `clip_stopped` and `state_update`, or `command_error` when "
            "no clip is playing."
        ),
    )
    async def stop(self) -> None:
        """Ask the playout loop to cut the current clip."""
        if self._current_clip() is None:
            await self._refuse("stop", "No clip is playing.")
            return
        self._stop_playout = True

    @event(
        name="get_queue",
        description=(
            "Return the queue's contents: every clip's UUID, prompt, metadata, "
            "length in frames and seconds, seed, and whether it is built and "
            "ready to play. The same payload the model broadcasts as "
            "`queue_update`. Valid at any time."
        ),
    )
    async def get_queue(self) -> QueueUpdate:
        """Answer with the same payload `queue_update` broadcasts."""
        return QueueUpdate(clips=self._queue.snapshot())

    @event(
        name="set_clip_seconds",
        description=(
            "Set the default length for enqueues that carry no `seconds` of "
            "their own. The value is snapped to the nearest length the model "
            "can produce, so read the effective one back from "
            "`clip_length_accepted`. Clips already in the queue keep the "
            "length they were enqueued with. Longer clips carry a scene "
            "further; shorter ones build faster. Emits `clip_length_accepted` "
            "and `state_update`."
        ),
    )
    async def set_clip_seconds(
        self,
        seconds: float = InputField(
            default=clip_plan.MAX_SECONDS_PUBLISHED,
            ge=clip_plan.MIN_SECONDS_PUBLISHED,
            le=clip_plan.MAX_SECONDS_PUBLISHED,
            description=(
                f"Clip length in seconds, between {_CLIP_RANGE}. Snapped to the "
                "nearest length the model can produce, so the value that takes "
                "effect can differ slightly; `state_update.clip_seconds` always "
                "carries the one in force."
            ),
        ),
    ) -> ClipLengthAccepted:
        """Set the length newly enqueued clips snapshot."""
        self._clip_frames = clip_plan.frames_for_seconds(float(seconds))
        await self._send_state_update()
        return ClipLengthAccepted(
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            frames=self._clip_frames,
        )

    @event(
        name="set_seed",
        description=(
            "Set the default seed — the one an `enqueue` without a seed of its "
            "own uses, advancing it by one, so re-enqueuing the same prompts "
            "in the same order reproduces the same clips. Clips already in "
            "the queue keep the seed they were enqueued with. Emits "
            "`seed_accepted` and `state_update`."
        ),
    )
    async def set_seed(
        self,
        seed: int = InputField(
            default=1000,
            ge=0,
            description=(
                "Default seed for enqueues that carry none. Reproduction is "
                "close rather than exact: the deployment runs fused kernels "
                "that can reorder arithmetic."
            ),
        ),
    ) -> SeedAccepted:
        """Set the default seed for enqueues that carry none."""
        self._seed = int(seed)
        await self._send_state_update()
        return SeedAccepted(seed=self._seed)

    @event(
        name="set_autoplay",
        description=(
            "Turn autoplay on or off. On, the oldest ready clip starts on its "
            "own whenever nothing is playing — right after a clip finishes, or "
            "the moment a build completes while the stream is idle — so a "
            "steadily fed queue plays through without a `play` per clip. Off "
            "(the default), the stream holds on black until an explicit "
            "`play`. Takes effect immediately and lasts for the session. Emits "
            "`autoplay_accepted` and `state_update`."
        ),
    )
    async def set_autoplay(
        self,
        enabled: bool = InputField(
            default=False,
            description=(
                "True plays ready clips on their own, oldest first; false "
                "holds the stream after each clip until `play`."
            ),
        ),
    ) -> AutoplayAccepted:
        """Set whether ready clips start without an explicit `play`."""
        self._autoplay = bool(enabled)
        await self._send_state_update()
        return AutoplayAccepted(enabled=self._autoplay)

    @event(
        name="set_canvas",
        description=(
            "Choose the aspect ratio of `main_video`. The video track keeps "
            "one size and queued clips are built at it, so this is only valid "
            "while the queue is empty and nothing is playing. Emits "
            "`canvas_accepted`, carrying the exact pixel size, and "
            "`state_update`, or `command_error` while clips are queued or "
            "playing, or when the ratio is not one this model offers."
        ),
    )
    async def set_canvas(
        self,
        aspect: str = InputField(
            default="16:9",
            choices=list(clip_plan.ASPECT_CHOICES),
            description=(
                "Aspect ratio of `main_video`. `canvas_accepted` and "
                "`state_update` report the width and height in pixels it "
                "resolves to."
            ),
        ),
    ) -> CanvasAccepted:
        """Set the session's canvas; refused while any clip depends on the old one."""
        if self._current_clip() is not None or len(self._queue) > 0:
            await self._refuse(
                "set_canvas",
                "The canvas is fixed while clips are queued or playing; "
                "`reset` or play the queue out first.",
            )
            return None
        try:
            height, width = clip_plan.canvas_for_choice(aspect)
        except ValueError as error:
            await self._refuse("set_canvas", str(error))
            return None
        self._aspect = aspect
        await self._send_state_update()
        return CanvasAccepted(aspect=aspect, width=width, height=height)

    @event(
        name="reset",
        description=(
            "Return every condition to its default, drop every queued clip, "
            "and clear the output tracks. A clip that is playing is cut, with "
            "a `clip_stopped` to mark it. Valid at any time. Replies "
            "`session_reset` and emits `queue_update` and `state_update`."
        ),
    )
    async def reset(self) -> SessionReset:
        """Clear the session back to its defaults."""
        current = self._current_clip()
        was_playing = current is not None
        cleared = self._queue.clear()
        # A pending play request is a clip already off the queue; dropping it
        # here counts it with the cleared ones. A clip mid-play is cut by the
        # playout loop, which owns its `clip_stopped`.
        if self._play_request is not None:
            self._play_request = None
            cleared += 1
        if self._playing is not None:
            self._stop_playout = True
        if self._build is not None:
            _entry, job, _submitted = self._build
            job.cancelled = True
        self._clip_frames = self.config.clip_frames
        self._seed = self.config.seed
        self._aspect = self.config.aspect
        self._autoplay = False
        self.output.flush()
        await self._send_queue_update()
        await self._send_state_update()
        return SessionReset(cleared_clips=cleared, was_playing=was_playing)

    @event(
        name="get_state",
        description=(
            "Return a snapshot of everything the session exposes except the "
            "queue's contents (`get_queue` carries those): the conditions in "
            "force, what is playing, progress counters, and the commands that "
            "are valid right now. The same payload the model broadcasts as "
            "`state_update`. Valid at any time."
        ),
    )
    async def get_state(self) -> StateUpdate:
        """Answer with the same snapshot `state_update` broadcasts."""
        return self._snapshot()

    # ------------------------------------------------------------- run loop

    async def run(self) -> None:
        """The model's control loop: park without an audience, serve with one.

        Nothing here may raise: an exception out of ``run()`` is an
        unrecoverable crash of the whole model loop, not the end of one
        session, so ``_serve`` owns its own failure reporting.
        """
        while True:
            await self.connected.wait()
            await self._serve()

    async def _serve(self) -> None:
        """Pump builds and play armed clips while an audience is connected.

        Generation is gated on having an audience: with nobody connected no new
        build is submitted, and the loop parks back in ``run()``. A build
        already on the worker finishes into the queue either way, because a
        clip cannot be cancelled mid-build.
        """
        while self.connected.is_set():
            try:
                await self._pump_builds()
                if (
                    self._autoplay
                    and self._play_request is None
                    and self._playing is None
                ):
                    # Autoplay is a standing `play`: whenever nothing is on the
                    # tracks and a built clip waits, the oldest one starts.
                    ready = self._queue.next_ready()
                    if ready is not None:
                        self._queue.remove(ready)
                        self._play_request = ready
                        await self._send_queue_update()
                        await self._send_state_update()
                entry = self._play_request
                if entry is not None:
                    self._play_request = None
                    await self._play_clip(entry)
                else:
                    await asyncio.sleep(POLL_SECONDS)
            except Exception:  # noqa: BLE001 — the model loop must survive anything
                logger.exception("error in the fasth3 serve loop")
                await asyncio.sleep(POLL_SECONDS)

    async def _pump_builds(self) -> None:
        """Apply a finished build and keep the worker fed, without blocking.

        Called from the idle loop and from every playout slice, so clips keep
        building while another one streams. A finished build whose entry is no
        longer in the queue — a `reset` or a session end removed it — is
        discarded silently; the queue owns what exists.
        """
        if self._build is not None:
            entry, job, submitted = self._build
            if not job.done.is_set():
                return
            self._build = None
            entry.building = False
            if job.cancelled or entry not in self._queue:
                pass
            elif job.error is not None:
                self._queue.remove(entry)
                await self.send(ClipFailed(clip=entry.snapshot(), reason=str(job.error)))
                await self._send_queue_update()
                await self._send_state_update()
            else:
                entry.video, entry.audio = job.result
                logger.info(
                    f"clip ready: {entry.clip_id} ({entry.frames}f) "
                    f"{time.monotonic() - submitted:.2f}s after submit, "
                    f"{len(self._queue)} queued"
                )
                await self._send_queue_update()
                await self._send_state_update()
        if self._build is None:
            pending = self._queue.next_to_build()
            if pending is not None:
                height, width = self._canvas()
                pending.building = True
                logger.info(
                    f"clip build submitted: {pending.clip_id} ({pending.frames}f), "
                    f"{len(self._queue)} queued"
                )
                self._build = (
                    pending,
                    self.backend.submit(
                        frames=pending.frames,
                        prompt=pending.prompt,
                        seed=pending.seed,
                        height=height,
                        width=width,
                    ),
                    time.monotonic(),
                )

    async def _play_clip(self, entry: ClipEntry) -> None:
        """Stream one built clip, then flush to black and report how it ended."""
        self._playing = entry
        outcome = "stopped"
        try:
            if not self._stop_playout:
                await self.send(ClipStarted(clip=entry.snapshot()))
                await self._send_state_update()
                outcome = await self._emit_clip(entry)
        finally:
            # Black between clips, always: whatever the transport still holds
            # of this clip is dropped, so the next `play` starts clean.
            self.output.flush()
            self._playing = None
            self._stop_playout = False
        self._clips_played += 1
        if outcome == "finished":
            await self.send(
                ClipFinished(clip=entry.snapshot(), seconds_sent=round(self._seconds_sent, 2))
            )
        elif outcome == "stopped":
            await self.send(
                ClipStopped(clip=entry.snapshot(), seconds_sent=round(self._seconds_sent, 2))
            )
        # A lost audience ends the playout with nobody to tell; the state
        # update below is a harmless no-op in that case.
        await self._send_state_update()

    # -------------------------------------------------------------- emitter

    async def _emit_clip(self, entry: ClipEntry) -> str:
        """Emit one clip as paced slices on a 24 fps metronome.

        - Paced by FRAMES, not slices: a clip's tail slice is short, and
          charging it a full slot would open a hole in the cadence.
        - Never burst to catch up: if the transport held a slice back,
          re-anchor instead. A catch-up burst only overflows the queue.
        - Emits omit ``compute_time``, so every slice is tagged at the pinned
          24 fps — the rate the audio is already sample-clocked against.
        - Builds keep moving: every slice pumps the worker, so a clip
          generating behind this one is ready sooner.

        Returns:
            ``"finished"`` when the whole clip went out, ``"stopped"`` when
            `stop` or `reset` cut it, ``"gone"`` when the audience left.
        """
        import numpy as np

        frames_list, samples = entry.video, entry.audio
        samples_per_frame = OUTPUT_SAMPLE_RATE / FRAME_RATE
        total = len(frames_list)
        clock_start: float | None = None
        frames_paced = 0
        for lo in range(0, total, EMIT_FRAMES):
            await self._pump_builds()
            if self._stop_playout:
                return "stopped"
            if not self.connected.is_set():
                return "gone"
            hi = min(lo + EMIT_FRAMES, total)
            alo = round(lo * samples_per_frame)
            ahi = round(hi * samples_per_frame)

            now = asyncio.get_running_loop().time()
            if clock_start is None:
                clock_start = now
            content_pos = frames_paced / FRAME_RATE
            clock_start = max(clock_start, now - content_pos)
            delay = clock_start + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            frames_paced += hi - lo
            self._frames_sent += hi - lo
            self._seconds_sent = self._frames_sent / FRAME_RATE
            video = np.ascontiguousarray(np.stack(frames_list[lo:hi]))
            await self.emit(FastH3Output(main_video=video, main_audio=samples[:, alo:ahi]))
        return "finished"
