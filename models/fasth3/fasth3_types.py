"""Client-facing types for the FastH3 Reactor model.

Everything a client can see lives here: the outbound video and audio tracks,
the `ClipInfo` structure every clip-referencing message embeds, and the typed
messages the model sends. ``fasth3.py`` imports these; a frontend developer
reads this file to learn the whole API without opening the inference code.

The conditions behind the `set_*` commands are not here. A ``ReactorModel``
owns its session state itself, so they are plain attributes on ``FastH3`` reset
in ``_reset_session_state``; their client-facing text lives on each handler's
own ``InputField`` declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

from reactor_runtime import (
    Audio,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

MAX_PROMPT_CHARS = 800
MAX_METADATA_CHARS = 2000


class FastH3Output(Output):
    """The generated video and its synchronized audio, streamed per clip."""

    main_video: Video
    main_audio: Audio


@dataclass(frozen=True)
class ClipInfo:
    """One queued generation, as every clip-referencing message reports it.

    Whole and self-contained on purpose: `clip_queued`, `queue_update`,
    `clip_started`, `clip_finished`, `clip_stopped` and `clip_failed` all carry
    this same structure, so a client never has to join a clip id against an
    earlier message to know what a clip is.

    ``clip_id`` is the UUID the session assigned at `enqueue`; every later
    reference to the clip uses it. ``prompt`` and ``metadata`` are exactly what
    the client enqueued — the metadata is an opaque string the model never
    reads, for frontends to carry their own tracking data. ``frames`` and
    ``seconds`` are the clip's length in both units, fixed when it was
    enqueued. ``seed`` is the value this clip generates from. ``ready`` is
    whether the clip is built and can be played.
    """

    clip_id: str
    prompt: str
    metadata: str
    frames: int
    seconds: float
    seed: int
    ready: bool


class StateUpdate(ModelMessage):
    """Emitted on connect and after every change to the session's state.

    One snapshot of everything observable except the queue's contents (those
    travel as `queue_update`), so a client can render its whole UI from this
    alone instead of accumulating the individual messages below.
    """

    clip_seconds: float = MessageField(
        description=(
            "Length a newly enqueued clip gets when `enqueue` carries no "
            "`seconds` of its own."
        )
    )
    clip_seconds_min: float = MessageField(
        description="Shortest clip length `set_clip_seconds` accepts."
    )
    clip_seconds_max: float = MessageField(
        description="Longest clip length `set_clip_seconds` accepts."
    )
    seed: int = MessageField(
        description=(
            "Seed the next enqueued clip will use when `enqueue` carries none; "
            "each such enqueue advances it by one."
        )
    )
    autoplay: bool = MessageField(
        description=(
            "Ready clips start on their own whenever nothing is playing. Off "
            "by default: playback waits for an explicit `play`."
        )
    )
    aspect: str = MessageField(description="Aspect ratio in effect, e.g. `16:9`.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")
    playing: bool = MessageField(description="A clip is streaming on the output tracks.")
    playing_clip_id: str | None = MessageField(
        description="UUID of the clip now playing, or null when the stream is idle."
    )
    queued: int = MessageField(
        description="Clips in the queue right now, built and still generating alike."
    )
    queue_capacity: int = MessageField(
        description="Most clips the queue holds; `enqueue` is refused beyond it."
    )
    clips_played: int = MessageField(
        description="Clips that finished playing or were stopped since the session began."
    )
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the session began."
    )
    valid_commands: list[str] = MessageField(
        description=(
            "Names of the commands the session would accept right now. Use this "
            "to enable or grey out controls instead of re-deriving the state "
            "machine client-side; any command not listed would be rejected."
        )
    )


class QueueUpdate(ModelMessage):
    """Emitted on connect and whenever the queue changes, and answers `get_queue`.

    The whole queue, oldest first, each entry a complete `ClipInfo`. A change
    is any of: a clip enqueued, a clip becoming ready, a clip leaving the queue
    to play, or the queue being cleared by `reset`.
    """

    clips: list[ClipInfo] = MessageField(
        description=(
            "Every clip in the queue, oldest first. Builds run through the "
            "queue in this order, so entries with `ready: true` always sit at "
            "the front."
        )
    )


class ClipQueued(ModelMessage):
    """Emitted when `enqueue` accepts a generation request."""

    clip: ClipInfo = MessageField(
        description=(
            "The queued clip, UUID included. `ready` is false here; watch "
            "`queue_update` for it turning true."
        )
    )


class ClipStarted(ModelMessage):
    """Emitted as a clip begins streaming on the output tracks."""

    clip: ClipInfo = MessageField(description="The clip now playing.")


class ClipFinished(ModelMessage):
    """Emitted when a clip has been fully sent on the output tracks.

    The stream then holds on black until the next `play`; nothing plays on its
    own.
    """

    clip: ClipInfo = MessageField(description="The clip that just finished.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the session began, this clip included."
    )


class ClipStopped(ModelMessage):
    """Emitted when `stop` cuts a playing clip.

    The rest of the clip is discarded — a stopped clip cannot be resumed — and
    the stream holds on black until the next `play`, exactly as after
    `clip_finished`.
    """

    clip: ClipInfo = MessageField(description="The clip that was cut.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the session began."
    )


class ClipFailed(ModelMessage):
    """Emitted when a clip's generation fails.

    The clip leaves the queue and the queue moves on; nothing else is
    affected.
    """

    clip: ClipInfo = MessageField(description="The clip whose build failed.")
    reason: str = MessageField(description="What went wrong.")


class ClipLengthAccepted(ModelMessage):
    """Emitted when `set_clip_seconds` is accepted.

    The requested length is snapped to the nearest length the model can produce,
    so the value here may differ slightly from the one sent.
    """

    clip_seconds: float = MessageField(description="Clip length now in effect, in seconds.")
    frames: int = MessageField(description="Frames each newly enqueued clip will carry.")


class SeedAccepted(ModelMessage):
    """Emitted when `set_seed` is accepted."""

    seed: int = MessageField(
        description="Seed the next enqueued clip will use when `enqueue` carries none."
    )


class AutoplayAccepted(ModelMessage):
    """Emitted when `set_autoplay` is accepted."""

    enabled: bool = MessageField(
        description="Whether ready clips now start on their own when nothing is playing."
    )


class CanvasAccepted(ModelMessage):
    """Emitted when `set_canvas` is accepted."""

    aspect: str = MessageField(description="Aspect ratio now in effect.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")


class SessionReset(ModelMessage):
    """Emitted when `reset` is accepted.

    Every condition is back to its default, the queue is empty, and the output
    stream is cleared.
    """

    cleared_clips: int = MessageField(
        description="Clips that were dropped from the queue, built and pending alike."
    )
    was_playing: bool = MessageField(
        description="A clip was playing and has been cut; a `clip_stopped` accompanies it."
    )


class CommandError(ModelMessage):
    """Emitted when a command is rejected. The command had no effect."""

    command: str = MessageField(description="Name of the command that was rejected.")
    reason: str = MessageField(description="Why it was rejected.")
