"""Client-facing types for the FastH3 Reactor model.

Everything a client can see lives here: the outbound video and audio tracks and
the typed messages the model sends. ``fasth3.py`` imports these; a frontend
developer reads this file to learn the whole API without opening the inference
code.

The conditions behind the `set_*` commands are not here. A ``ReactorModel`` owns
its session state itself, so they are plain attributes on ``FastH3`` reset in
``_reset_session_state``; their client-facing text lives on each handler's own
``InputField`` declaration.
"""

from __future__ import annotations

from reactor_runtime import (
    Audio,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

MAX_PROMPT_CHARS = 800


class FastH3Output(Output):
    """The generated video and its synchronized audio, streamed continuously."""

    main_video: Video
    main_audio: Audio


class StateUpdate(ModelMessage):
    """Emitted on connect and after every change to the session's state.

    One snapshot of everything observable, so a client can render its whole UI
    from this alone instead of accumulating the individual messages below.
    """

    prompt: str | None = MessageField(
        description=(
            "Prompt in effect for the next clip the model starts, or null when "
            "none is set."
        )
    )
    clip_seconds: float = MessageField(
        description="Length of each clip the channel produces, in seconds."
    )
    clip_seconds_min: float = MessageField(
        description="Shortest clip length `set_clip_seconds` accepts."
    )
    clip_seconds_max: float = MessageField(
        description="Longest clip length `set_clip_seconds` accepts."
    )
    seed: int = MessageField(description="Seed the channel started from; each clip advances it by one.")
    aspect: str = MessageField(description="Aspect ratio in effect, e.g. `16:9`.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")
    ready: bool = MessageField(description="A prompt is set, so `start` is valid.")
    running: bool = MessageField(description="The channel is live and streaming clips.")
    paused: bool = MessageField(
        description="The output stream is held; `resume` continues it instantly."
    )
    clip_index: int = MessageField(
        description="Zero-based index of the clip currently streaming, or -1 before the first."
    )
    clips_sent: int = MessageField(description="Clips fully streamed since `start`.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since `start`."
    )
    prompt_effective_clip_index: int = MessageField(
        description=(
            "Clip the prompt shown here will first appear on. The model always "
            "builds one clip ahead, so a prompt changed mid-channel lands two "
            "clips out, not on the next one."
        )
    )
    prompt_effective_in_seconds: float = MessageField(
        description=(
            "Seconds of already-built video still to play before the prompt "
            "shown here starts. Zero while the channel is idle."
        )
    )
    valid_commands: list[str] = MessageField(
        description=(
            "Names of the commands the session would accept right now. Use this "
            "to enable or grey out controls instead of re-deriving the state "
            "machine client-side; any command not listed would be rejected."
        )
    )


class PromptAccepted(ModelMessage):
    """Emitted when `set_prompt` is accepted."""

    prompt: str | None = MessageField(
        description="Prompt now in effect, or null when it was cleared."
    )
    effective_clip_index: int = MessageField(
        description="Clip this prompt will first appear on."
    )
    effective_in_seconds: float = MessageField(
        description=(
            "Seconds of already-built video still to play before this prompt "
            "starts. Zero while the channel is idle, so the next `start` uses it "
            "immediately."
        )
    )


class ClipLengthAccepted(ModelMessage):
    """Emitted when `set_clip_seconds` is accepted.

    The requested length is snapped to the nearest length the model can produce,
    so the value here may differ slightly from the one sent.
    """

    clip_seconds: float = MessageField(description="Clip length now in effect, in seconds.")
    frames: int = MessageField(description="Frames each clip will carry on `main_video`.")


class SeedAccepted(ModelMessage):
    """Emitted when `set_seed` is accepted."""

    seed: int = MessageField(description="Seed the next channel run starts from.")


class CanvasAccepted(ModelMessage):
    """Emitted when `set_canvas` is accepted."""

    aspect: str = MessageField(description="Aspect ratio now in effect.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")


class ChannelStarted(ModelMessage):
    """Emitted once when `start` is accepted, before any frame is sent.

    The first clip must be built before anything can stream, so expect several
    seconds of no video. Treat this as the cue to show progress, not to expect
    frames immediately.
    """

    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")
    clip_seconds: float = MessageField(description="Length of each steady-state clip, in seconds.")
    first_clip_seconds: float = MessageField(
        description=(
            "Length of the opening clip. The channel can open with a shorter "
            "clip so video starts sooner; it equals `clip_seconds` when it does not."
        )
    )


class ClipStarted(ModelMessage):
    """Emitted as each clip begins streaming on the output tracks.

    Every clip is an independent piece of video and audio, so the picture and
    the sound cut at this boundary rather than continuing from the last frame.
    """

    clip_index: int = MessageField(description="Zero-based index of the clip now streaming.")
    clip_seconds: float = MessageField(description="Length of this clip, in seconds.")
    prompt: str = MessageField(description="Prompt this clip was built from.")


class ClipComplete(ModelMessage):
    """Emitted when a clip has been fully sent on the output tracks."""

    clip_index: int = MessageField(description="Zero-based index of the clip just finished.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since `start`, this clip included."
    )


class ChannelPaused(ModelMessage):
    """Emitted when `pause` is accepted.

    The stream freezes on its current frame while the model keeps building
    ahead, so `resume` continues without any warm-up.
    """

    seconds_sent: float = MessageField(description="Seconds streamed before the pause.")


class ChannelResumed(ModelMessage):
    """Emitted when `resume` is accepted. The stream continues where it froze."""

    seconds_sent: float = MessageField(description="Seconds streamed so far.")


class ChannelStopped(ModelMessage):
    """Emitted when `stop` is accepted and the channel ends.

    Every condition is kept, so `start` immediately begins a fresh channel with
    the same setup. The clip already being built is discarded, so the model can
    take a few seconds to go idle.
    """

    seconds_sent: float = MessageField(description="Seconds streamed before the stop.")
    clips_sent: int = MessageField(description="Clips fully streamed before the stop.")


class ChannelFailed(ModelMessage):
    """Emitted when the channel ends early because something went wrong.

    The model then idles; adjust the conditions and `start` again.
    """

    reason: str = MessageField(description="What went wrong.")
    seconds_sent: float = MessageField(description="Seconds streamed before it stopped.")


class ChannelReset(ModelMessage):
    """Emitted when `reset` is accepted.

    Every condition is back to its default, the output stream is cleared, and
    the model is waiting for new conditions.
    """

    was_running: bool = MessageField(
        description="A channel was live and has been stopped, so no `channel_stopped` will follow it."
    )


class CommandError(ModelMessage):
    """Emitted when a command is rejected. The command had no effect."""

    command: str = MessageField(description="Name of the command that was rejected.")
    reason: str = MessageField(description="Why it was rejected.")
