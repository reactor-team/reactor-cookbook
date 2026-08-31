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
MAX_STYLE_CHARS = 400
DEFAULT_STYLE_PROMPT = ""


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
            "Content prompt the next unassigned clip will use: the head of the "
            "queue, or the repeating fallback when the queue is empty."
        )
    )
    current_prompt: str | None = MessageField(
        description="Prompt that produced the clip currently visible, or null before it starts."
    )
    current_style_prompt: str | None = MessageField(
        description=(
            "Style instruction that produced the clip currently visible, or null "
            "before it starts."
        )
    )
    current_prompt_source: str | None = MessageField(
        description=(
            "Origin of the clip currently visible: manual, bilibili, or ai; null "
            "before playback starts."
        )
    )
    current_prompt_viewer_name: str | None = MessageField(
        description=(
            "Bilibili viewer who requested the visible clip, or null for other sources."
        )
    )
    current_prompt_original_request: str | None = MessageField(
        description=(
            "Original Bilibili creative direction behind the visible clip, or null "
            "for other sources."
        )
    )
    next_prompt: str | None = MessageField(
        description=(
            "Prompt assigned to the lookahead clip being built or waiting to play, "
            "or null before a channel starts."
        )
    )
    next_style_prompt: str | None = MessageField(
        description=(
            "Style assigned to the lookahead clip being built or waiting to play, "
            "or null before a channel starts."
        )
    )
    style_prompt: str = MessageField(
        description=(
            "Style appended to future clip prompts. Empty text leaves future prompts "
            "without a shared style instruction."
        )
    )
    queued_prompts: list[str] = MessageField(
        description="Prompts waiting to be assigned, in generation order."
    )
    prompt_queue_depth: int = MessageField(
        description="Number of prompts still waiting to be assigned to clips."
    )
    auto_story_enabled: bool = MessageField(
        description=(
            "Whether the story writer keeps the prompt queue supplied after the "
            "channel has run for its configured startup delay."
        )
    )
    auto_story_generating: bool = MessageField(
        description="Whether the story writer is currently drafting the next scene."
    )
    auto_story_queue_target: int = MessageField(
        description="Number of waiting prompts the story writer maintains."
    )
    live_chat_enabled: bool = MessageField(
        description="Whether this session listens for prompt commands from a live room."
    )
    live_chat_connected: bool = MessageField(
        description="Whether the live-room WebSocket is receiving messages."
    )
    live_chat_room_id: int | None = MessageField(
        description="Configured Bilibili room id, or null when live chat is disabled."
    )
    live_prompt_pending: int = MessageField(
        description="Viewer requests waiting to be rewritten into complete prompts."
    )
    live_prompt_queue_depth: int = MessageField(
        description=(
            "Bilibili requests waiting for rewrite or assignment to a future clip."
        )
    )
    live_prompt_queue_limit: int = MessageField(
        description="Maximum Bilibili backlog accepted by this session."
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
    continuity: bool = MessageField(
        description=(
            "Whether the channel stitches clips into one continuous stream. When "
            "true, each clip after the first is anchored on the previous clip's "
            "last frame and crossfaded onto it, so scene boundaries are soft "
            "transitions rather than hard cuts. A deployment setting, fixed for "
            "the session."
        )
    )
    seed: int = MessageField(
        description="Seed the channel started from; each clip advances it by one."
    )
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
            "Index of the next unassigned clip. The `prompt` and `style_prompt` "
            "conditions in this snapshot are captured there."
        )
    )
    prompt_effective_in_seconds: float = MessageField(
        description=(
            "Seconds of assigned video before the next unassigned clip starts. "
            "Zero while the channel is idle."
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
    queue_position: int = MessageField(
        description=(
            "Zero-based position in the waiting FIFO when accepted. Zero means "
            "this is the next queued prompt assigned after current lookahead."
        )
    )
    queue_depth: int = MessageField(
        description="Number of prompts waiting after this request was accepted."
    )


class AutoStoryAccepted(ModelMessage):
    """Emitted when `set_auto_story` is accepted."""

    enabled: bool = MessageField(
        description="Whether automatic scene writing is enabled for this session."
    )


class AutoPromptQueued(ModelMessage):
    """Emitted when the story writer adds one scene to the prompt queue."""

    prompt: str = MessageField(description="Complete FastH3 prompt for the new scene.")
    queue_depth: int = MessageField(
        description="Number of prompts waiting after the generated scene was added."
    )
    based_on_scenes: int = MessageField(
        description="Number of recent scenes supplied to the story writer."
    )
    fallback_used: bool = MessageField(
        description="Whether a deterministic scene replaced an unusable LLM response."
    )


class LiveChatStatus(ModelMessage):
    """Emitted when the live-room connection becomes available or unavailable."""

    connected: bool = MessageField(
        description="Whether the live-room WebSocket is receiving messages."
    )
    room_id: int = MessageField(description="Bilibili room being monitored.")
    detail: str | None = MessageField(
        description="Connection failure detail, or null during normal operation."
    )


class LivePromptReceived(ModelMessage):
    """Emitted when a viewer prompt command enters the rewrite FIFO."""

    viewer_name: str = MessageField(description="Display name attached to the comment.")
    request: str = MessageField(
        description="Creative direction after the command prefix."
    )
    pending_requests: int = MessageField(
        description="Viewer requests waiting for GPT rewrite, including this one."
    )


class LivePromptQueued(ModelMessage):
    """Emitted when a viewer request becomes a complete FastH3 prompt."""

    viewer_name: str = MessageField(description="Display name attached to the comment.")
    request: str = MessageField(
        description="Original creative direction from the viewer."
    )
    prompt: str = MessageField(description="Complete three-field FastH3 prompt queued.")
    queue_depth: int = MessageField(
        description="Number of complete prompts waiting after this prompt was added."
    )
    generation_seconds: float = MessageField(
        description="Seconds GPT took to rewrite the viewer request."
    )
    effective_clip_index: int = MessageField(
        description="Clip where this viewer direction will first appear."
    )
    effective_in_seconds: float = MessageField(
        description="Seconds of assigned video before this viewer direction starts."
    )
    fallback_used: bool = MessageField(
        description="Whether a minimal local rewrite replaced an unavailable GPT result."
    )


class StyleAccepted(ModelMessage):
    """Emitted when `set_style` is accepted."""

    style_prompt: str = MessageField(
        description=(
            "Style now applied to future clips. Empty text means no shared style "
            "instruction."
        )
    )
    effective_clip_index: int = MessageField(
        description="First clip that can use this style."
    )
    effective_in_seconds: float = MessageField(
        description=(
            "Seconds of assigned video before this style can appear. Zero while "
            "the channel is idle."
        )
    )


class ClipLengthAccepted(ModelMessage):
    """Emitted when `set_clip_seconds` is accepted.

    The requested length is snapped to the nearest length the model can produce,
    so the value here may differ slightly from the one sent.
    """

    clip_seconds: float = MessageField(
        description="Clip length now in effect, in seconds."
    )
    frames: int = MessageField(
        description="Frames each clip will carry on `main_video`."
    )


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
    clip_seconds: float = MessageField(
        description="Length of each steady-state clip, in seconds."
    )
    first_clip_seconds: float = MessageField(
        description=(
            "Length of the opening clip. The channel can open with a shorter "
            "clip so video starts sooner; it equals `clip_seconds` when it does not."
        )
    )


class ClipStarted(ModelMessage):
    """Emitted as each clip begins streaming on the output tracks.

    By default every clip is an independent piece of video and audio, so the
    picture and the sound cut at this boundary rather than continuing from the
    last frame. When the channel runs in continuity mode (`state_update.continuity`
    is true) the boundary is instead a short crossfade onto the previous clip, so
    this marks where a new prompt's content begins rather than a hard cut.
    """

    clip_index: int = MessageField(
        description="Zero-based index of the clip now streaming."
    )
    clip_seconds: float = MessageField(description="Length of this clip, in seconds.")
    prompt: str = MessageField(description="Prompt this clip was built from.")
    style_prompt: str = MessageField(
        description=(
            "Style instruction this clip was built with. Empty text means no shared "
            "style instruction."
        )
    )
    source: str = MessageField(
        description="Origin of this clip: manual, bilibili, or ai."
    )
    viewer_name: str | None = MessageField(
        description="Bilibili viewer who requested this clip, or null otherwise."
    )
    original_request: str | None = MessageField(
        description=(
            "Original Bilibili creative direction behind this clip, or null otherwise."
        )
    )


class ClipComplete(ModelMessage):
    """Emitted when a clip has been fully sent on the output tracks."""

    clip_index: int = MessageField(
        description="Zero-based index of the clip just finished."
    )
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
    seconds_sent: float = MessageField(
        description="Seconds streamed before it stopped."
    )


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
