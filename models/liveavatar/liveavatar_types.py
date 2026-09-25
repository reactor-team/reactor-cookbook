"""Client-facing avatar inputs, audiovisual tracks and session snapshots."""

from reactor_runtime import Audio, InputState, MessageField, ModelMessage, Output, Video


class AvatarAudio(Audio):
    sample_rate = 48000


class LiveAvatarOutput(Output):
    main_video: Video
    main_audio: AvatarAudio


class LiveAvatarState(InputState):
    _image_name: str | None = None
    _audio_name: str | None = None
    _pose_name: str | None = None
    _prompt: str = ""
    _negative_prompt: str = ""
    _seed: int = 420
    _running: bool = False
    _chunks: int = 0
    _frames: int = 0
    _max_chunks: int = 10000
    _error: str | None = None


class StateUpdate(ModelMessage):
    """Emitted on connection, accepted changes, generated clips and automatic take completion or failure."""

    image_name: str | None = MessageField(
        description="Filename selected by `set_avatar_image` for the next `start`, or null until selected. Retained by `stop` and cleared by `reset`."
    )
    audio_name: str | None = MessageField(
        description="Speech filename selected by `set_audio` for the next `start`, or null until selected. Retained by `stop` and cleared by `reset`."
    )
    pose_name: str | None = MessageField(
        description="Prepared pose filename selected for the next `start`, or null for audio-driven motion. Cleared by `set_pose_video` with null or by `reset`; retained by `stop`."
    )
    prompt: str | None = MessageField(
        description="Selected scene and performance description, used from the next `start`; null means empty text. Retained through `stop` and automatic completion, and cleared by `reset`."
    )
    negative_prompt: str | None = MessageField(
        description="Stored compatibility text, or null for the model's default text. This serving profile applies no negative conditioning. Retained by `stop` and cleared by `reset`."
    )
    seed: int = MessageField(
        description="Selected sampling seed from 0 through 2147483647, read at `start`. Retained by `stop` and restored to 420 by `reset`."
    )
    ready: bool = MessageField(
        description="True when an image and speech audio are selected. Describes input readiness even during a take; enable `start` only when this is true and `running` is false. `reset` clears readiness."
    )
    running: bool = MessageField(
        description="True from accepted `start` until automatic completion, failure, `stop` or `reset`. While true, input and generation-option changes are rejected."
    )
    completed_chunks: int = MessageField(
        description="Number of clips generated in the current or most recent take, starting at zero. Updated with each `chunk_complete`, retained by `stop`, and cleared by `start` or `reset`."
    )
    frames: int = MessageField(
        description="Cumulative generated video frames in the current or most recent take, played at 25 FPS. Updated with each clip, retained by `stop`, and cleared by `start` or `reset`. Client playback can lag this count."
    )
    max_chunks: int = MessageField(
        description="Selected per-take clip limit from 1 through 10000, read at `start`; audio duration can finish a take earlier. Retained by `stop`; `reset` restores 10000."
    )
    error: str | None = MessageField(
        description="Most recent generation failure in this session, or null when clear. Set with a failed `generation_ended`, retained by `stop`, and cleared by `start` or `reset`. Rejected input commands leave this value unchanged."
    )

    @classmethod
    def from_state(cls, state: LiveAvatarState) -> "StateUpdate":
        return cls(
            image_name=state._image_name,
            audio_name=state._audio_name,
            pose_name=state._pose_name,
            prompt=state._prompt or None,
            negative_prompt=state._negative_prompt or None,
            seed=state._seed,
            ready=bool(state._image_name and state._audio_name),
            running=state._running,
            completed_chunks=state._chunks,
            frames=state._frames,
            max_chunks=state._max_chunks,
            error=state._error,
        )


class InputAccepted(ModelMessage):
    """Emitted as the reply when an uploaded input, prompt, or generation options are accepted."""

    field: str = MessageField(
        description="Condition selected by the successful command: `avatar_image`, `audio`, `pose_video`, `prompt`, or `generation_options`. The accompanying `state_update` contains the complete selection; generation waits for `start`."
    )


class TakeChanged(ModelMessage):
    """Emitted as the command-correlated reply when `start`, `stop` or `reset` succeeds."""

    action: str = MessageField(
        description="Accepted command wire name: `start`, `stop`, or `reset`. An accompanying `state_update` describes the resulting inputs and progress; clients should not assume the reply arrives before that snapshot."
    )


class ChunkComplete(ModelMessage):
    """Emitted once per generated clip for `main_video` and `main_audio`, alongside `state_update`."""

    chunk: int = MessageField(
        description="One-based generated clip number within the take; the first clip after each `start` is 1. Reports generation progress independently of client playback."
    )
    frames: int = MessageField(
        description="Video frames in this clip: 45 for clip 1, then 48 per clip, played at 25 FPS. This count contributes to the cumulative `state_update.frames`."
    )


class GenerationEnded(ModelMessage):
    """Emitted on automatic take completion or generation failure, alongside `state_update`."""

    reason: str = MessageField(
        description="`complete` when the audio or selected clip limit ends the take; otherwise the generation error text also reported in `state_update.error`. Explicit `stop` and `reset` are acknowledged through `take_changed`."
    )
