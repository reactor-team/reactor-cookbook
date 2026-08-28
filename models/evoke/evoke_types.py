"""Define the public Reactor schema for EVOKE."""

from __future__ import annotations

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class EvokeOutput(Output):
    """Carry one generated EVOKE RGB frame."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted after a viewer connects, state changes, or a video chunk completes."""

    mode: str = MessageField(
        description=(
            "Active conditioning mode: `i2v` for an anchor image, `v2v` for a reference "
            "video and pose track, or `t2v` for prompt-only generation."
        )
    )
    prompt: str = MessageField(
        description=(
            "Active scene prompt. Omitted user text resolves to the configured scene-neutral "
            "exposure and temporal-stability prompt. A `set_prompt` change is sampled at the "
            "next native EVOKE chunk boundary without clearing autoregressive history."
        )
    )
    input_source: str = MessageField(
        description=(
            "Source of the active conditioning media: `built_in`, `uploaded`, or `none` in "
            "prompt-only mode."
        )
    )
    input_name: str = MessageField(
        description=(
            "Filename of the active image or reference video, or an empty string in "
            "prompt-only mode."
        )
    )
    pose_name: str = MessageField(
        description=(
            "Filename of the uploaded reference pose track in `v2v` mode. Empty in live "
            "camera-controlled `i2v` and prompt-only `t2v` modes."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed selected for the active or queued fresh rollout. It changes only "
            "when a conditioning command or `reset` supplies a non-negative seed."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of completed native chunks in the current rollout. Camera-conditioned "
            "chunks emit 36 RGB frames; prompt-only t2v emits 33 for chunk 1 and 36 later."
        )
    )
    next_chunk: int = MessageField(
        description=(
            "One-based chunk that camera and prompt commands accepted now will first affect. "
            "It returns to 1 after a rollout reset."
        )
    )
    max_chunks: int = MessageField(
        description=(
            "Chunks generated before the adapter automatically starts a fresh rollout from "
            "the active conditioning input."
        )
    )
    forward: float = MessageField(
        description=(
            "Active backward-to-forward translation in [-1, 1], sampled at `next_chunk` and "
            "held until changed or released. Always 0 in `t2v` mode."
        )
    )
    strafe: float = MessageField(
        description=(
            "Active left-to-right translation in [-1, 1], sampled at `next_chunk` and held "
            "until changed or released. Always 0 in `t2v` mode."
        )
    )
    vertical: float = MessageField(
        description=(
            "Active down-to-up translation in [-1, 1], sampled at `next_chunk` and held until "
            "changed or released. Always 0 in `t2v` mode."
        )
    )
    pitch: float = MessageField(
        description=(
            "Active downward-to-upward pitch in [-1, 1], sampled at `next_chunk` and held until "
            "changed or released. Always 0 in `t2v` mode."
        )
    )
    yaw: float = MessageField(
        description=(
            "Active left-to-right yaw in [-1, 1], sampled at `next_chunk` and held until "
            "changed or released. Always 0 in `t2v` mode."
        )
    )
    roll: float = MessageField(
        description=(
            "Active counterclockwise-to-clockwise roll in [-1, 1], sampled at `next_chunk` "
            "and held until changed or released. Always 0 in `t2v` mode."
        )
    )


class CommandApplied(ModelMessage):
    """Emitted when a requested control change succeeds."""

    action: str = MessageField(
        description=(
            "Command name that changed the world, such as `set_prompt`, `set_yaw`, "
            "or `reset`."
        )
    )
    applies_to_chunk: int = MessageField(
        description=(
            "One-based chunk that first observes the change. Playback-only changes report the "
            "next chunk they gate."
        )
    )
    detail: str = MessageField(
        description=(
            "Human-readable successful result, including the selected mode, filename, prompt, "
            "seed, or complete six-axis camera state as appropriate."
        )
    )


class RolloutRestarted(ModelMessage):
    """Emitted when the chunk horizon starts a fresh rollout automatically."""

    replaced_chunks: int = MessageField(
        description="Number of completed chunks replaced by the automatic fresh rollout."
    )
    max_chunks: int = MessageField(
        description="Configured horizon that triggered this automatic rollout restart."
    )
    seed: int = MessageField(
        description="Seed retained by the fresh rollout so the restart is observable."
    )


class EvokeState(InputState):
    """Expose shared text and camera controls for one EVOKE world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Active scene prompt, up to 4096 characters. Empty user text selects the "
            "configured scene-neutral stability prompt. Changes are sampled at the next "
            "native chunk boundary and do not reset autoregressive or geometric memory."
        ),
    )
    forward: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Backward (-1) to forward (1) camera translation, sampled at chunk boundaries "
            "and held until changed or released. Valid only in camera-controlled modes."
        ),
    )
    strafe: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) camera translation, sampled at chunk boundaries and held "
            "until changed or released. Valid only in camera-controlled modes."
        ),
    )
    vertical: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Down (-1) to up (1) camera translation, sampled at chunk boundaries and held "
            "until changed or released. Valid only in camera-controlled modes."
        ),
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Downward (-1) to upward (1) camera pitch, sampled at chunk boundaries and held "
            "until changed or released. Valid only in camera-controlled modes."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) camera yaw, sampled at chunk boundaries and held until "
            "changed or released. Valid only in camera-controlled modes."
        ),
    )
    roll: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Counterclockwise (-1) to clockwise (1) camera roll, sampled at chunk boundaries "
            "and held until changed or released. Valid only in camera-controlled modes."
        ),
    )
    _restart_requested: bool = True
