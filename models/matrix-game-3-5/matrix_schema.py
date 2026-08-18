"""Define the public Reactor schema for Matrix-Game-3.5."""

from __future__ import annotations

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class MatrixGame35Output(Output):
    """Carry one generated Matrix-Game-3.5 RGB frame."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted after a viewer connects, state changes, or a `main_video` chunk completes."""

    prompt: str = MessageField(
        description=(
            "Active scene prompt. A successful `set_prompt` change is sampled at the next "
            "generated 12-frame chunk boundary."
        )
    )
    image_source: str = MessageField(
        description=(
            "Source of the active anchor image: `built_in` for the default or `upload` after "
            "`set_image`."
        )
    )
    image_name: str = MessageField(
        description=(
            "Filename of the active anchor image. Changes when `set_image` starts a fresh world."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed selected for the current or queued fresh world. Changes when `reset` "
            "receives a non-negative `seed`."
        )
    )
    paused: bool = MessageField(
        description=(
            "Whether continuous generation will stop before the next chunk; an in-flight chunk "
            "can still finish."
        )
    )
    step_queued: bool = MessageField(
        description=(
            "Whether `step` has queued one chunk while paused. Returns to false when generation "
            "of that chunk begins or another playback command cancels it."
        )
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether generation has reached `max_chunks`. When true, use `reset` or `set_image` "
            "before requesting another chunk."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of completed 12-frame `main_video` chunks in the current world. Returns to "
            "0 after `reset` or `set_image`."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that commands accepted now will first affect. Null when "
            "`limit_reached` is true."
        )
    )
    max_chunks: int = MessageField(
        description=(
            "Maximum 12-frame chunks available in one world before `reset` or `set_image` is "
            "required."
        )
    )
    forward: float = MessageField(
        description=(
            "Active backward-to-forward translation in [-1, 1]. Sampled by `next_chunk` and "
            "held until the camera axes are changed or released."
        )
    )
    strafe: float = MessageField(
        description=(
            "Active left-to-right translation in [-1, 1]. Sampled by `next_chunk` and held "
            "until the camera axes are changed or released."
        )
    )
    vertical: float = MessageField(
        description=(
            "Active down-to-up translation in [-1, 1]. Sampled by `next_chunk` and held until "
            "the camera axes are changed or released."
        )
    )
    pitch: float = MessageField(
        description=(
            "Active downward-to-upward pitch in [-1, 1]. Sampled by `next_chunk` and held until "
            "the camera axes are changed or released."
        )
    )
    yaw: float = MessageField(
        description=(
            "Active left-to-right yaw in [-1, 1]. Sampled by `next_chunk` and held until the "
            "camera axes are changed or released."
        )
    )
    roll: float = MessageField(
        description=(
            "Active counterclockwise-to-clockwise roll in [-1, 1]. Sampled by `next_chunk` and "
            "held until the camera axes are changed or released."
        )
    )


class RolloutLimitReached(ModelMessage):
    """Emitted once when `main_video` pauses after the final available chunk."""

    completed_chunks: int = MessageField(
        description="Number of 12-frame chunks completed when the world reached its limit."
    )
    max_chunks: int = MessageField(
        description=(
            "Configured limit reached by `completed_chunks`; use `reset` or `set_image` to "
            "continue generation."
        )
    )


class MatrixGame35State(InputState):
    """Expose shared text, camera, and playback controls for one Matrix world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        description=(
            "Active scene prompt, up to 4096 characters. Changes are sampled at the next "
            "generated 12-frame chunk boundary and persist across `reset`."
        ),
    )
    forward: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Backward (-1) to forward (1) translation sampled at chunk boundaries and held "
            "until the camera axes are changed or released."
        ),
    )
    strafe: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) translation sampled at chunk boundaries and held until "
            "the camera axes are changed or released."
        ),
    )
    vertical: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Down (-1) to up (1) translation sampled at chunk boundaries and held until the "
            "camera axes are changed or released."
        ),
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Downward (-1) to upward (1) pitch sampled at chunk boundaries and held until the "
            "camera axes are changed or released."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) yaw sampled at chunk boundaries and held until the camera "
            "axes are changed or released."
        ),
    )
    roll: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Counterclockwise (-1) to clockwise (1) roll sampled at chunk boundaries and held "
            "until the camera axes are changed or released."
        ),
    )
    paused: bool = InputField(
        default=False,
        description=(
            "Whether continuous generation pauses before the next chunk. Changes preserve the "
            "current world and release all camera axes plus a queued `step`."
        ),
    )
    _step_requested: bool = False
    _restart_requested: bool = True
    _limit_reached: bool = False
