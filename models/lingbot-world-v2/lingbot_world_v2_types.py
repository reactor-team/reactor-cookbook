"""Define the public Reactor schema for LingBot-World-V2."""

from __future__ import annotations

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class LingBotWorldV2Output(Output):
    """Carry one generated LingBot-World-V2 RGB frame batch."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted after connection, state mutation, reset, or chunk completion."""

    prompt: str | None = MessageField(
        description=(
            "Active scene prompt, or null before an image is selected. A successful "
            "`set_prompt` command is sampled when `next_chunk` begins."
        )
    )
    image_source: str = MessageField(
        description=(
            "Source of the selected anchor: `none`, `built_in`, or `uploaded`. Generation "
            "requires either a built-in or uploaded image."
        )
    )
    image_name: str = MessageField(
        description=(
            "Filename of the selected anchor image, or an empty string before an image is "
            "selected."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed used by the active or queued fresh rollout. A non-negative `reset` "
            "seed replaces it when that reset begins."
        )
    )
    reset_queued: bool = MessageField(
        description=(
            "Whether a fresh rollout from the selected image, prompt, and seed will begin "
            "before another chunk is generated."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of chunks completed in the current causal rollout. It returns to zero "
            "when `set_image`, `random_image`, or `reset` starts a fresh rollout."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that newly accepted prompt and camera controls will first affect, "
            "or null after the rollout limit is reached."
        )
    )
    next_chunk_frames: int | None = MessageField(
        description=(
            "Expected RGB frames in `next_chunk`: 13 for a fresh rollout and 16 afterward, "
            "or null after the rollout limit is reached."
        )
    )
    max_chunks: int = MessageField(
        description=(
            "Maximum chunks available in one world. Use `reset`, `set_image`, or "
            "`random_image` to start another world after reaching it."
        )
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether `completed_chunks` has reached `max_chunks`. When true, `reset` or image "
            "selection is required before another chunk can run."
        )
    )
    forward: float = MessageField(
        description=(
            "Active backward-to-forward camera direction in [-1, 1], sampled when "
            "`next_chunk` starts and held until changed or released."
        )
    )
    strafe: float = MessageField(
        description=(
            "Active left-to-right camera direction in [-1, 1], sampled when `next_chunk` "
            "starts and held until changed or released."
        )
    )
    vertical: float = MessageField(
        description=(
            "Active down-to-up camera direction in [-1, 1], sampled when `next_chunk` starts "
            "and held until changed or released."
        )
    )
    pitch: float = MessageField(
        description=(
            "Active downward-to-upward pitch rate in [-1, 1], sampled when `next_chunk` starts "
            "and held until changed or released."
        )
    )
    yaw: float = MessageField(
        description=(
            "Active left-to-right yaw rate in [-1, 1], sampled when `next_chunk` starts and "
            "held until changed or released."
        )
    )
    roll: float = MessageField(
        description=(
            "Active counterclockwise-to-clockwise roll rate in [-1, 1], sampled when "
            "`next_chunk` starts and held until changed or released."
        )
    )


class PromptQueued(ModelMessage):
    """Emitted when a prompt is accepted for a forthcoming chunk."""

    prompt: str = MessageField(
        description="Normalized prompt accepted by `set_prompt`."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample the accepted prompt."
    )


class CameraMotionChanged(ModelMessage):
    """Emitted when one or more held camera axes change successfully."""

    forward: float = MessageField(
        description="Active backward-to-forward camera direction."
    )
    strafe: float = MessageField(description="Active left-to-right camera direction.")
    vertical: float = MessageField(description="Active down-to-up camera direction.")
    pitch: float = MessageField(description="Active downward-to-upward pitch rate.")
    yaw: float = MessageField(description="Active left-to-right yaw rate.")
    roll: float = MessageField(
        description="Active counterclockwise-to-clockwise roll rate."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample this complete camera state."
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image starts a fresh rollout."""

    source: str = MessageField(
        description="Selected image source: `uploaded` or `built_in`."
    )
    filename: str = MessageField(description="Filename of the selected anchor image.")
    prompt: str = MessageField(description="Prompt selected for the fresh rollout.")
    applies_to_chunk: int = MessageField(
        description="First chunk of the fresh rollout; always 1."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when a fresh causal rollout is queued successfully."""

    seed: int = MessageField(description="Seed selected for the fresh rollout.")
    replaced_chunks: int = MessageField(
        description="Number of completed chunks that the fresh rollout will replace."
    )
    applies_to_chunk: int = MessageField(
        description="First chunk of the fresh rollout; always 1."
    )


class ChunkCompleted(ModelMessage):
    """Emitted after one causal chunk finishes and before its RGB frames stream."""

    chunk: int = MessageField(description="One-based index of the completed chunk.")
    frames: int = MessageField(
        description="Number of RGB frames emitted by this chunk."
    )
    generation_seconds: float = MessageField(
        description="Wall-clock seconds spent generating and decoding this chunk."
    )
    prompt: str = MessageField(description="Prompt sampled by the completed chunk.")
    forward: float = MessageField(
        description="Forward axis sampled by the completed chunk."
    )
    strafe: float = MessageField(
        description="Strafe axis sampled by the completed chunk."
    )
    vertical: float = MessageField(
        description="Vertical axis sampled by the completed chunk."
    )
    pitch: float = MessageField(
        description="Pitch axis sampled by the completed chunk."
    )
    yaw: float = MessageField(description="Yaw axis sampled by the completed chunk.")
    roll: float = MessageField(description="Roll axis sampled by the completed chunk.")


class RolloutLimitReached(ModelMessage):
    """Emitted once when generation idles after the last available chunk."""

    completed_chunks: int = MessageField(
        description="Number of chunks completed when the rollout reached its limit."
    )
    max_chunks: int = MessageField(
        description="Configured rollout limit reached by `completed_chunks`."
    )


class LingBotWorldV2State(InputState):
    """Store shared prompt and camera state."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Active scene prompt, up to 4096 characters. A non-empty change is sampled at the "
            "next chunk boundary and preserves the current causal world."
        ),
    )
    _forward: float = 0.0
    _strafe: float = 0.0
    _vertical: float = 0.0
    _pitch: float = 0.0
    _yaw: float = 0.0
    _roll: float = 0.0
    _reset_requested: bool = False
