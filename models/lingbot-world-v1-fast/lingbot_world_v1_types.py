"""Define the public Reactor schema for LingBot-World v1 Fast."""

from __future__ import annotations

from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class LingBotWorldOutput(Output):
    """Carry one generated LingBot-World RGB frame batch."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Report the complete shared world state after each accepted transition."""

    prompt: str = MessageField(
        description=(
            "Active scene prompt, or an empty string before an image is selected. A successful "
            "`set_prompt` change is encoded for the next generated `main_video` chunk while the "
            "existing visual self-KV history remains."
        )
    )
    image_source: str = MessageField(
        description=(
            "Source of the active anchor image: `none` before selection, `built_in` after "
            "`random_image`, and `uploaded` after `set_image`."
        )
    )
    image_name: str = MessageField(
        description="Filename of the active anchor image used by the current or queued rollout."
    )
    seed: int = MessageField(
        description=(
            "Non-negative random seed for the current or queued rollout. It changes only when "
            "`reset` receives a non-negative seed."
        )
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether the rollout exhausted its safe RoPE timeline. Use `reset`, `set_image`, or "
            "`random_image` before requesting another chunk."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of completed native three-latent chunks in the active world. Chunk 1 emits "
            "9 frames and each later chunk emits 12 frames."
        )
    )
    last_chunk_seconds: float | None = MessageField(
        description=(
            "Wall-clock seconds spent in model generation and causal VAE decode for the most "
            "recent completed chunk. Null before the first chunk of a fresh world."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that newly accepted camera or prompt controls will first affect. "
            "Null before image selection or after the rollout limit is reached."
        )
    )
    next_chunk_frames: int | None = MessageField(
        description=(
            "Frames emitted by `next_chunk`: 9 for the first causal chunk, 12 thereafter, and "
            "null before image selection or after the rollout limit is reached."
        )
    )
    max_chunks: int = MessageField(
        description="Maximum chunks available before a fresh anchor rollout is required."
    )
    forward: float = MessageField(
        description=(
            "Active backward-to-forward translation in [-1, 1], sampled at the next chunk "
            "boundary and held until changed or released."
        )
    )
    strafe: float = MessageField(
        description=(
            "Active left-to-right translation in [-1, 1], sampled at the next chunk boundary "
            "and held until changed or released."
        )
    )
    vertical: float = MessageField(
        description=(
            "Active down-to-up translation in [-1, 1], sampled at the next chunk boundary and "
            "held until changed or released."
        )
    )
    pitch: float = MessageField(
        description=(
            "Active downward-to-upward pitch in [-1, 1], sampled at the next chunk boundary and "
            "held until changed or released."
        )
    )
    yaw: float = MessageField(
        description=(
            "Active left-to-right yaw in [-1, 1], sampled at the next chunk boundary and held "
            "until changed or released."
        )
    )
    roll: float = MessageField(
        description=(
            "Active counterclockwise-to-clockwise roll in [-1, 1], sampled at the next chunk "
            "boundary and held until changed or released."
        )
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image starts a fresh world."""

    source: Literal["uploaded", "built_in"] = MessageField(
        description="Image source accepted by the command."
    )
    filename: str = MessageField(description="Selected anchor-image filename.")
    prompt: str = MessageField(description="Non-empty prompt for the fresh world.")
    seed: int = MessageField(description="Random seed for the fresh world.")


class PromptQueued(ModelMessage):
    """Emitted when `set_prompt` queues text conditioning for the active world."""

    prompt: str = MessageField(description="Trimmed prompt accepted by `set_prompt`.")
    applies_to_chunk: int = MessageField(
        description="One-based chunk that first encodes the new prompt."
    )


class CameraMotionChanged(ModelMessage):
    """Emitted when one held camera axis changes."""

    forward: float = MessageField(description="Active backward-to-forward motion.")
    strafe: float = MessageField(description="Active left-to-right motion.")
    vertical: float = MessageField(description="Active down-to-up motion.")
    pitch: float = MessageField(description="Active downward-to-upward pitch motion.")
    yaw: float = MessageField(description="Active left-to-right yaw motion.")
    roll: float = MessageField(
        description="Active counterclockwise-to-clockwise roll motion."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk expected to sample these values first."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when `reset` queues a fresh world from the selected image."""

    seed: int = MessageField(description="Random seed for the fresh world.")
    replaced_chunks: int = MessageField(
        description="Completed chunks discarded by the reset."
    )


class RolloutLimitReached(ModelMessage):
    """Emitted when the rollout reaches the end of its safe timeline."""

    completed_chunks: int = MessageField(
        description="Number of completed chunks when the rollout reached its configured limit."
    )
    max_chunks: int = MessageField(
        description=(
            "Configured limit reached by `completed_chunks`; start a fresh anchor rollout to "
            "continue generation."
        )
    )


class LingBotWorldState(InputState):
    """Expose shared text and camera controls for one LingBot world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Active non-empty scene prompt, up to 4096 characters. A change is encoded at the "
            "next generated chunk boundary without clearing visual self-KV history."
        ),
    )
    forward: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Backward (-1) to forward (1) translation sampled at chunk boundaries and held "
            "until changed or released."
        ),
    )
    strafe: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) translation sampled at chunk boundaries and held until "
            "changed or released."
        ),
    )
    vertical: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Down (-1) to up (1) translation sampled at chunk boundaries and held until changed "
            "or released."
        ),
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Downward (-1) to upward (1) pitch sampled at chunk boundaries and held until "
            "changed or released."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) yaw sampled at chunk boundaries and held until changed or "
            "released."
        ),
    )
    roll: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Counterclockwise (-1) to clockwise (1) roll sampled at chunk boundaries and held "
            "until changed or released."
        ),
    )
    _restart_requested: bool = True
    _limit_reached: bool = False
