"""Define the public Reactor schema for HY-World 1.5."""

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


class HYWorld15Output(Output):
    """Carry one generated HY-World 1.5 RGB frame batch."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted after connection, state mutation, rollout reset, or chunk completion."""

    image_source: Literal["uploaded", "built_in"] | None = MessageField(
        description=(
            'Source of the active anchor image: "uploaded", "built_in", or null before '
            "an image is selected."
        )
    )
    image_name: str | None = MessageField(
        description="Active image filename, or null before an image is selected."
    )
    prompt: str | None = MessageField(
        description=(
            "Scene prompt queued for the next generated chunk, or null before an image is selected."
        )
    )
    active_prompt: str | None = MessageField(
        description=(
            "Prompt used by the most recently completed chunk, or null before generation starts."
        )
    )
    seed: int = MessageField(
        description="Random seed used to initialize the current or queued fresh world."
    )
    reset_queued: bool = MessageField(
        description=(
            "Whether the selected image, prompt, and seed will initialize a fresh world before "
            "the next chunk."
        )
    )
    generating: bool = MessageField(
        description="Whether the model is currently resetting or generating a chunk."
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of completed chunks in the current world. Chunk 1 has 13 frames; later "
            "chunks have 16 frames."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that commands accepted now will first affect, or null when the "
            "rollout limit is reached."
        )
    )
    next_chunk_frames: Literal[13, 16] | None = MessageField(
        description=(
            "Number of RGB frames expected in `next_chunk`: 13 for the initial causal chunk, "
            "16 afterward, or null at the rollout limit."
        )
    )
    max_chunks: int = MessageField(
        description="Maximum chunks available before a fresh world is required."
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether generation reached `max_chunks` and now requires `reset`, `set_image`, or "
            "`random_image`."
        )
    )
    forward: float = MessageField(
        description=(
            "Held backward-to-forward movement in [-1, 1], sampled when `next_chunk` begins."
        )
    )
    strafe: float = MessageField(
        description=(
            "Held left-to-right movement in [-1, 1], sampled when `next_chunk` begins."
        )
    )
    pitch: float = MessageField(
        description="Held downward-to-upward pitch in [-1, 1], sampled by `next_chunk`."
    )
    yaw: float = MessageField(
        description="Held left-to-right yaw in [-1, 1], sampled by `next_chunk`."
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image queues a fresh world."""

    source: Literal["uploaded", "built_in"] = MessageField(
        description=(
            'Selected image source: "uploaded" for `set_image` or "built_in" for `random_image`.'
        )
    )
    filename: str = MessageField(
        description="Selected image filename used to initialize the fresh world."
    )
    prompt: str = MessageField(
        description="Effective non-empty scene prompt used by the fresh world's first chunk."
    )


class PromptQueued(ModelMessage):
    """Emitted when a prompt is queued for a forthcoming chunk."""

    prompt: str = MessageField(
        description="Trimmed non-empty scene prompt accepted by the model."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first use the new prompt."
    )


class CameraMotionChanged(ModelMessage):
    """Emitted when the complete native camera motion state changes."""

    forward: float = MessageField(
        description="Backward (-1) to forward (1) movement held until changed or released."
    )
    strafe: float = MessageField(
        description="Left (-1) to right (1) movement held until changed or released."
    )
    pitch: float = MessageField(
        description="Look down (-1) to up (1) pitch held until changed or released."
    )
    yaw: float = MessageField(
        description="Turn left (-1) to right (1) yaw held until changed or released."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample all four camera values together."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when a manual reset queues a fresh world."""

    seed: int = MessageField(description="Random seed that the fresh world will use.")
    replaced_chunks: int = MessageField(
        description="Number of completed chunks discarded by the reset."
    )


class ChunkCompleted(ModelMessage):
    """Emitted after one causal video chunk finishes successfully."""

    chunk: int = MessageField(description="One-based chunk that finished generating.")
    frames: Literal[13, 16] = MessageField(
        description="RGB frame count produced by the completed chunk."
    )
    prompt: str = MessageField(
        description="Scene prompt sampled by the completed chunk."
    )
    generation_seconds: float = MessageField(
        description="Wall-clock seconds spent generating and decoding the completed chunk."
    )
    forward: float = MessageField(
        description="Forward movement sampled by the completed chunk."
    )
    strafe: float = MessageField(
        description="Strafe movement sampled by the completed chunk."
    )
    pitch: float = MessageField(description="Pitch sampled by the completed chunk.")
    yaw: float = MessageField(description="Yaw sampled by the completed chunk.")


class RolloutLimitReached(ModelMessage):
    """Emitted once when generation idles after the configured final chunk."""

    completed_chunks: int = MessageField(
        description="Number of chunks completed when the world reached its limit."
    )
    max_chunks: int = MessageField(
        description="Configured chunk limit requiring a fresh world before generation continues."
    )


class HYWorld15State(InputState):
    """Expose prompt and native camera motion for one shared world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Scene prompt sampled at the next generated chunk boundary. Use `set_prompt` to "
            "change it after selecting an image."
        ),
    )
    _forward: float = 0.0
    _strafe: float = 0.0
    _pitch: float = 0.0
    _yaw: float = 0.0
    _restart_requested: bool = False
    _limit_reached: bool = False
