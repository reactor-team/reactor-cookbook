"""Define the public Reactor schema for Matrix-Game 3.0."""

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

MovementKey = Literal["w", "a", "s", "d"]
MOVEMENT_KEYS = ["w", "a", "s", "d"]


class MatrixGame30Output(Output):
    """Carry one generated Matrix-Game 3.0 RGB frame batch."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted after connection, state mutation, or a `main_video` chunk boundary."""

    prompt: str = MessageField(
        description=(
            "Scene prompt encoded for the current or queued fresh rollout. `set_prompt` "
            "restarts from the selected anchor because Matrix encodes text before its "
            "autoregressive loop."
        )
    )
    image_source: str = MessageField(
        description=(
            "Source of the selected anchor image: `none` before selection, `built_in` after "
            "`random_image`, or `uploaded` after `set_image`."
        )
    )
    image_name: str = MessageField(
        description=(
            "Filename of the anchor image used by the current or queued fresh rollout."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed used by the current or queued fresh rollout. A non-negative "
            "`reset.seed` replaces it."
        )
    )
    restart_queued: bool = MessageField(
        description=(
            "Whether the next generated chunk first restarts the official autoregressive "
            "rollout from the selected image, prompt, and seed."
        )
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether the official 12-iteration rollout has completed. Use `reset`, "
            "`set_image`, `set_prompt`, or `random_image` to begin another rollout."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of completed native chunks in the current rollout. Chunk 1 contains 57 "
            "frames and later chunks contain 40 frames."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based native chunk that accepted keyboard, pitch, and yaw commands will "
            "first affect, or null before image selection or after the rollout limit."
        )
    )
    next_chunk_frames: int | None = MessageField(
        description=(
            "Frame count of `next_chunk`: 57 for a fresh rollout, 40 thereafter, or null "
            "before image selection or after the rollout limit."
        )
    )
    max_chunks: int = MessageField(
        description=(
            "Native iteration count retained from the upstream distilled inference recipe; "
            "the default is 12 chunks."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "W/S/A/D keys held for the next native chunk. Each key maps directly to one "
            "binary channel in Matrix's six-value keyboard condition."
        )
    )
    pitch: float = MessageField(
        description=(
            "Normalized camera pitch in [-1, 1] held for the next native chunk. The adapter "
            "scales it to Matrix's native mouse-x range [-0.1, 0.1]."
        )
    )
    yaw: float = MessageField(
        description=(
            "Normalized camera yaw in [-1, 1] held for the next native chunk. The adapter "
            "scales it to Matrix's native mouse-y range [-0.1, 0.1]."
        )
    )


class ControlsChanged(ModelMessage):
    """Emitted after a keyboard, pitch, or yaw control command is applied."""

    control: str = MessageField(
        description=(
            "Wire name of the command that produced this snapshot: `set_key_state`, "
            "`set_pitch`, or `set_yaw`."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "W/S/A/D keys held after the command. Multiple perpendicular keys produce the "
            "diagonal binary combinations supported by Matrix's pose implementation."
        )
    )
    pitch: float = MessageField(
        description="Normalized pitch held after the command; zero is neutral."
    )
    yaw: float = MessageField(
        description="Normalized yaw held after the command; zero is neutral."
    )
    applies_to_chunk: int | None = MessageField(
        description=(
            "One-based native chunk that will sample these controls, or null after the "
            "rollout reaches its limit."
        )
    )


class RolloutLimitReached(ModelMessage):
    """Emitted when the final native `main_video` chunk completes."""

    completed_chunks: int = MessageField(
        description="Number of native chunks completed when the rollout reached its limit."
    )
    max_chunks: int = MessageField(
        description=(
            "Configured native iteration limit reached by `completed_chunks`; start a fresh "
            "rollout to continue."
        )
    )


class MatrixGame30State(InputState):
    """Expose shared controls for one Matrix-Game 3.0 autoregressive world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Scene description, up to 4096 characters. `set_prompt` requires non-empty text, "
            "restarts from the selected anchor, and queues the fresh 57-frame chunk because "
            "the upstream pipeline encodes text once before generation."
        ),
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Normalized downward (-1) to upward (1) pitch sampled at native chunk boundaries "
            "and held until changed or controls are released."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Normalized left (-1) to right (1) yaw sampled at native chunk boundaries and "
            "held until changed or controls are released."
        ),
    )
    _pressed_keys: frozenset[str] = frozenset()
    _restart_requested: bool = True
    _limit_reached: bool = False
