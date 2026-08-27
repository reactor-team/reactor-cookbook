"""Define ABot-World configuration and its public Reactor schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

DEFAULT_PROMPT = (
    "A realistic outdoor world scene with a navigable path, natural lighting, "
    "detailed ground texture, and stable forward motion."
)


@dataclass(frozen=True)
class ModelAsset:
    """Describe one pinned public model snapshot and its local directory."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class ExampleScene:
    """Pair one built-in starting image with its scene prompt."""

    image: Path
    prompt: str


@dataclass(frozen=True)
class ABotWorldConfig:
    """Hold validated source, checkpoint, stream, and example settings."""

    source_path: Path
    source_url: str
    source_revision: str
    checkpoint: ModelAsset
    seed: int
    height: int
    width: int
    max_chunks: int
    examples: tuple[ExampleScene, ...]


class ABotWorldOutput(Output):
    """Stream one native ABot-World RGB chunk on ``main_video``."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted when shared world state changes, a chunk completes, or a viewer connects."""

    image_source: Literal["uploaded", "built_in"] | None = MessageField(
        description=(
            'Source of the active starting image: "uploaded", "built_in", or null before '
            "`set_image` or `random_image` succeeds."
        )
    )
    image_name: str | None = MessageField(
        description="Active starting-image filename, or null before an image is selected."
    )
    prompt: str = MessageField(
        description=(
            "Scene prompt queued for `next_chunk`. It becomes `active_prompt` when that "
            "chunk completes."
        )
    )
    active_prompt: str | None = MessageField(
        description=(
            "Prompt used by the most recently completed chunk, or null before generation starts."
        )
    )
    seed: int = MessageField(
        description="Random seed used when the current or queued rollout is initialized."
    )
    reset_queued: bool = MessageField(
        description=(
            "Whether the selected image, prompt, and seed will initialize a fresh rollout at "
            "the next inference boundary."
        )
    )
    generating: bool = MessageField(
        description="Whether the adapter is initializing or generating an upstream chunk."
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether the rollout has reached `max_chunks`; use `reset`, `set_image`, or "
            "`random_image` before requesting another chunk."
        )
    )
    completed_chunks: int = MessageField(
        description="Number of completed autoregressive chunks in the active rollout."
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that accepted prompt and key changes will first affect, or null "
            "when no image is selected or the rollout limit is reached."
        )
    )
    max_chunks: int = MessageField(
        description="Maximum chunks generated before a fresh rollout is required."
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Native W/A/S/D/I/J/K/L keys held for later chunks, ordered as the upstream "
            "action channels."
        )
    )
    queued_taps: list[str] = MessageField(
        description=(
            "Keys pressed since the previous chunk sample. These survive an early key release "
            "and are consumed once by `next_chunk`."
        )
    )
    sampled_keys: list[str] = MessageField(
        description="Native keys sampled by the most recently completed chunk."
    )


class ActionChanged(ModelMessage):
    """Emitted when ``set_key_state`` changes held or queued native controls."""

    key: str = MessageField(description="Native ABot-World key changed by the command.")
    pressed: bool = MessageField(
        description="Physical state accepted for `key`: true for held or false for released."
    )
    pressed_keys: list[str] = MessageField(
        description="Complete ordered key set held after the command is processed."
    )
    queued_taps: list[str] = MessageField(
        description="Complete ordered short-tap set waiting for the next chunk sample."
    )
    applies_to_chunk: int | None = MessageField(
        description=(
            "One-based chunk that will first sample this control state, or null before an "
            "image is selected or after the rollout limit."
        )
    )


class ControlsReleased(ModelMessage):
    """Emitted when ``release_controls`` returns all native keys to neutral."""

    applies_to_chunk: int | None = MessageField(
        description=(
            "One-based chunk that will first receive a neutral action, or null when no chunk "
            "can currently be generated."
        )
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image queues a fresh rollout."""

    source: Literal["uploaded", "built_in"] = MessageField(
        description='Image source selected by the command: "uploaded" or "built_in".'
    )
    filename: str = MessageField(description="Filename selected for the fresh rollout.")
    prompt: str = MessageField(
        description="Effective non-empty prompt for the fresh rollout."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the image selection; always 1."
    )


class PromptQueued(ModelMessage):
    """Emitted when ``set_prompt`` queues text for a future chunk."""

    prompt: str = MessageField(
        description="Trimmed, non-empty prompt accepted by the command."
    )
    applies_to_chunk: int | None = MessageField(
        description=(
            "One-based chunk that will first use `prompt`, or null until an image is selected."
        )
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when ``reset`` queues the selected image as a fresh world."""

    seed: int = MessageField(description="Random seed selected for the fresh rollout.")
    replaced_chunks: int = MessageField(
        description="Number of completed chunks in the rollout being replaced."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the reset; always 1 for the fresh rollout."
    )


class RolloutLimitReached(ModelMessage):
    """Emitted when the active rollout reaches its configured chunk limit."""

    completed_chunks: int = MessageField(
        description="Number of chunks completed when generation stopped."
    )
    max_chunks: int = MessageField(
        description="Configured chunk limit reached by the active rollout."
    )


class ABotWorldState(InputState):
    """Expose the prompt shared by one ABot-World session."""

    prompt: str = InputField(
        default=DEFAULT_PROMPT,
        max_length=4096,
        moderate=True,
        description=(
            "Scene prompt queued for the next generated chunk. Whitespace-only values are "
            "rejected by `set_prompt`."
        ),
    )
    _pressed_keys: frozenset[str] = frozenset()
    _activated_keys: frozenset[str] = frozenset()
    _reset_requested: bool = False
    _limit_reached: bool = False
    _seed: int = 0
