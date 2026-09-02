"""Public Reactor contract for Zing 0.5."""

from __future__ import annotations

from typing import Literal
from reactor_runtime import InputField, InputState, MessageField, ModelMessage, Output, Video

ZingKey = Literal["w", "a", "s", "d", "i", "j", "k", "l"]


class ZingOutput(Output):
    """Carry one generated RGB frame batch."""
    main_video: Video


class ZingState(InputState):
    """Store the shared prompt and held world controls."""
    prompt: str = InputField(
        default="", max_length=4096, moderate=True,
        description=(
            "Active scene and motion description, up to 4096 characters. A non-empty change is "
            "sampled at the next chunk boundary and preserves the current world history."
        ),
    )
    _pressed_keys: frozenset[str] = frozenset()
    _reset_requested: bool = True


class StateUpdate(ModelMessage):
    """Emitted after connection, state mutation, reset, or chunk completion."""
    prompt: str = MessageField(
        description="Prompt for the current or queued world, or an empty string before input."
    )
    active_prompt: str | None = MessageField(
        description="Prompt used by the latest completed chunk, or null before generation."
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Controls currently held by `set_key`, sampled when the next chunk starts and held "
            "until changed or released."
        )
    )
    conditioning: Literal["none", "text", "uploaded", "built_in"] = MessageField(
        description=(
            "Source that starts the world: `none` while waiting for input, `text` for a prompt, "
            "or `uploaded`/`built_in` for an anchor image."
        )
    )
    image_name: str | None = MessageField(
        description="Selected anchor-image filename, or null for text-to-video or before input."
    )
    seed: int = MessageField(description="Random seed for the current or queued world.")
    completed_chunks: int = MessageField(
        description="Number of chunks completed since the latest world reset."
    )
    reset_queued: bool = MessageField(
        description="Whether a fresh world will be initialized before the next chunk."
    )
    generating: bool = MessageField(
        description="Whether image preparation or one video chunk is in progress."
    )


class PromptQueued(ModelMessage):
    """Emitted when a prompt is accepted for a forthcoming chunk."""
    prompt: str = MessageField(description="Normalized prompt accepted by `set_prompt`.")
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample the accepted prompt."
    )
    resets_rollout: bool = MessageField(
        description="Whether the prompt starts a fresh text-to-video world from chunk one."
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image starts a fresh world."""
    source: Literal["uploaded", "built_in"] = MessageField(
        description="Selected image source: `uploaded` or `built_in`."
    )
    filename: str = MessageField(description="Filename of the selected anchor image.")
    prompt: str = MessageField(description="Prompt selected for the fresh world.")
    seed: int = MessageField(description="Random seed selected for the fresh world.")


class ActionChanged(ModelMessage):
    """Emitted when one held movement or look control changes successfully."""
    key: ZingKey = MessageField(description="Control changed by `set_key`.")
    pressed: bool = MessageField(description="Whether `key` is now held or released.")
    pressed_keys: list[str] = MessageField(
        description="Complete set of controls held after the change."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample the complete held state."
    )


class ControlsReleased(ModelMessage):
    """Emitted when every held movement and look control is released."""
    released_keys: list[str] = MessageField(
        description="Controls that were held before `release_controls`."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample neutral controls."
    )


class RolloutReset(ModelMessage):
    """Emitted when `reset` queues a fresh world from the selected condition."""
    seed: int = MessageField(description="Random seed selected for the fresh world.")
    replaced_chunks: int = MessageField(
        description="Number of completed chunks that the fresh world will replace."
    )


class ChunkCompleted(ModelMessage):
    """Emitted after one causal chunk finishes and before its RGB frames stream."""
    chunk: int = MessageField(description="One-based index of the completed chunk.")
    video_frames: int = MessageField(
        description="Number of RGB frames carried by `main_video`; normally 16."
    )
    generation_seconds: float = MessageField(
        description="Wall-clock seconds spent generating and decoding this chunk."
    )
    prompt: str = MessageField(description="Prompt sampled by the completed chunk.")
    action_keys: list[str] = MessageField(
        description="Complete held control state sampled by the completed chunk."
    )
    cache_frames: int = MessageField(
        description="Number of prior world positions retained for subsequent chunks."
    )
