"""Public Reactor contract for YUME-1.5."""

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

Movement = Literal[
    "none",
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
]
View = Literal[
    "none",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "tilt_up_left",
    "tilt_up_right",
    "tilt_down_left",
    "tilt_down_right",
]


class YumeOutput(Output):
    """Carry the next 29-frame continuation on `main_video`."""

    main_video: Video


class YumeState(InputState):
    """Store the prompt exposed through Reactor's generated state command."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description="Scene and event description for forthcoming chunks. Scene commands initialize it, and the generated setter or `set_prompt` changes it at the next chunk boundary without restarting the world.",
    )
    _pressed_keys: frozenset[str] = frozenset()
    _reset_requested: bool = False


class StateUpdate(ModelMessage):
    """Emitted on connection, after every state mutation, and after each completed chunk."""

    mode: Literal[
        "uninitialized", "image_to_video", "video_to_video", "text_to_video"
    ] = MessageField(
        description="How the current world was initialized, or `uninitialized` before any scene has been selected."
    )
    conditioning_name: str | None = MessageField(
        description="Filename of the image or video anchoring the current or queued world, or null for text-only and uninitialized sessions."
    )
    prompt: str = MessageField(
        description="Scene and event description currently scheduled for forthcoming chunks."
    )
    pressed_keys: list[str] = MessageField(
        description="Ordered keys held for forthcoming chunks. W/A/S/D control translation, while arrow keys control pan and tilt; compatible keys can be combined."
    )
    seed: int = MessageField(
        description="Seed used to initialize the current rollout or its queued replacement."
    )
    reset_queued: bool = MessageField(
        description="Whether the next chunk boundary will restart from the selected scene and discard accumulated history."
    )
    generating: bool = MessageField(
        description="Whether a continuation chunk is currently being generated."
    )
    completed_chunks: int = MessageField(
        description="Number of chunks completed since the current world was initialized or last reset."
    )
    next_chunk: int | None = MessageField(
        description="One-based index of the first chunk affected by a newly accepted prompt or control change, or null before scene selection."
    )


class SceneQueued(ModelMessage):
    """Emitted when a text, image, or video scene is accepted for a fresh rollout."""

    mode: Literal["image_to_video", "video_to_video", "text_to_video"] = MessageField(
        description="Initialization mode selected for the fresh rollout."
    )
    conditioning_name: str | None = MessageField(
        description="Filename of the uploaded image or video, or null for a text-only scene."
    )
    prompt: str = MessageField(
        description="Normalized scene and event description selected for the fresh rollout."
    )
    seed: int = MessageField(description="Seed that will initialize the fresh rollout.")


class ActionChanged(ModelMessage):
    """Emitted after one control key changes state or all controls are released."""

    key: str = MessageField(description="Changed key, or `all` when `release_controls` released the complete set.")
    pressed: bool = MessageField(
        description="Whether `key` is held after the change; always false when `key` is `all`."
    )
    pressed_keys: list[str] = MessageField(
        description="Complete ordered set of movement and view keys held after the change."
    )
    applies_to_chunk: int = MessageField(
        description="One-based index of the first chunk that will use the resulting held-key set."
    )


class PromptChanged(ModelMessage):
    """Emitted when a new prompt is accepted for forthcoming chunks."""

    prompt: str = MessageField(
        description="Normalized scene and event description accepted for forthcoming chunks."
    )
    applies_to_chunk: int = MessageField(
        description="One-based index of the first chunk that will use the new prompt."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when the selected scene is accepted for restart at the next boundary."""

    seed: int = MessageField(description="Seed that will initialize the restarted rollout.")
    replaced_chunks: int = MessageField(
        description="Number of completed chunks whose accumulated history the reset will discard."
    )


class ChunkCompleted(ModelMessage):
    """Emitted after one 29-frame continuation chunk completes."""

    chunk: int = MessageField(description="One-based index of the completed chunk in the current rollout.")
    frames: int = MessageField(
        description="Number of RGB frames delivered for this chunk on `main_video`; currently 29."
    )
    generation_seconds: float = MessageField(
        description="Wall-clock seconds spent producing this chunk, including video decoding."
    )
    prompt: str = MessageField(
        description="Scene and event description used for this chunk, excluding YUME's generated control text."
    )
    conditioned_prompt: str = MessageField(
        description="Exact text condition used for this chunk, including YUME's movement, view, and speed controls."
    )
    movement: Movement = MessageField(
        description="Translation direction applied throughout this completed chunk."
    )
    view: View = MessageField(
        description="Pan and tilt direction applied throughout this completed chunk."
    )
