"""Public Reactor contract for Lyra 2.0 autoregressive video exploration."""

from reactor_runtime import InputField, InputState, MessageField, ModelMessage, Output, Video


class Lyra2Output(Output):
    """One native 80-frame Lyra 2.0 video chunk."""
    main_video: Video


class Lyra2State(InputState):
    """The queued caption and held six-degree-of-freedom camera motion."""
    prompt: str = InputField(default="", max_length=4096, moderate=True, description=(
        "Active scene prompt, up to 4096 characters. A non-empty change is sampled at the next "
        "chunk boundary and preserves the current continuous world."
    ))
    _forward: float = 0.0
    _strafe: float = 0.0
    _vertical: float = 0.0
    _pitch: float = 0.0
    _yaw: float = 0.0
    _roll: float = 0.0
    _reset_requested: bool = False


class StateUpdate(ModelMessage):
    """Emitted after connection, state mutation, reset, or chunk completion."""
    image_name: str | None = MessageField(description="Selected seed-image filename, or null before selection.")
    prompt: str = MessageField(description="Caption queued for the next native chunk.")
    active_prompt: str | None = MessageField(description="Caption used by the latest completed chunk.")
    seed: int = MessageField(description="Base random seed; Lyra adds the zero-based AR chunk index.")
    generating: bool = MessageField(description="Whether seeding or one 80-frame AR step is running.")
    completed_chunks: int = MessageField(description="Number of committed 80-frame chunks in this world.")
    forward: float = MessageField(description="Held backward (-1) to forward (1) translation.")
    strafe: float = MessageField(description="Held left (-1) to right (1) translation.")
    vertical: float = MessageField(description="Held down (-1) to up (1) translation.")
    pitch: float = MessageField(description="Held look-down (-1) to look-up (1) rotation.")
    yaw: float = MessageField(description="Held turn-left (-1) to turn-right (1) rotation.")
    roll: float = MessageField(description="Held counter-clockwise (-1) to clockwise (1) roll.")


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image starts a fresh rollout."""
    filename: str = MessageField(description="Accepted image filename.")
    prompt: str = MessageField(description="Caption queued for the first generated chunk.")
    seed: int = MessageField(description="Base random seed for the fresh world.")


class PromptQueued(ModelMessage):
    """Emitted when a prompt is accepted for a forthcoming chunk."""
    prompt: str = MessageField(description="Normalized prompt accepted by `set_prompt`.")
    applies_to_chunk: int = MessageField(description="One-based chunk that will first sample the accepted prompt.")


class CameraChanged(ModelMessage):
    """Emitted when one or more held camera axes change successfully."""
    forward: float = MessageField(description="Active backward-to-forward camera direction.")
    strafe: float = MessageField(description="Active left-to-right camera direction.")
    vertical: float = MessageField(description="Active down-to-up camera direction.")
    pitch: float = MessageField(description="Active downward-to-upward pitch rate.")
    yaw: float = MessageField(description="Active left-to-right yaw rate.")
    roll: float = MessageField(description="Active counterclockwise-to-clockwise roll rate.")
    applies_to_chunk: int | None = MessageField(description=(
        "One-based first chunk expected to sample this state, or null before image selection."
    ))


class ChunkCompleted(ModelMessage):
    """Emitted after one continuous chunk finishes and before its RGB frames stream."""
    chunk: int = MessageField(description="One-based index of the completed chunk.")
    video_frames: int = MessageField(description="Number of RGB frames carried by `main_video`; always 80.")
    generation_seconds: float = MessageField(description="Wall-clock seconds spent generating and decoding this chunk.")
    prompt: str = MessageField(description="Prompt sampled by the completed chunk.")


class ResetQueued(ModelMessage):
    """Emitted when `reset` queues a fresh rollout from the selected image."""
    seed: int = MessageField(description="Seed selected for the fresh rollout.")
    replaced_chunks: int = MessageField(description="Number of completed chunks the fresh rollout replaces.")
