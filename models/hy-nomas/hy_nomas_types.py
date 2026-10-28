"""Hy-Nomas client state, video track, and shared state messages."""

from __future__ import annotations

import numpy as np
from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

from hy_nomas_camera import CAMERAS
from hy_nomas_model import WorldStart


class HyNomasOutput(Output):
    main_video: Video


class HyNomasState(InputState):
    prompt: str = InputField(
        default="",
        max_length=8192,
        moderate=True,
        description="Scene/action text or a single-segment JSON prompt for the next chunk. Live edits continue the world; frames already generated keep the previous prompt.",
    )
    paused: bool = InputField(
        default=True,
        description="Hold generation between chunks. Frames already generated can keep playing.",
    )
    camera: str = InputField(
        default="text_only",
        choices=CAMERAS,
        description="Camera motion for the next chunk. text_only follows the prompt; stationary holds the view still.",
    )
    translation_strength: float = InputField(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Camera movement strength for the next chunk.",
    )
    rotation_strength: float = InputField(
        default=2.5,
        ge=0.1,
        le=3.0,
        description="Camera look strength for the next chunk.",
    )
    _world_id: int = 0
    _applied_world_id: int | None = None
    _start: WorldStart | None = None
    _completed: int = 0
    _generated: int = 0
    _emitted: int = 0
    _complete: bool = False
    _last_seconds: float | None = None
    _camera_poses: np.ndarray | None = None
    _camera_intrinsics: np.ndarray | None = None
    _camera_name: str = ""
    _camera_cursor: int = 0
    _held_keys: tuple[str, ...] = ()
    _pending_keys: tuple[str, ...] | None = None


class StateUpdate(ModelMessage):
    """Emitted when the world changes and sent to each viewer on connection."""

    pressed_keys: list[str] = MessageField(description="Held WASD/arrow keys.")
    world_id: int = MessageField(
        description="World identity; changes only on Start or Reset."
    )
    prompt: str = MessageField(description="Accepted prompt for the next chunk.")
    paused: bool = MessageField(description="Whether generation is paused.")
    camera: str = MessageField(description="Accepted camera motion for the next chunk.")
    translation_strength: float = MessageField(
        description="Accepted camera translation gain."
    )
    rotation_strength: float = MessageField(
        description="Accepted camera rotation gain."
    )
    completed_chunks: int = MessageField(
        description="Successfully generated chunks in this world."
    )
    emitted_frames: int = MessageField(
        description="Total frames submitted for playback."
    )
    last_chunk_seconds: float | None = MessageField(
        description="Runtime-measured duration of the last chunk."
    )
    active: bool = MessageField(description="Whether Start has created a world.")
    limit_reached: bool = MessageField(
        description="Whether the 120-second world duration has completed."
    )
    seed: int | None = MessageField(
        description="Seed for this world, or null before Start."
    )
    chunk_frames: int | None = MessageField(
        description="Requested frames per chunk, or null before Start."
    )
    camera_path: str = MessageField(
        description="Uploaded camera trajectory filename, or empty if none is staged."
    )
    camera_frames_remaining: int = MessageField(
        description="Frames remaining in the staged camera path."
    )
