"""Client state and audio-video tracks for InWorld."""

from dataclasses import field
from typing import Literal

import numpy as np
from reactor_runtime import (
    Audio,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

from inworld_model import SAMPLE_RATE, InWorldAnchor


class InWorldAudio(Audio):
    sample_rate = SAMPLE_RATE


class InWorldOutput(Output):
    main_video: Video
    main_audio: InWorldAudio


class InWorldState(InputState):
    prompt: str = InputField(
        default="",
        max_length=8192,
        moderate=True,
        description="Scene and audio description. Updates condition the next segment while retaining the previous segment's 17-frame tail.",
    )
    paused: bool = InputField(
        default=True,
        description="Pause generation at a segment boundary. Buffered media keeps playing.",
    )
    _references: dict[int, np.ndarray] = field(default_factory=dict)
    _video: bytes | None = None
    _world_id: int = 0
    _applied_world_id: int | None = None
    _anchor: InWorldAnchor | None = None
    _completed: int = 0
    _last_seconds: float | None = None


class StateUpdate(ModelMessage):
    """Emitted when shared video settings or generation progress changes."""

    world_id: int = MessageField(
        description="Current video identity; changes only on Start."
    )
    completed_segments: int = MessageField(
        description="Successfully generated segments in this video."
    )
    paused: bool = MessageField(description="Whether generation is paused.")
    prompt: str = MessageField(description="Text to use for the next segment.")
    reference_count: int = MessageField(
        description="Number of staged ordered reference images."
    )
    has_video: bool = MessageField(
        description="Whether a parent video is staged for Start."
    )
    last_segment_seconds: float | None = MessageField(
        description="Runtime-measured generation time of the latest segment."
    )


class InputAccepted(ModelMessage):
    """Returned to the caller after the requested input or control is accepted."""

    kind: Literal["reference", "video", "clear", "prompt", "paused", "start"] = (
        MessageField(
            description="Accepted operation: reference, video, clear, prompt, paused, or start."
        )
    )
