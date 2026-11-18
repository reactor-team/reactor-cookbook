"""Public controls, private session state, and client-visible video progress."""

from typing import Literal

import numpy as np
from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

from movieroll_model import MAX_PROMPTS, Anchor


class MovieRollState(InputState):
    prompt: str = InputField(
        default="",
        moderate=True,
        max_length=8192,
        description="Full scene description for upcoming rolling windows.",
    )
    paused: bool = InputField(
        default=False,
        description="Hold generation between chunks; buffered video can still play.",
    )
    _world_id: int = 0
    _applied_world_id: int | None = None
    _anchor: Anchor | None = None
    _image: np.ndarray | None = None
    _completed: int = 0
    _frames: int = 0
    _last_seconds: float | None = None
    _prompts_used: int = 0
    _end_reason: Literal["stopped", "history_full"] | None = None


class MovieRollOutput(Output):
    """Silent landscape video at 864 by 480; Runtime adapts playout to generation."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Shared snapshot after changes and chunks; the joining snapshot goes only to its caller."""

    prompt: str = MessageField(
        description="Accepted full caption for upcoming windows."
    )
    active: bool = MessageField(description="Whether a video has been started.")
    paused: bool = MessageField(description="Whether future generation is paused.")
    has_image: bool = MessageField(
        description="Whether an initial image is staged for Start."
    )
    mode: Literal["t2v", "i2v"] | None = MessageField(
        description="Current video's checkpoint, null before Start."
    )
    seed: int | None = MessageField(
        description="Current video's seed, null before Start."
    )
    world_id: int = MessageField(description="Video identity within this session.")
    completed_chunks: int = MessageField(
        description="Successfully published rolling chunks in this video."
    )
    output_frames: int = MessageField(description="Frames published in this video.")
    last_chunk_seconds: float | None = MessageField(
        description="Runtime's measured latest generation time."
    )
    prompt_updates_remaining: int = MessageField(
        description=f"Further accepted live captions before this video's {MAX_PROMPTS}-caption limit."
    )
    end_reason: Literal["stopped", "history_full"] | None = MessageField(
        description="Why generation ended, when applicable."
    )
