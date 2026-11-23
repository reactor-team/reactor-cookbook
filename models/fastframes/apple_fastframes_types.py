"""FastFrames streaming commands and output contract."""

from typing import Literal

from reactor_runtime import InputState, ModelMessage, Output, Video

from apple_fastframes_model import Variant

Phase = Literal["idle", "generating", "paused", "stopped", "complete"]


class FastFramesState(InputState):
    _world_id: int = 0
    _applied_id: int | None = None
    _phase: Phase = "idle"
    _prompt: str = ""
    _seed: int = 0
    _prompt_revision: int = 0
    _applied_prompt_revision: int | None = None
    _frames: int = 0
    _chunks: int = 0
    _last_chunk_seconds: float | None = None


class FastFramesOutput(Output):
    main_video: Video


class ClipCommandAccepted(ModelMessage):
    """Acknowledges the command for the caller."""

    world_id: int


class StateUpdate(ModelMessage):
    """Shared rollout state, including whether the accepted prompt is in use."""

    variant: Variant
    world_id: int
    phase: Phase
    prompt: str
    prompt_pending: bool
    completed_chunks: int
    output_frames: int
    last_chunk_seconds: float | None
