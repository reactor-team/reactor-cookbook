"""Client commands, status messages and video track."""

from typing import Literal

from reactor_runtime import InputState, ModelMessage, Output, Video

from fastframes_lingbot_model import Anchor, CameraAction

Phase = Literal["idle", "active", "stopped", "complete"]


class WorldState(InputState):
    _world_id: int = 0
    _owner: str | None = None
    _phase: Phase = "idle"
    _anchor: Anchor | None = None
    _prompt: str = ""
    _prompt_revision: int = 0
    _applied_revision: int | None = None
    _pending: CameraAction | None = None
    _held: CameraAction | None = None
    _last_action: CameraAction | None = None
    _chunks: int = 0
    _frames: int = 0
    _last_chunk_seconds: float | None = None


class WorldStarted(ModelMessage):
    world_id: int


class WorldStopped(ModelMessage):
    world_id: int


class CameraAccepted(ModelMessage):
    world_id: int
    action: CameraAction | None


class PromptAccepted(ModelMessage):
    world_id: int
    revision: int


class StateUpdate(ModelMessage):
    world_id: int
    phase: Phase
    queued_action: CameraAction | None
    held_action: CameraAction | None
    last_action: CameraAction | None
    prompt_pending: bool
    completed_chunks: int
    max_chunks: int
    playback_fps: int
    output_frames: int
    last_chunk_seconds: float | None


class WorldOutput(Output):
    main_video: Video
