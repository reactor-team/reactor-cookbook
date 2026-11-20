"""Commands report a shared world and a streamed RGB video track."""
from collections import deque
from dataclasses import field

from reactor_runtime import InputState, MessageField, ModelMessage, Output, Video
from worldcrafter_camera import CameraPath, CameraPlanner


class WorldCrafterState(InputState):
    _world_id: int = 0
    _applied_world_id: int | None = None
    _image: bytes | None = None
    _prompt: str = ""
    _seed: int = 0
    _gravity: bool = True
    _queue: deque = field(default_factory=deque)
    _paused: bool = False
    _completed: int = 0
    _inflight: int | None = None
    _planner: CameraPlanner = field(default_factory=CameraPlanner)
    _camera_path: CameraPath | None = None


class WorldCrafterOutput(Output):
    main_video: Video


class StateUpdate(ModelMessage):
    world_id: int = MessageField(description="Identity of the selected world.")
    image_selected: bool = MessageField(description="Whether an image has been selected.")
    paused: bool = MessageField(description="Whether generation is paused.")
    completed_chunks: int = MessageField(description="Completed 33-frame chunks in this world.")
    queued_chunks: int = MessageField(description="Number of camera chunks awaiting generation.")
    max_chunks: int = MessageField(description="Maximum chunks per world before reset is required.")
    camera_path_chunks: int | None = MessageField(default=None, description="Chunks in the uploaded camera path, or null for `move` commands.")


class WorldStarted(ModelMessage):
    world_id: int = MessageField(description="New world identity.")
    seed: int = MessageField(description="Seed used for the new world.")


class CameraPathSelected(ModelMessage):
    world_id: int = MessageField(description="New world identity that follows the path.")
    chunks: int = MessageField(description="Number of 33-frame chunks in the path.")
    frames: int = MessageField(description="Total output video frames.")


class MotionQueued(ModelMessage):
    action: str = MessageField(description="Accepted camera action.")
    chunks: int = MessageField(description="Number of queued chunks for this action.")


class PlaybackChanged(ModelMessage):
    paused: bool = MessageField(description="Whether generation is paused.")


class ChunkGenerated(ModelMessage):
    world_id: int = MessageField(description="World identity for this video chunk.")
    chunk_index: int = MessageField(description="Zero-based chunk index.")
    frames: int = MessageField(description="Number of emitted RGB frames.")
    generation_seconds: float = MessageField(description="Runtime-measured generation duration.")
    camera_end: list[list[float]] = MessageField(description="Last frame's 3x4 camera-to-world pose relative to the input camera.")
    gravity_aligned: bool = MessageField(description="Whether image gravity calibration succeeded and is applied.")
    gpu_count: int = MessageField(description="Number of participating inference GPUs.")


class RolloutLimitReached(ModelMessage):
    completed_chunks: int = MessageField(description="Completed chunks; reset or start a new world to continue.")
