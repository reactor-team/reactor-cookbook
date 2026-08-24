# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Type definitions for the RLDX-1 VLA model (video-in -> action-out).

RLDX-1 is a vision-language-action model: it consumes camera views + robot
proprioceptive state + a language task, and emits an action chunk (it does NOT
generate video). So the Reactor port declares the 3 RoboCasa camera views as
**input** tracks, the robot state + task as ``InputState``, and streams the
predicted action chunk back as an :class:`ActionPrediction` ``ModelMessage`` over
the data channel — which the cpp_sdk client receives via ``on_message``.
"""

from __future__ import annotations

from typing import Dict, List

from reactor_runtime import (
    Input,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Video,
)

# No `@dataclass` decorators below: on the standalone runtime the Input /
# InputState / ModelMessage bases build the dataclass themselves when the
# subclass is declared (Input additionally binds one live buffer per declared
# track through its own __init__, which an author-applied @dataclass would
# overwrite).


# --- Input tracks: the 3 RoboCasa camera views the client publishes ---
class RLDXInput(Input):
    left_view: Video
    right_view: Video
    wrist_view: Video


# --- Client state: robot proprio state (fallback carrier) + the task ---
class RLDXState(InputState):
    # 16-dim RoboCasa proprio state for the current frame, as JSON:
    #   {"end_effector_position_relative":[x,y,z],
    #    "end_effector_rotation_relative":[qx,qy,qz,qw],
    #    "gripper_qpos":[a,b], "base_position":[x,y,z], "base_rotation":[qx,qy,qz,qw]}
    # The pipeline reads this JSON off the video FRAME METADATA now — every view's
    # frame is tagged with it — and falls back to this field for a client whose
    # SDK has no per-frame metadata API. Same bytes either way (see robot_state.py).
    state_json: str = InputField(
        default="",
        max_length=2000,
        description=(
            "JSON of the robot proprioceptive state vectors for the current frame. "
            "Fallback carrier: prefer tagging each video frame with these bytes as "
            "that frame's metadata, which attaches the state to the frames it was "
            "read with. Sent here only if your SDK cannot tag frames."
        ),
    )
    task_description: str = InputField(
        default="pick up the mug",
        max_length=300,
        moderate=True,
        description="Language task / instruction conditioning the policy",
    )
    # private: episode-reset flag
    _reset: bool = False


# --- Session-start handshake: what the loaded checkpoint expects ---
class ModelSchema(ModelMessage):
    """The loaded checkpoint's input/output contract, announced at session start.

    Configure the client from this instead of hardcoding: which camera views to
    publish, the frame resolution, how the temporal window is sampled, the
    robot-state vectors the state JSON must carry and where to send them
    (`state_source`), and the action chunk shape `action_prediction` will stream
    back. Sent on connect and again when the
    first frames arrive; request it explicitly any time with `get_schema` —
    the reliable pattern is to send `get_schema` once the data channel opens.
    Values come from the checkpoint's own configuration, so they change when
    the served checkpoint changes.
    """
    views: List[str] = MessageField(default=None, description="Camera views the model consumes, in checkpoint order. Publish one input video track per entry, named exactly as listed.")
    resolution: List[int] = MessageField(default=None, description="[height, width] each view is processed at. Frames published at other sizes are resized server-side.")
    video_delta_indices: List[int] = MessageField(default=None, description="Temporal window offsets in control steps, most-recent last (e.g. [-6,-4,-2,0] = 4 frames, every 2nd control step). The server samples the window; publish frames at any rate of at least `control_hz`.")
    control_hz: float = MessageField(default=None, description="Control steps per second the window offsets and `exec_horizon` are expressed in.")
    state_dims: Dict[str, int] = MessageField(default=None, description="Robot proprioceptive state vectors and their dimensions. The state JSON must carry every key at exactly this length, whichever carrier it arrives on (see `state_source`).")
    state_source: str = MessageField(default=None, description="Where the server reads robot state from, in preference order. `frame_metadata` = tag every video frame with the state JSON (the state then arrives attached to the frames it was read with); `set_state_json` remains accepted as a fallback for clients that cannot tag frames.")
    state_tag_keys: List[str] = MessageField(default=None, description="Reserved optional keys a tagging client may embed in the state JSON alongside the state vectors, as integers: `capture_us` (microseconds on the client's own clock for the snapshot the state and frames were read from) and `seq` (the client's tick counter). The server selects state only from the aligned frame commit; these keys order its candidates when capture-time alignment is unavailable and are echoed in `action_prediction` as `source_capture_us` / `source_seq`.")
    action_dims: Dict[str, int] = MessageField(default=None, description="Action vectors and their dimensions, as streamed in `action_prediction`.")
    action_horizon: int = MessageField(default=None, description="Steps per action chunk: each `action_prediction` field is [action_horizon, dim].")
    exec_horizon: int = MessageField(default=None, description="Control steps the client is expected to execute per chunk; the server re-plans on this cadence.")
    rtc_mode: str = MessageField(default=None, description="Real-time-chunking inference mode the checkpoint was loaded with.")
    dtype: str = MessageField(default=None, description="Numeric dtype for state and action vectors.")
    embodiment: str = MessageField(default=None, description="Embodiment the checkpoint is configured for.")
    state_fallback: str = MessageField(default=None, description="Server behaviour when robot state is missing or invalid: `hold_last`, `zero`, or `error`. A `command_error` signals whenever it engages.")


# --- Outbound message: the predicted action chunk ---
class ActionPrediction(ModelMessage):
    """One action chunk. Each field is an [`action_horizon`, dim] nested list of floats (shapes announced in `model_schema`)."""
    end_effector_position: List[List[float]] = MessageField(default=None, description="End-effector position chunk")
    end_effector_rotation: List[List[float]] = MessageField(default=None, description="End-effector rotation chunk")
    gripper_close: List[List[float]] = MessageField(default=None, description="Gripper-close chunk")
    base_motion: List[List[float]] = MessageField(default=None, description="Base-motion chunk")
    control_mode: List[List[float]] = MessageField(default=None, description="Control-mode chunk")
    step: int = MessageField(default=0, description="inference step index")
    source_capture_us: int = MessageField(default=None, description="Client-clock microseconds echoed from the `capture_us` the client embedded in its state tag — attach this chunk to your own timeline with it; null when the client embeds no stamp.")
    source_seq: int = MessageField(default=None, description="Client tick counter echoed from the `seq` the client embedded in its state tag — attach this chunk to your own timeline with it; null when the client embeds no stamp.")
    view_skew_us: int = MessageField(default=None, description="Capture-time spread, in microseconds, across the views' committed frames — the freshest commit's, i.e. the frames nearest this chunk's execution point, not the whole temporal window's. Zero when every view contributed a frame stamped with the same instant, which is what a client stamping a tick once should normally see; a value near a control period means one view lagged a whole step and the others were held back to match it. Null when the transport carries no per-frame capture stamps.")


class CommandError(ModelMessage):
    command: str = MessageField(default="", description="the command that failed")
    reason: str = MessageField(default="", description="why it failed")
