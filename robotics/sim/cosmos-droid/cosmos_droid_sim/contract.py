# ──────────────────────────────────────────────────────────────────────────
# Wire contracts, BOTH of them, because this example is a gateway.
#
# Sim side (openpi msgpack-WebSocket, what RoboLab's own Cosmos3Client
# speaks): one observation dict in, one {"action": [32, 8]} chunk out, per
# request. RoboLab pre-composes its three cameras into a single image
# (wrist full-width on top, the two exterior views half-size side by side
# below), so the gateway's first job is to split that composite back into
# the three views the served model declares.
#
# Reactor side (reactor-sdk): three video tracks + three commands
# (set_task_description / set_proprio_json / set_executed_step_json) in,
# action_prediction messages out. The executed-step echo is the model's
# flow-control gate: it will not predict chunk N+1 until the echoed step
# counter passes chunk N (the model is stateless per request, and there is
# no reset event on this wire at all).
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json

import numpy as np

# Reactor track names, in the order the checkpoint declares them
# (wrist_image_left, exterior_image_1_left, exterior_image_2_left).
TRACK_WRIST = "wrist_view"
TRACK_EXTERIOR_1 = "exterior_view_1"
TRACK_EXTERIOR_2 = "exterior_view_2"
TRACKS = (TRACK_WRIST, TRACK_EXTERIOR_1, TRACK_EXTERIOR_2)

# Commands / fields / message keys on the Reactor wire.
CMD_SET_TASK = "set_task_description"
CMD_SET_PROPRIO = "set_proprio_json"
CMD_SET_EXECUTED_STEP = "set_executed_step_json"
FIELD_TASK = "task_description"
FIELD_PROPRIO = "proprio_json"
FIELD_EXECUTED_STEP = "executed_step_json"
MESSAGE_ACTION_FIELD = "action"
MESSAGE_STEP_FIELD = "step"

ACTION_HORIZON = 32  # rows per predicted chunk
ACTION_DOF = 8       # 7 absolute joint positions (rad) + 1 gripper
CONTROL_HZ = 15.0    # the DROID control rate; one chunk = ~2.13 s of motion

# openpi observation keys as RoboLab's Cosmos3Client sends them.
OBS_IMAGE = "observation/image"
OBS_JOINTS = "observation/joint_position"
OBS_GRIPPER = "observation/gripper_position"
OBS_PROMPT = "prompt"


def split_composite(comp: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose RoboLab's pre-composed frame into the three wire views.

    The layout is wrist (H, W) on top and the two exterior views at
    (H/2, W/2) side by side below, so the wrist occupies the top 2/3 of the
    composite's height. The served model re-composes this exact layout
    internally. Publishing the three views separately is what exercises
    the shipped three-track contract rather than bypassing it.
    """
    comp = np.asarray(comp, dtype=np.uint8)
    if comp.ndim != 3 or comp.shape[2] != 3:
        raise ValueError(f"composite has shape {comp.shape}, want (H, W, 3)")
    h = comp.shape[0] * 2 // 3
    w = comp.shape[1]
    return {
        TRACK_WRIST: np.ascontiguousarray(comp[:h]),
        TRACK_EXTERIOR_1: np.ascontiguousarray(comp[h:, : w // 2]),
        TRACK_EXTERIOR_2: np.ascontiguousarray(comp[h:, w // 2 :]),
    }


def encode_proprio(obs: dict) -> str:
    """The model's proprio_json: row-lists, N>=1 rows, last row = current.

    RoboLab sends one sample per request, so this emits exactly one row per
    key, the same shape a robot client sends.
    """
    joints = np.atleast_2d(np.asarray(obs[OBS_JOINTS], dtype=np.float64))
    gripper = np.asarray(obs[OBS_GRIPPER], dtype=np.float64).reshape(-1, 1)
    payload = {
        "joint_position": joints.tolist(),
        "gripper_position": gripper.tolist(),
    }
    for key, rows in payload.items():
        if not np.isfinite(np.asarray(rows)).all():
            raise ValueError(f"proprio key {key!r} is not finite")
    return json.dumps(payload, separators=(",", ":"))


def encode_executed_step(step: int, rows: np.ndarray) -> str:
    """The flow-gate echo. The gate advances on the step counter alone, but
    the echoed rows are the chunk the sim actually executed: a genuine
    report, which keeps this payload honest if the engine ever starts
    conditioning on it."""
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"executed rows have shape {arr.shape}, want (n, cols)")
    return json.dumps({"step": int(step), "action": arr.tolist()}, separators=(",", ":"))


def decode_chunk(message: object) -> tuple[int, np.ndarray] | None:
    """action_prediction -> (step, [H, 8] array), or None for other traffic.

    Accepts the raw envelope ``{"data": {...}}`` or the inner dict directly;
    the data channel delivers both shapes in the wild.
    """
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except ValueError:
            return None
    if not isinstance(message, dict):
        return None
    if MESSAGE_ACTION_FIELD not in message and isinstance(message.get("data"), dict):
        message = message["data"]
    raw = message.get(MESSAGE_ACTION_FIELD)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, ACTION_DOF)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DOF or not np.isfinite(arr).all():
        raise ValueError(f"malformed action chunk: shape {arr.shape}")
    return int(message.get(MESSAGE_STEP_FIELD, -1)), arr
