# ──────────────────────────────────────────────────────────────────────────
# Wire contracts, BOTH of them, because this example is a gateway.
#
# Sim side (openpi msgpack-WebSocket, what RoboLab's own DreamZero client
# speaks): one observation dict in, one (24, 8) action chunk out, per
# request, synchronously.
#
# Reactor side (reactor-sdk): three named video tracks + three commands
# (set_prompt / set_joint_position / set_gripper_position) in, action_chunk
# messages out. The model is FREE-RUNNING: it does not wait to be asked. Once
# a prompt is set and every camera has a frame, it infers whenever the
# cameras go fresh and broadcasts the result. Reconciling that with RoboLab's
# synchronous client is what bridge.py's obs_seq gate is for.
#
# The camera index shift below is the one real trap in this file.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

# ── Reactor side ────────────────────────────────────────────────────────────

#: Track names, in the order the model declares them. ``exterior_1`` is the
#: REAL left exterior view. See :data:`OBS_KEY_TO_TRACK`.
TRACKS: tuple[str, str, str] = ("exterior_1", "exterior_2", "wrist")

CMD_SET_PROMPT = "set_prompt"
CMD_SET_JOINTS = "set_joint_position"
CMD_SET_GRIPPER = "set_gripper_position"
CMD_RESET = "reset"
FIELD_PROMPT = "prompt"
FIELD_JOINTS = "joint_position"
FIELD_GRIPPER = "gripper_position"

MESSAGE_ACTION_CHUNK = "action_chunk"
MESSAGE_EPISODE_STARTED = "episode_started"
MESSAGE_EPISODE_RESET = "episode_reset"
MESSAGE_PROMPT_ACCEPTED = "prompt_accepted"
MESSAGE_COMMAND_ERROR = "command_error"

#: Every message type this gateway subscribes to.
SUBSCRIPTIONS: tuple[str, ...] = (
    MESSAGE_ACTION_CHUNK,
    MESSAGE_EPISODE_STARTED,
    MESSAGE_EPISODE_RESET,
    MESSAGE_PROMPT_ACCEPTED,
    MESSAGE_COMMAND_ERROR,
)

#: (action horizon, dim): 24 steps of 7 absolute joint targets + 1 gripper.
ACTION_SHAPE: tuple[int, int] = (24, 8)

#: The DROID checkpoint's eval transform geometry. Neither side resizes:
#: RoboLab already sends this size, and the model resizes anything else
#: itself. Used here only to size the tracks' opening black frame.
FRAME_HW: tuple[int, int] = (180, 320)

#: Joints the model's state command carries.
JOINT_DIM = 7

# ── Sim side: RoboLab's openpi observation keys ──────────────────────────────

OBS_JOINTS = "observation/joint_position"
OBS_GRIPPER = "observation/gripper_position"
OBS_PROMPT = "prompt"
OBS_SESSION_ID = "session_id"

#: RoboLab observation key -> Reactor track name.
#:
#: Read this carefully, because getting it backwards feeds the model a black
#: primary view and does **not** error. RoboLab numbers its exterior cameras
#: from **0**; the checkpoint numbers its video keys from **1**. So:
#:
#: - RoboLab's ``exterior_image_0_left`` (its real left camera, the one with
#:   the scene in it) is Reactor's ``exterior_1``.
#: - RoboLab's ``exterior_image_1_left`` is Reactor's ``exterior_2``, and
#:   under RoboLab's default ``--cam2-source black`` it is an all-black frame.
#:   That is deliberate: it matches the checkpoint's training-time camera
#:   dropout. Leave it black.
OBS_KEY_TO_TRACK: dict[str, str] = {
    "observation/exterior_image_0_left": "exterior_1",
    "observation/exterior_image_1_left": "exterior_2",
    "observation/wrist_image_left": "wrist",
}

REPLY_ACTIONS = "actions"


def extract_frames(obs: dict) -> dict[str, np.ndarray]:
    """Pull the three camera frames out of a RoboLab infer request.

    Returns ``{track_name: (H, W, 3) uint8}``. A camera the client omitted
    becomes a black frame, the same substitution RoboLab itself makes for
    its second exterior slot, so the model sees the training-time dropout
    pattern rather than a shape error.
    """
    frames: dict[str, np.ndarray] = {}
    for obs_key, track in OBS_KEY_TO_TRACK.items():
        value = obs.get(obs_key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 4:
            # A client may batch temporal context. The model keeps its own
            # rolling window, so take the newest frame only.
            arr = arr[-1]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{obs_key!r} has shape {arr.shape}, want (H, W, 3) uint8 RGB"
            )
        frames[track] = np.ascontiguousarray(arr, dtype=np.uint8)

    if not frames:
        raise ValueError(
            "infer request carried no camera frames; expected at least one of "
            f"{sorted(OBS_KEY_TO_TRACK)}; is this RoboLab's DreamZero client?"
        )

    reference = next(iter(frames.values()))
    for track in TRACKS:
        frames.setdefault(track, np.zeros_like(reference))
    return {track: frames[track] for track in TRACKS}


def extract_state(obs: dict) -> tuple[list[float], float]:
    """Pull ``(joint_position[7], gripper_position)`` out of a request.

    Missing state falls back to zeros, which is what the model does too, but
    note what that means: with zero joints the model's joint outputs are
    *relative deltas*, not absolute targets. Losing the state silently
    changes what the numbers mean, so it is logged by the caller rather than
    passed over.
    """
    joint = obs.get(OBS_JOINTS)
    if joint is None:
        joints = [0.0] * JOINT_DIM
    else:
        flat = np.asarray(joint, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError(f"{OBS_JOINTS!r} contains non-finite values")
        joints = [float(v) for v in flat[:JOINT_DIM]]
        joints.extend([0.0] * (JOINT_DIM - len(joints)))

    gripper = obs.get(OBS_GRIPPER)
    if gripper is None:
        grip = 0.0
    else:
        flat = np.asarray(gripper, dtype=np.float64).reshape(-1)
        grip = float(flat[0]) if flat.size else 0.0
    # The model clamps gripper_position to [0, 1]; RoboLab's sim value is
    # already normalised, so clamp here rather than let the whole state
    # update be rejected on a rounding overshoot.
    return joints, min(1.0, max(0.0, grip))


def decode_chunk(data: dict) -> tuple[np.ndarray, int, int, float]:
    """``action_chunk`` payload -> ``(actions, obs_seq, chunk_index, seconds)``.

    ``obs_seq`` is not optional. It names the observations the chunk was
    computed from, and it is the only thing that tells a fresh chunk from the
    one that was already in flight when the observation was pushed. Without
    it this gateway would hand RoboLab a plan computed from the *previous*
    frame: right shape, finite values, plausible trajectory, wrong.
    """
    if "obs_seq" not in data:
        raise RuntimeError(
            "action_chunk carries no obs_seq field, so a fresh chunk cannot be "
            "told from one already in flight. The served model predates the "
            "obs_seq contract. Ask your Reactor contact which deployment to "
            "point at."
        )
    actions = np.asarray(data[REPLY_ACTIONS], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != ACTION_SHAPE[1]:
        raise ValueError(
            f"actions have shape {actions.shape}, want (n, {ACTION_SHAPE[1]})"
        )
    if not np.isfinite(actions).all():
        raise ValueError("action chunk contains non-finite values")
    return (
        actions,
        int(data["obs_seq"]),
        int(data.get("chunk_index", -1)),
        float(data.get("inference_seconds", 0.0)),
    )


def action_reply(actions: np.ndarray) -> dict:
    """Wrap a chunk in the reply RoboLab unpacks.

    RoboLab's client accepts either a bare ndarray or a dict with an
    ``actions`` key. This sends the dict, which is self-describing and leaves
    room to attach diagnostics without changing the client.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {REPLY_ACTIONS: arr}
