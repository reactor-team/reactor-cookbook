# ──────────────────────────────────────────────────────────────────────────
# Wire contract. There is no proprioception to assemble: the policy's state
# carries only the task string and an echo of the actions just executed.
# robosuite consumes the predicted actions
# directly, so this module is just the JSON schema plus the view mapping.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from typing import Sequence

import numpy as np

# Track name -> LIBERO obs key. ORDER MATTERS on the wire, so publish in
# this order.
VIEWS: dict[str, str] = {
    "agentview": "agentview_image",
    "eye_in_hand": "robot0_eye_in_hand_image",
}

ACTION_HORIZON = 16  # steps per predicted chunk
ACTION_DOF = 7       # 7-DoF OSC_POSE delta + gripper

# The server treats the seed chunk's leading frame as conditioning rather
# than a prediction, and skips it in its own eval, so this harness does too.
SEED_SKIP_STEPS = ACTION_HORIZON // 4

CAM_SIZE = 128        # matches the training set's native resolution
CONTROL_HZ = 20       # robosuite's default control_freq for LIBERO envs

ECHO_FLOAT_DIGITS = 5  # keep the echo well under the state field's length cap
ECHO_MAX_LEN = 8000
TASK_MAX_LEN = 300


def encode_executed(actions: Sequence[np.ndarray]) -> str:
    """Encode the actions just executed as the state's executed_action_json.

    The policy conditions on the realized trajectory, so this must report
    what the env actually stepped, not what was predicted. Empty string for
    the first prediction of an episode, which the model reads as "nothing
    executed yet".
    """
    if not len(actions):
        return ""
    rows = [
        [round(float(v), ECHO_FLOAT_DIGITS) for v in np.asarray(a).ravel()[:ACTION_DOF]]
        for a in actions
    ]
    payload = json.dumps(rows, separators=(",", ":"))
    if len(payload) > ECHO_MAX_LEN:
        raise ValueError(
            f"executed_action_json is {len(payload)} chars, over the model's "
            f"{ECHO_MAX_LEN} limit ({len(rows)} steps); reduce --exec-steps"
        )
    return payload


def decode_chunk(action: object) -> list[np.ndarray]:
    """Decode a predicted [16, 7] chunk into per-step action vectors.

    Short/ragged chunks are truncated rather than padded, so a malformed
    prediction degrades instead of crashing the rollout.
    """
    if not isinstance(action, (list, tuple)):
        return []
    steps: list[np.ndarray] = []
    for row in list(action)[:ACTION_HORIZON]:
        if not isinstance(row, (list, tuple)):
            continue
        vec = np.zeros(ACTION_DOF, dtype=float)
        vals = [float(v) for v in row[:ACTION_DOF]]
        vec[: len(vals)] = vals
        steps.append(vec)
    return steps
