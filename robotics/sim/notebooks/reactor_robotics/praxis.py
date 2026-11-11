# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Praxis-1 helpers: track names, geometry and robot-state encoders.

Praxis-1 is served as two endpoints with one prediction protocol:

* ``reactor/praxis-1-droid``: a single Franka arm, ``exterior`` and ``wrist``
  cameras.
* ``reactor/praxis-1-yam``: a bimanual YAM rig, ``top``, ``wrist_left`` and
  ``wrist_right`` cameras.

The protocol is **lock-step, echo-gated after the first chunk**. Once the
session has a task, 25 frames of every camera and at least 25 rows of
measured state, it emits ``action_prediction`` with ``step: 0`` (21 rows).
Every later chunk waits for ``set_executed_step`` with the previous ``step``
and has 24 rows (1.6 s at 15 Hz). Only one chunk is outstanding at a time.

Robot state is a JSON **string** inside ``robot_state_json``: a history of
end-effector poses ``[x, y, z, qw, qx, qy, qz]`` (meters, scalar-first
quaternion) and absolute gripper values (``0`` open), oldest row first.
Targets come back absolute, as ``[x, y, z]`` followed by the 3 x 3 rotation
matrix row by row.

``step`` arrives as a JSON number that the SDK may decode as a float
(``0.0``); compare it with ``int(step)``.

See https://docs.reactor.inc/robotics/praxis/overview for the full contract.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

DROID_MODEL = "reactor/praxis-1-droid"
YAM_MODEL = "reactor/praxis-1-yam"

#: Camera track names. Every track must stream before the first chunk.
DROID_TRACKS: tuple[str, str] = ("exterior", "wrist")
YAM_TRACKS: tuple[str, str, str] = ("top", "wrist_left", "wrist_right")

#: Largest frame the session uses, as (height, width). DROID accepts any
#: aspect up to 512 x 288; YAM requires 16:10 and at most 416 on the long side.
DROID_FRAME_HW: tuple[int, int] = (288, 512)
YAM_FRAME_HW: tuple[int, int] = (260, 416)

#: Frames per second for cameras and rows per second for state.
FPS = 15
#: Frames per camera and state rows each prediction consumes.
HISTORY_ROWS = 25
#: Rows in the first chunk of an episode, and in every later chunk.
FIRST_CHUNK_ROWS = 21
CHUNK_ROWS = 24
#: Values per returned pose row: xyz + a row-major 3 x 3 rotation matrix.
POSE_WIDTH = 12

#: DROID ``set_pose_frame`` values. YAM has no pose-frame command: it uses
#: the rig's world frame.
DROID_POSE_FRAMES: tuple[str, ...] = ("fr3_hand", "robolab_droid", "molmo_spaces_droid")

DROID_STATE_MAX_LEN = 16_000
YAM_STATE_MAX_LEN = 32_000


def _arm_rows(ee_pose, grip, name: str) -> dict:
    poses = np.atleast_2d(np.asarray(ee_pose, dtype=np.float64))
    grips = np.asarray(grip, dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"{name} ee_pose must be (N, 7), got {poses.shape}")
    if grips.shape[0] != poses.shape[0]:
        raise ValueError(
            f"{name} has {poses.shape[0]} pose rows but {grips.shape[0]} grip rows"
        )
    if not (np.isfinite(poses).all() and np.isfinite(grips).all()):
        raise ValueError(f"{name} state contains non-finite values")
    if (np.linalg.norm(poses[:, 3:], axis=1) == 0).any():
        raise ValueError(f"{name} has a zero quaternion")
    return {"ee_pose": poses.tolist(), "grip": grips.tolist()}


def _dump(state: dict, max_len: int) -> str:
    out = json.dumps(state, separators=(",", ":"))
    if len(out) > max_len:
        raise ValueError(
            f"robot_state_json is {len(out)} characters, over the {max_len} limit; "
            "send fewer history rows"
        )
    return out


def encode_droid_state(
    ee_pose: Sequence[Sequence[float]] | np.ndarray,
    grip: Sequence[float] | np.ndarray,
) -> str:
    """Build DROID ``robot_state_json`` from (N, 7) poses and N gripper values."""
    return _dump(_arm_rows(ee_pose, grip, "arm"), DROID_STATE_MAX_LEN)


def encode_yam_state(
    left_ee_pose: Sequence[Sequence[float]] | np.ndarray,
    left_grip: Sequence[float] | np.ndarray,
    right_ee_pose: Sequence[Sequence[float]] | np.ndarray,
    right_grip: Sequence[float] | np.ndarray,
) -> str:
    """Build YAM ``robot_state_json`` for both arms, in the rig world frame."""
    return _dump(
        {
            "left": _arm_rows(left_ee_pose, left_grip, "left"),
            "right": _arm_rows(right_ee_pose, right_grip, "right"),
        },
        YAM_STATE_MAX_LEN,
    )


def check_chunk(reply: dict, fields: dict[str, int], step: int, rows: int) -> None:
    """Assert a reply's step, row count, widths and finiteness.

    ``fields`` maps each array field to its row width (``0`` for a vector).
    Raises ``AssertionError`` on any mismatch; never execute a chunk that
    fails this check.
    """
    assert int(reply["step"]) == step, f"step {reply['step']}, expected {step}"
    for name, width in fields.items():
        arr = np.asarray(reply[name], dtype=np.float64)
        expected = (rows, width) if width else (rows,)
        assert arr.shape == expected, f"{name} is {arr.shape}, expected {expected}"
        assert np.isfinite(arr).all(), f"{name} contains non-finite values"
