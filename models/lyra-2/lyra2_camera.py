"""Turn held six-axis controls into Lyra-2's native 80-pose camera chunks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraChunk:
    """World-to-camera poses and intrinsics for one native AR step.

    ``intrinsics`` None means the seed's, known only once the world is seeded.
    """

    w2c: np.ndarray
    intrinsics: np.ndarray | None


# lyra2_custom_traj_inference.py --pose_scale default (:138): the script
# multiplies every trajectory w2c translation by this factor (:82-83).
POSE_SCALE = 1.1


@dataclass(frozen=True)
class CameraPath:
    """An uploaded camera-to-world path and its optional per-frame intrinsics."""

    poses: np.ndarray
    intrinsics: np.ndarray | None


def load_camera_path(data: bytes) -> CameraPath:
    """Validate anchor plus a whole number of native 80-frame C2W chunks.

    ``intrinsics`` is optional: one 3x3 matrix or one per pose, in pixels at the
    served 832x480 resolution, like the official script's trajectory file.
    Without it, the seed's DA3-predicted intrinsics are used.
    """
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Camera path exceeds 2 MiB")
    payload = json.loads(data)
    if not isinstance(payload, dict) or not {"poses"} <= set(payload) <= {
        "poses",
        "intrinsics",
    }:
        raise ValueError("Expected JSON containing poses and optional intrinsics")
    poses = np.asarray(payload["poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N,4,4]")
    if len(poses) < 81 or (len(poses) - 1) % 80:
        raise ValueError("Pose count must be 80*chunks+1")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Poses must be finite homogeneous transforms")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Pose rotations must be proper orthonormal matrices")
    intrinsics = None
    if "intrinsics" in payload:
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
        if intrinsics.shape == (3, 3):
            intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
        if intrinsics.shape != (len(poses), 3, 3):
            raise ValueError("intrinsics must have shape [3,3] or [N,3,3]")
        if (
            not np.isfinite(intrinsics).all()
            or not (intrinsics[:, 0, 0] > 0).all()
            or not (intrinsics[:, 1, 1] > 0).all()
            or not np.array_equal(
                intrinsics[:, 2], np.broadcast_to([0.0, 0.0, 1.0], (len(poses), 3))
            )
        ):
            raise ValueError("intrinsics must be finite pinhole matrices")
        intrinsics = np.ascontiguousarray(intrinsics, dtype=np.float32)
    return CameraPath(np.ascontiguousarray(poses), intrinsics)


def camera_path_chunk(
    path: CameraPath,
    calibration: np.ndarray,
    index: int,
    intrinsics: np.ndarray | None,
    pose_scale: float = POSE_SCALE,
) -> CameraChunk:
    """Slice the fixed authored trajectory after one-time native seed calibration.

    The path's own intrinsics win over ``intrinsics`` (the seed's), and the
    float32 w2c translations are scaled as the official script scales them.
    """
    selected = path.poses[1 + 80 * index : 81 + 80 * index]
    if len(selected) != 80:
        raise ValueError("Camera path exhausted")
    w2c = np.ascontiguousarray(np.linalg.inv(calibration @ selected), dtype=np.float32)
    w2c[:, :3, 3] *= np.float32(pose_scale)
    if path.intrinsics is not None:
        chunk_intrinsics = path.intrinsics[1 + 80 * index : 81 + 80 * index].copy()
    elif intrinsics is None:
        chunk_intrinsics = None
    else:
        chunk_intrinsics = np.repeat(
            np.asarray(intrinsics, dtype=np.float32)[None], 80, axis=0
        )
    return CameraChunk(w2c=w2c, intrinsics=chunk_intrinsics)


def _rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)
    if axis == "y":
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


class Lyra2CameraPlanner:
    """Preserve an authored camera pose across Lyra-2 chunk boundaries."""

    def __init__(
        self, *, translation_per_frame: float, rotation_degrees_per_frame: float
    ):
        self.translation_step = float(translation_per_frame)
        self.rotation_step = math.radians(float(rotation_degrees_per_frame))
        self.reset()

    def reset(self, c2w: np.ndarray | None = None) -> None:
        self.c2w = (
            np.eye(4, dtype=np.float64)
            if c2w is None
            else np.asarray(c2w, dtype=np.float64).copy()
        )

    def plan_chunk(
        self,
        *,
        forward: float,
        strafe: float,
        vertical: float,
        pitch: float,
        yaw: float,
        roll: float,
        frame_count: int,
        intrinsics: np.ndarray | None,
    ) -> CameraChunk:
        if frame_count != 80:
            raise ValueError(
                "Lyra-2 requires exactly 80 camera poses per autoregressive step"
            )
        poses = []
        for _ in range(frame_count):
            local_rotation = (
                _rotation("y", yaw * self.rotation_step)
                @ _rotation("x", pitch * self.rotation_step)
                @ _rotation("z", roll * self.rotation_step)
            )
            self.c2w[:3, :3] = self.c2w[:3, :3] @ local_rotation
            local_delta = (
                np.array((strafe, -vertical, forward), dtype=np.float64)
                * self.translation_step
            )
            self.c2w[:3, 3] += self.c2w[:3, :3] @ local_delta
            poses.append(np.linalg.inv(self.c2w))
        return CameraChunk(
            w2c=np.ascontiguousarray(poses, dtype=np.float32),
            intrinsics=None
            if intrinsics is None
            else np.repeat(
                np.asarray(intrinsics, dtype=np.float32)[None], frame_count, axis=0
            ),
        )
