"""Translate normalized six-axis input into EVOKE camera trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPath:
    """Finite absolute camera trajectory in the native OpenCV convention.

    ``poses`` keeps the uploaded camera-to-world matrices verbatim, 36 per
    native chunk. The worker conditions them with EVOKE's released pose-track
    recipe before the rollout starts, as the official launcher does for a
    pose file.
    """

    poses: np.ndarray

    @property
    def chunk_count(self) -> int:
        """Return the native 36-frame chunks the path spans."""
        return len(self.poses) // 36


def load_camera_path(data: bytes, max_chunks: int) -> CameraPath:
    """Validate a pixel-rate pose sequence spanning whole native 36-pose windows."""
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Camera path exceeds 16 MiB")
    payload = json.loads(data)
    if not isinstance(payload, dict) or set(payload) != {"poses"}:
        raise ValueError("Camera path must contain only poses")
    poses = np.asarray(payload["poses"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N, 4, 4]")
    if not 36 <= len(poses) <= max_chunks * 36 or len(poses) % 36:
        raise ValueError("A path needs 36 * chunks poses within rollout capacity")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Camera poses must be finite homogeneous transforms")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Camera poses require proper rotations")
    return CameraPath(np.ascontiguousarray(poses))


@dataclass(frozen=True)
class MotionConfig:
    """Define full-scale camera rates used to synthesize a pose track."""

    fps: float
    translation_units_per_second: float
    rotation_degrees_per_second: float


class CameraMotionPlanner:
    """Build continuous camera-to-world chunks in EVOKE's OpenCV convention."""

    def __init__(self, config: MotionConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        """Return the relative camera timeline to its identity anchor."""
        self._current_c2w = np.eye(4, dtype=np.float64)
        self._first_chunk = True

    def plan_chunk(
        self,
        *,
        strafe: float,
        vertical: float,
        forward: float,
        pitch: float,
        yaw: float,
        roll: float,
        frame_count: int,
    ) -> np.ndarray:
        """Return one native pose chunk and advance the relative camera timeline."""
        generated_count = frame_count - 1 if self._first_chunk else frame_count
        generated = _plan_motion(
            self._current_c2w,
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            frame_count=generated_count,
            config=self._config,
        )
        if self._first_chunk:
            poses = np.concatenate([self._current_c2w[None], generated], axis=0)
            self._first_chunk = False
        else:
            poses = generated
        self._current_c2w = poses[-1].astype(np.float64, copy=True)
        return np.ascontiguousarray(poses, dtype=np.float32)


def _plan_motion(
    start_c2w: np.ndarray,
    *,
    strafe: float,
    vertical: float,
    forward: float,
    pitch: float,
    yaw: float,
    roll: float,
    frame_count: int,
    config: MotionConfig,
) -> np.ndarray:
    """Integrate camera-local velocity into absolute camera-to-world matrices."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    values = (strafe, vertical, forward, pitch, yaw, roll)
    if any(not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("camera axes must be between -1 and 1")
    if config.fps <= 0:
        raise ValueError("motion fps must be positive")

    pose = np.asarray(start_c2w, dtype=np.float64).copy()
    translation = _normalized_vector(strafe, -vertical, forward)
    translation *= config.translation_units_per_second / config.fps
    rotation = _normalized_vector(pitch, yaw, roll)
    pitch_step, yaw_step, roll_step = np.radians(
        rotation * config.rotation_degrees_per_second / config.fps
    )
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = (
        _rotation_z(float(roll_step))
        @ _rotation_y(float(yaw_step))
        # OpenCV C2W: positive X rotation points the optical axis toward -Y (up).
        @ _rotation_x(float(pitch_step))
    )
    delta[:3, 3] = translation

    poses = np.empty((frame_count, 4, 4), dtype=np.float32)
    for index in range(frame_count):
        pose = pose @ delta
        poses[index] = pose
    return poses


def _normalized_vector(x: float, y: float, z: float) -> np.ndarray:
    """Return an axis vector whose magnitude does not exceed one."""
    value = np.asarray([x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0 else value


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
