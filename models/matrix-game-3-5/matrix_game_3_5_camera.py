"""Translate normalized six-axis input into Matrix-Game-3.5 camera poses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    """Define the full-scale camera rates used to synthesize a trajectory."""

    fps: float
    translation_meters_per_second: float
    rotation_degrees_per_second: float


@dataclass(frozen=True)
class CameraPath:
    """Finite native-calibrated camera path and its blocks with shared anchors.

    ``poses`` is the whole calibrated path. A rollout started from it gives
    the worker the full trajectory up front, as the official ``--camera``
    input does, so every pose the rollout reads matches that run.
    """

    chunks: tuple[np.ndarray, ...]
    poses: np.ndarray


def load_camera_path(data: bytes, max_chunks: int, anchor: np.ndarray) -> CameraPath:
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Camera path exceeds 2 MiB")
    payload = json.loads(data)
    if not isinstance(payload, dict) or set(payload) != {"poses"}:
        raise ValueError("Expected JSON containing only poses")
    poses = np.asarray(payload["poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N,4,4]")
    if len(poses) < 13 or (len(poses) - 1) % 12 or (len(poses) - 1) // 12 > max_chunks:
        raise ValueError("Pose count must be 12*chunks+1 within the rollout limit")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Poses must be finite homogeneous transforms")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Pose rotations must be proper orthonormal matrices")
    # The worker keeps the configured first-frame camera as its image anchor.
    calibrated = _validate_pose(anchor) @ np.linalg.inv(poses[0]) @ poses
    calibrated = np.ascontiguousarray(calibrated, dtype=np.float32)
    return CameraPath(
        tuple(calibrated[i : i + 13].copy() for i in range(0, len(poses) - 1, 12)),
        calibrated,
    )


class CameraMotionPlanner:
    """Build smooth camera-to-world trajectories from normalized axes."""

    def __init__(self, initial_c2w: np.ndarray, config: MotionConfig) -> None:
        self._initial_c2w = _validate_pose(initial_c2w).copy()
        self._current_c2w = self._initial_c2w.copy()
        self._config = config

    def reset(self) -> None:
        """Return the camera to its configured anchor pose."""
        self._current_c2w = self._initial_c2w.copy()

    def plan_block(
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
        """Return anchor plus generated poses and advance the current pose.

        Args:
            strafe: Normalized left-to-right translation velocity.
            vertical: Normalized down-to-up translation velocity.
            forward: Normalized backward-to-forward translation velocity.
            pitch: Normalized down-to-up pitch velocity.
            yaw: Normalized left-to-right yaw velocity.
            roll: Normalized counterclockwise-to-clockwise roll velocity.
            frame_count: Number of new RGB camera slots required by Matrix.

        Returns:
            A ``(frame_count + 1, 4, 4)`` float32 camera-to-world trajectory.
            Element zero is the current anchor pose.
        """
        generated = plan_camera_motion(
            self._current_c2w,
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            frame_count=frame_count,
            config=self._config,
        )
        trajectory = np.concatenate([self._current_c2w[None], generated], axis=0)
        self._current_c2w = generated[-1].copy()
        return np.ascontiguousarray(trajectory, dtype=np.float32)


def plan_camera_motion(
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
    """Integrate normalized six-axis input into camera-to-world matrices.

    Matrix uses OpenCV camera axes: positive X points right, positive Y points
    down, and positive Z points forward. The public vertical axis is positive
    upward, so it is negated when mapped to camera-local Y. Translation and
    rotation are normalized independently before integration.

    Args:
        start_c2w: Current camera-to-world transform.
        strafe: Normalized left-to-right translation velocity in ``[-1, 1]``.
        vertical: Normalized down-to-up translation velocity in ``[-1, 1]``.
        forward: Normalized backward-to-forward velocity in ``[-1, 1]``.
        pitch: Normalized down-to-up pitch velocity in ``[-1, 1]``.
        yaw: Normalized left-to-right yaw velocity in ``[-1, 1]``.
        roll: Normalized counterclockwise-to-clockwise roll in ``[-1, 1]``.
        frame_count: Number of poses to generate after ``start_c2w``.
        config: Motion rate and output cadence.

    Returns:
        Contiguous float32 transforms with shape ``(frame_count, 4, 4)``.

    Raises:
        ValueError: If the pose, controls, frame count, or rates are invalid.
    """
    pose = _validate_pose(start_c2w).copy()
    controls = {
        "strafe": strafe,
        "vertical": vertical,
        "forward": forward,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
    for name, value in controls.items():
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between -1 and 1")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if config.fps <= 0:
        raise ValueError("motion fps must be positive")
    if config.translation_meters_per_second < 0:
        raise ValueError("translation speed must be non-negative")
    if config.rotation_degrees_per_second < 0:
        raise ValueError("rotation speed must be non-negative")

    translation = _normalized_vector(strafe, -vertical, forward)
    translation *= config.translation_meters_per_second / config.fps

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
    return np.ascontiguousarray(poses)


def _normalized_vector(x: float, y: float, z: float) -> np.ndarray:
    """Return an axis vector whose magnitude does not exceed one."""
    vector = np.asarray([x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0:
        vector /= norm
    return vector


def _rotation_x(angle: float) -> np.ndarray:
    """Return a right-handed rotation about the local camera X axis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    """Return a right-handed rotation about the local camera Y axis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    """Return a right-handed rotation about the local camera Z axis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _validate_pose(value: np.ndarray) -> np.ndarray:
    """Return a validated float64 camera-to-world transform."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"camera pose must have shape (4, 4), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("camera pose must contain only finite values")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera pose must have a homogeneous final row")
    return pose
