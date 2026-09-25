"""Translate six-axis input into LingBot's native framewise camera poses."""

from __future__ import annotations

import math

import numpy as np


class CameraMotionPlanner:
    """Build native four-latent camera blocks while preserving camera pose."""

    def __init__(self, fps: float, rotation_degrees_per_second: float) -> None:
        if fps <= 0:
            raise ValueError("camera FPS must be positive")
        if rotation_degrees_per_second <= 0:
            raise ValueError("camera rotation speed must be positive")
        self._fps = fps
        self._rotation_speed = rotation_degrees_per_second
        self._current_c2w = np.eye(4, dtype=np.float64)
        self._first_chunk = True

    def reset(self, initial_c2w: np.ndarray | None = None) -> None:
        """Start a fresh camera trajectory from an optional OpenCV pose."""
        self._current_c2w = (
            np.eye(4, dtype=np.float64)
            if initial_c2w is None
            else _validate_pose(initial_c2w).copy()
        )
        self._first_chunk = True

    def plan_chunk(
        self,
        *,
        forward: float,
        strafe: float,
        vertical: float,
        pitch: float,
        yaw: float,
        roll: float,
        latent_frames: int,
        temporal_stride: int,
    ) -> np.ndarray:
        """Return framewise relative OpenCV poses for one causal chunk.

        LingBot conditions each latent frame on the transform from the previous
        camera pose. The first latent of a fresh rollout is the anchor and must
        receive identity; every later latent receives one integrated transform.
        Translation is normalized across the chunk exactly like the upstream
        ``compute_relative_poses(..., framewise=True)`` path.

        Args:
            forward: Backward-to-forward camera direction.
            strafe: Left-to-right camera direction.
            vertical: Down-to-up camera direction.
            pitch: Downward-to-upward pitch rate.
            yaw: Left-to-right yaw rate.
            roll: Counterclockwise-to-clockwise roll rate.
            latent_frames: Number of camera conditions required by the chunk.
            temporal_stride: RGB frames represented by each latent step.

        Returns:
            Contiguous float32 transforms with shape ``(latent_frames, 4, 4)``.
        """
        controls = (forward, strafe, vertical, pitch, yaw, roll)
        if any(not -1.0 <= value <= 1.0 for value in controls):
            raise ValueError("camera controls must be between -1 and 1")
        if latent_frames <= 0 or temporal_stride <= 0:
            raise ValueError("latent_frames and temporal_stride must be positive")

        translation = _bounded_vector(strafe, -vertical, forward)
        rotation = _bounded_vector(pitch, yaw, roll)
        seconds = temporal_stride / self._fps
        pitch_step, yaw_step, roll_step = np.radians(
            rotation * self._rotation_speed * seconds
        )
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = (
            _rotation_z(float(roll_step))
            @ _rotation_y(float(yaw_step))
            # OpenCV C2W: positive X rotation points the optical axis toward -Y (up).
            @ _rotation_x(float(pitch_step))
        )
        delta[:3, 3] = translation

        framewise = np.empty((latent_frames, 4, 4), dtype=np.float64)
        first_index = 0
        if self._first_chunk:
            framewise[0] = np.eye(4, dtype=np.float64)
            first_index = 1
            self._first_chunk = False
        for index in range(first_index, latent_frames):
            previous = self._current_c2w
            self._current_c2w = previous @ delta
            framewise[index] = np.linalg.inv(previous) @ self._current_c2w

        translations = framewise[:, :3, 3]
        max_norm = float(np.linalg.norm(translations, axis=-1).max())
        if max_norm > 0:
            framewise[:, :3, 3] = translations / max_norm
        return np.ascontiguousarray(framewise, dtype=np.float32)


def _bounded_vector(x: float, y: float, z: float) -> np.ndarray:
    """Return an axis vector whose magnitude does not exceed one."""
    vector = np.asarray([x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0 else vector


def _rotation_x(angle: float) -> np.ndarray:
    """Return a right-handed rotation around camera-local X."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    """Return a right-handed rotation around camera-local Y."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    """Return a right-handed rotation around camera-local Z."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _validate_pose(value: np.ndarray) -> np.ndarray:
    """Return a validated homogeneous camera-to-world pose."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"camera pose must have shape (4, 4), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("camera pose must contain finite values")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera pose must have a homogeneous final row")
    return pose
