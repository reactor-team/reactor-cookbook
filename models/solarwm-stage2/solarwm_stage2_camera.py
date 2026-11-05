"""Translate normalized six-axis controls into SolarWM camera conditions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPath:
    poses: np.ndarray
    frames: int

    @property
    def chunks(self) -> int:
        return len(self.poses) // 3

    def chunk(self, index: int) -> np.ndarray:
        if not 0 <= index < self.chunks:
            raise ValueError("Camera path exhausted")
        return self.poses[index * 3 : (index + 1) * 3].copy()


def load_camera_path(data: bytes, max_chunks: int) -> CameraPath:
    """Sample RGB C2W poses using SolarWM's causal [0,1,5,9,...] indices."""
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Camera path exceeds 32 MiB")
    value = json.loads(data)
    if not isinstance(value, dict) or "poses" not in value:
        raise ValueError("Expected a JSON object containing poses")
    poses = np.asarray(value["poses"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [12K-3,4,4]")
    frames = len(poses)
    if frames < 9 or (frames + 3) % 12 or (frames + 3) // 12 > max_chunks:
        raise ValueError("Use exactly 12K-3 RGB poses within the rollout limit")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Camera poses must be finite homogeneous transforms")
    rotation = poses[:, :3, :3]
    if not np.allclose(
        rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Camera rotations must be right-handed and orthonormal")

    # Upstream build_camera_tokens selects the latent-aligned poses first, then
    # rebases them with _relative_c2w_fp32 (data.py), not the shared relative_c2w.
    selected = np.ascontiguousarray(poses[[0, *range(1, frames, 4)]])
    return CameraPath(relative_c2w_fp32(selected), frames)


def relative_c2w_fp32(c2w: np.ndarray) -> np.ndarray:
    """Rebase absolute C2W poses on the first with upstream's FP32 operation order."""
    w2c = _invert_se3(c2w)
    target_c2w = np.eye(4, dtype=np.float32)
    absolute_to_relative = target_c2w @ w2c[0]
    relative = np.empty_like(c2w)
    relative[0] = target_c2w
    for index in range(1, c2w.shape[0]):
        relative[index] = absolute_to_relative @ c2w[index]
    return relative


def _invert_se3(matrices: np.ndarray) -> np.ndarray:
    """Invert SE3 matrices exactly as upstream solarwm.data.camera.invert_se3."""
    result = np.zeros_like(matrices)
    rotation_t = np.swapaxes(matrices[:, :3, :3], -1, -2)
    result[:, :3, :3] = rotation_t
    result[:, :3, 3] = -np.einsum("tij,tj->ti", rotation_t, matrices[:, :3, 3])
    result[:, 3, 3] = 1
    return result


@dataclass(frozen=True)
class MotionConfig:
    """Define motion applied between SolarWM latent camera slots."""

    translation_units_per_latent: float
    rotation_degrees_per_latent: float


class CameraMotionPlanner:
    """Build native framewise camera-to-world transforms for each chunk."""

    def __init__(self, config: MotionConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        """Return the camera to the identity anchor pose."""
        self._current_c2w = np.eye(4, dtype=np.float64)
        self._chunk_index = 0

    def plan_chunk(
        self,
        *,
        strafe: float,
        vertical: float,
        forward: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> np.ndarray:
        """Return three framewise transforms and advance the absolute camera pose.

        The first causal chunk contains the anchor latent followed by two new
        latent slots, matching the VAE's 9-frame first decode. Later chunks
        contain three new slots and decode to 12 frames.
        """
        controls = (strafe, vertical, forward, pitch, yaw, roll)
        if any(not -1.0 <= value <= 1.0 for value in controls):
            raise ValueError("camera controls must be between -1 and 1")
        config = self._config
        if config.translation_units_per_latent <= 0:
            raise ValueError("translation_units_per_latent must be positive")
        if config.rotation_degrees_per_latent <= 0:
            raise ValueError("rotation_degrees_per_latent must be positive")

        delta = _camera_delta(
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            config=config,
        )
        relative = np.empty((3, 4, 4), dtype=np.float32)
        start = 1 if self._chunk_index == 0 else 0
        if start:
            relative[0] = np.eye(4, dtype=np.float32)
        for index in range(start, 3):
            self._current_c2w = self._current_c2w @ delta
            # SolarWM conditions every slot relative to the rollout's first pose.
            relative[index] = self._current_c2w
        self._chunk_index += 1
        return np.ascontiguousarray(relative)


def _camera_delta(
    *,
    strafe: float,
    vertical: float,
    forward: float,
    pitch: float,
    yaw: float,
    roll: float,
    config: MotionConfig,
) -> np.ndarray:
    """Return one local OpenCV camera transform for a latent interval."""
    translation = _normalized_vector(strafe, -vertical, forward)
    translation *= config.translation_units_per_latent
    rotation = _normalized_vector(pitch, yaw, roll)
    pitch_step, yaw_step, roll_step = np.radians(
        rotation * config.rotation_degrees_per_latent
    )
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = (
        _rotation_z(float(roll_step))
        @ _rotation_y(float(yaw_step))
        # OpenCV +Y points down: positive X rotation tilts +Z upward.
        @ _rotation_x(float(pitch_step))
    )
    delta[:3, 3] = translation
    return delta


def _normalized_vector(x: float, y: float, z: float) -> np.ndarray:
    """Return a vector whose magnitude does not exceed one."""
    vector = np.asarray([x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0:
        vector /= norm
    return vector


def _rotation_x(angle: float) -> np.ndarray:
    """Return a right-handed X-axis rotation."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    """Return a right-handed Y-axis rotation."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    """Return a right-handed Z-axis rotation."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
