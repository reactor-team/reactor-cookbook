"""Integrate Echo-WM's native pure-camera motion across interactive chunks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    """Hold the native camera integration constants."""

    fps: float
    translation_speed: float
    rotation_speed_degrees: float
    pitch_speed_degrees: float
    pitch_limit_degrees: float


@dataclass(frozen=True)
class CameraChunk:
    """Carry the three camera poses aligned with one causal latent block."""

    latent_poses: np.ndarray


@dataclass(frozen=True)
class CameraPath:
    """Finite RGB-rate camera-to-world path, sampled in native temporal units.

    ``poses`` keeps the uploaded per-frame float32 matrices so the backend can
    anchor and calibrate them with upstream's own CUDA arithmetic; ``chunks``
    holds the anchored latent-rate poses of each native step.
    """

    poses: np.ndarray
    chunks: np.ndarray

    @property
    def frames(self) -> int:
        return len(self.chunks) * 24 + 1


def load_camera_path(data: bytes, max_chunks: int) -> CameraPath:
    """Anchor RGB camera poses and select every eighth pose as upstream does."""
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Camera path exceeds 4 MiB")
    payload = json.loads(data)
    if not isinstance(payload, dict) or set(payload) != {"poses"}:
        raise ValueError("Camera path must contain only poses")
    poses = np.asarray(payload["poses"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N, 4, 4]")
    if not 25 <= len(poses) <= max_chunks * 24 + 1 or (len(poses) - 1) % 24:
        raise ValueError("A path needs 24 * chunks + 1 poses within rollout capacity")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Camera poses must be finite homogeneous transforms")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Camera poses require proper rotations")
    # Calibration remains in the backend, shared with native interactive controls.
    anchored = np.linalg.inv(poses[0]) @ poses[::8]
    return CameraPath(
        poses=np.ascontiguousarray(poses),
        chunks=np.ascontiguousarray(anchored[1:].reshape(-1, 3, 4, 4)),
    )


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


class EchoCameraPlanner:
    """Preserve Echo-WM camera pose, inertia, and pitch across chunks."""

    def __init__(self, config: MotionConfig) -> None:
        if config.fps <= 0 or config.translation_speed <= 0:
            raise ValueError("Echo-WM camera rates must be positive")
        self._config = config
        self.reset()

    def reset(self) -> None:
        """Return camera position, rotation, and inertia to upstream defaults."""
        self._pose = np.eye(4, dtype=np.float64)
        self._pitch = 0.0
        self._velocity = np.zeros(4, dtype=np.float64)
        self._active_axes: frozenset[str] = frozenset()

    def plan_chunk(
        self,
        *,
        forward: float,
        strafe: float,
        pitch: float,
        yaw: float,
        frame_count: int,
    ) -> CameraChunk:
        """Advance native camera integration and return latent-aligned poses."""
        if frame_count <= 0 or frame_count % 8:
            raise ValueError(
                "Echo-WM camera chunks must be a positive multiple of 8 frames"
            )
        pixel_poses = [
            self._advance_one_frame(
                forward=forward,
                strafe=strafe,
                pitch=pitch,
                yaw=yaw,
            )
            for _ in range(frame_count)
        ]
        indices = np.arange(7, frame_count, 8)
        return CameraChunk(
            latent_poses=np.ascontiguousarray(
                np.stack(pixel_poses)[indices], dtype=np.float32
            )
        )

    def _advance_one_frame(
        self,
        *,
        forward: float,
        strafe: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        config = self._config
        target = np.asarray(
            [
                forward * config.translation_speed,
                strafe * config.translation_speed,
                yaw * math.radians(config.rotation_speed_degrees),
                pitch * math.radians(config.pitch_speed_degrees),
            ],
            dtype=np.float64,
        )
        active = frozenset(
            name
            for name, value in zip(
                ("forward", "strafe", "yaw", "pitch"),
                (forward, strafe, yaw, pitch),
                strict=True,
            )
            if value != 0.0
        )
        sign_changed = any(
            np.sign(target[index]) != np.sign(self._velocity[index])
            and target[index] != 0.0
            for index in range(4)
        )
        if active - self._active_axes or sign_changed:
            self._velocity = target
        else:
            time_constant = 0.45 if np.any(target) else 1.0
            blend = 1.0 - math.exp(-(1.0 / config.fps) / time_constant)
            self._velocity += (target - self._velocity) * blend
        self._active_axes = active

        next_pitch = float(
            np.clip(
                self._pitch + self._velocity[3],
                -math.radians(config.pitch_limit_degrees),
                math.radians(config.pitch_limit_degrees),
            )
        )
        pitch_step = next_pitch - self._pitch
        self._pitch = next_pitch
        rotation = (
            _rotation_y(self._velocity[2])
            @ self._pose[:3, :3]
            @ _rotation_x(pitch_step)
        )
        forward_axis = rotation[:, 2].copy()
        right_axis = rotation[:, 0].copy()
        forward_axis[1] = 0.0
        right_axis[1] = 0.0
        forward_axis /= max(float(np.linalg.norm(forward_axis)), 1e-6)
        right_axis /= max(float(np.linalg.norm(right_axis)), 1e-6)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = (
            self._pose[:3, 3]
            + forward_axis * self._velocity[0]
            + right_axis * self._velocity[1]
        )
        self._pose = pose
        return pose.copy()
