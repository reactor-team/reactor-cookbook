"""Integrate DreamX-World's native keyboard camera trajectory one chunk at a time."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

_FIRST_CHUNK_PIXEL_FRAMES = 9
FRAMES_PER_CHUNK = 12
_FIRST_LATENT_INDICES = (0, 1, 5)
_LATER_LATENT_INDICES = (0, 4, 8)


@dataclass(frozen=True)
class CameraChunk:
    """Carry three current latent poses and the previous chunk reference pose."""

    poses: np.ndarray
    reference_pose: np.ndarray | None


@dataclass(frozen=True)
class CameraPath:
    """Finite RGB-sampled camera-to-world trajectory in native latent layout."""

    chunks: tuple[CameraChunk, ...]
    frames: int


def load_camera_path(data: bytes, max_chunks: int) -> CameraPath:
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Camera path exceeds 2 MiB")
    payload = json.loads(data)
    if not isinstance(payload, dict) or set(payload) != {"poses"}:
        raise ValueError("Expected JSON containing only poses")
    poses = np.asarray(payload["poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N,4,4]")
    frames = len(poses)
    if frames < 9 or (frames + 3) % 12 or (frames + 3) // 12 > max_chunks:
        raise ValueError("Pose count must be 12*chunks-3 within the rollout limit")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Poses must be finite homogeneous transforms")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Pose rotations must be proper orthonormal matrices")
    indices = np.array([0] + list(range(1, frames - 3, 4)))
    rows = np.zeros((len(indices), 19), dtype=np.float32)
    rows[:, 0] = indices
    rows[:, 1:5] = [0.8, 0.8, 0.5, 0.5]
    rows[:, 7:] = np.linalg.inv(poses[indices])[:, :3].reshape(-1, 12)
    chunks = tuple(
        CameraChunk(rows[i : i + 3].copy(), None if i == 0 else rows[i - 1].copy())
        for i in range(0, len(rows), 3)
    )
    return CameraPath(chunks, frames)


class DreamXCameraController:
    """Preserve upstream camera position and rotation across native chunks."""

    def __init__(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("DreamX camera speed must be positive")
        self._speed = speed
        self.reset()

    def reset(self) -> None:
        """Return the camera to the upstream identity starting pose."""
        self._position = np.zeros(3, dtype=np.float64)
        self._rotation = np.eye(3, dtype=np.float64)
        self._frame_index = 0
        self._previous_latent_pose: np.ndarray | None = None

    def plan_chunk(
        self, pressed_keys: frozenset[str], *, first_chunk: bool
    ) -> CameraChunk:
        """Return latent-aligned poses for one native three-latent-frame chunk."""
        frame_count = _FIRST_CHUNK_PIXEL_FRAMES if first_chunk else FRAMES_PER_CHUNK
        selected_indices = (
            _FIRST_LATENT_INDICES if first_chunk else _LATER_LATENT_INDICES
        )
        pixel_poses = np.stack(
            [self._advance_one_frame(pressed_keys) for _ in range(frame_count)],
            axis=0,
        )
        latent_poses = np.ascontiguousarray(
            pixel_poses[list(selected_indices)], dtype=np.float32
        )
        reference = self._previous_latent_pose
        self._previous_latent_pose = latent_poses[-1].copy()
        return CameraChunk(poses=latent_poses, reference_pose=reference)

    def _advance_one_frame(self, pressed_keys: frozenset[str]) -> np.ndarray:
        """Apply one upstream pixel-frame camera integration step."""
        move_step = self._speed * 0.05
        rotate_step = self._speed * math.pi / 180.0

        pitch_delta = rotate_step * (
            int("i" in pressed_keys) - int("k" in pressed_keys)
        )
        yaw_delta = rotate_step * (int("l" in pressed_keys) - int("j" in pressed_keys))
        if pitch_delta != 0.0 or yaw_delta != 0.0:
            self._rotation = self._rotation @ _euler_to_rotation_matrix(
                pitch_delta,
                yaw_delta,
                0.0,
            )

        local_movement = np.zeros(3, dtype=np.float64)
        local_movement[2] = move_step * (
            int("w" in pressed_keys) - int("s" in pressed_keys)
        )
        local_movement[0] = move_step * (
            int("d" in pressed_keys) - int("a" in pressed_keys)
        )
        self._position += self._rotation @ local_movement

        world_to_camera_rotation = self._rotation.T
        world_to_camera_translation = -world_to_camera_rotation @ self._position
        world_to_camera = np.hstack(
            (world_to_camera_rotation, world_to_camera_translation.reshape(3, 1))
        )
        pose = np.asarray(
            [
                self._frame_index,
                0.8,
                0.8,
                0.5,
                0.5,
                0.0,
                0.0,
                *world_to_camera.flatten(),
            ],
            dtype=np.float32,
        )
        self._frame_index += 1
        return pose


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the rotation matrix used by upstream trajectory generation."""
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ]
    )
    rotation_y = np.asarray(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    rotation_z = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotation_z @ rotation_y @ rotation_x
