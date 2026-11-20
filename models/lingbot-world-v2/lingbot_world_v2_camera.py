"""Translate six-axis input into LingBot's native framewise camera poses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPath:
    """A finite rollout's globally normalized camera conditions."""

    chunks: np.ndarray
    intrinsics: np.ndarray
    source_frames: int

    @property
    def output_frames(self) -> int:
        return len(self.chunks) * self.chunks.shape[1] * 4 - 3


def load_camera_path(data: bytes, *, max_chunks: int, chunk_latents: int) -> CameraPath:
    """Validate OpenCV C2W JSON and reproduce native whole-sequence resampling."""
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    if not data or len(data) > 2 * 1024 * 1024:
        raise ValueError("Camera path must be nonempty JSON no larger than 2 MiB")
    try:
        document = json.loads(data)
        if not isinstance(document, dict) or set(document) != {"poses", "intrinsics"}:
            raise ValueError("Expected exactly poses and intrinsics")
        poses = np.asarray(document["poses"], dtype=np.float32)
        intrinsics = np.asarray(document["intrinsics"], dtype=np.float32)
    except (TypeError, KeyError, OverflowError, UnicodeError) as error:
        raise ValueError("Invalid camera path JSON") from error
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [RGB frames, 4, 4]")
    if not 13 <= len(poses) <= ((max_chunks + 1) * chunk_latents - 1) * 4:
        raise ValueError(
            "Camera path length exceeds the supported rollout or is shorter than one chunk"
        )
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3], [0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Poses must be finite homogeneous transforms")
    rotation = poses[:, :3, :3]
    if not np.allclose(
        rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Pose rotation blocks must be proper orthonormal rotations")
    if (
        intrinsics.shape != (4,)
        or not np.isfinite(intrinsics).all()
        or np.any(intrinsics[:2] <= 0)
    ):
        raise ValueError(
            "intrinsics must be finite [fx, fy, cx, cy] with positive focal lengths"
        )
    source_frames = len(poses)
    # Match native generate(): align RGB count, floor latent count to whole
    # chunks, then sample uniformly over the aligned original pose sequence.
    poses = poses[: ((len(poses) - 1) // 4) * 4 + 1]
    latent_count = ((len(poses) - 1) // 4 + 1) // chunk_latents * chunk_latents
    source = np.arange(len(poses), dtype=np.float64)
    target = np.linspace(0, len(poses) - 1, latent_count)
    absolute = np.tile(np.eye(4), (latent_count, 1, 1))
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat().copy()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], quaternions[index - 1]) < 0:
            quaternions[index] *= -1
    absolute[:, :3, :3] = Slerp(source, Rotation.from_quat(quaternions))(
        target
    ).as_matrix()
    absolute[:, :3, 3] = interp1d(
        source,
        poses[:, :3, 3],
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(target)
    absolute = absolute.astype(np.float32)

    def inverse(transforms):
        result = np.tile(np.eye(4, dtype=np.float32), (len(transforms), 1, 1))
        result[:, :3, :3] = transforms[:, :3, :3].transpose(0, 2, 1)
        result[:, :3, 3:] = -np.einsum(
            "bij,bjk->bik", result[:, :3, :3], transforms[:, :3, 3:], optimize=False
        )
        return result

    # Preserve native Torch CPU bmm accumulation before BF16 ray embedding.
    relative = np.einsum(
        "bij,bjk->bik", inverse(absolute[:1]), absolute, optimize=False
    )
    relative[0] = np.eye(4, dtype=np.float32)
    relative[1:] = np.einsum(
        "bij,bjk->bik", inverse(relative[:-1]), relative[1:], optimize=False
    )
    maximum = np.linalg.norm(relative[:, :3, 3], axis=-1).max()
    if maximum > 0:
        relative[:, :3, 3] /= maximum
    chunks = np.ascontiguousarray(relative.reshape(-1, chunk_latents, 4, 4))
    return CameraPath(chunks, intrinsics, source_frames)


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
