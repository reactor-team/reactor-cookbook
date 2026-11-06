"""Translate normalized six-axis controls into LingBot camera conditions."""

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
    if not 9 <= len(poses) <= ((max_chunks + 1) * chunk_latents - 1) * 4:
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


@dataclass(frozen=True)
class MotionConfig:
    """Define motion applied between LingBot latent camera slots."""

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
            previous = self._current_c2w
            self._current_c2w = previous @ delta
            relative[index] = np.linalg.inv(previous) @ self._current_c2w
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
        @ _rotation_x(float(-pitch_step))
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
