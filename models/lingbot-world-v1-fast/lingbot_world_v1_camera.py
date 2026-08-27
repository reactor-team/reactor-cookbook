"""Translate normalized six-axis controls into LingBot camera conditions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
