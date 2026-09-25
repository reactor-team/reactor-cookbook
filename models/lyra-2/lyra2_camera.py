"""Turn held six-axis controls into Lyra-2's native 80-pose camera chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraChunk:
    """World-to-camera poses and intrinsics for one native AR step."""

    w2c: np.ndarray
    intrinsics: np.ndarray
    c2w_last: np.ndarray


def _rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)
    if axis == "y":
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


class Lyra2CameraPlanner:
    """Preserve an authored camera pose across Lyra-2 chunk boundaries."""

    def __init__(self, *, translation_per_frame: float, rotation_degrees_per_frame: float):
        self.translation_step = float(translation_per_frame)
        self.rotation_step = math.radians(float(rotation_degrees_per_frame))
        self.reset()

    def reset(self, c2w: np.ndarray | None = None) -> None:
        self.c2w = np.eye(4, dtype=np.float64) if c2w is None else np.asarray(c2w, dtype=np.float64).copy()

    def plan_chunk(self, *, forward: float, strafe: float, vertical: float, pitch: float,
                   yaw: float, roll: float, frame_count: int, intrinsics: np.ndarray) -> CameraChunk:
        if frame_count != 80:
            raise ValueError("Lyra-2 requires exactly 80 camera poses per autoregressive step")
        poses = []
        for _ in range(frame_count):
            local_rotation = (_rotation("y", yaw * self.rotation_step)
                              @ _rotation("x", pitch * self.rotation_step)
                              @ _rotation("z", roll * self.rotation_step))
            self.c2w[:3, :3] = self.c2w[:3, :3] @ local_rotation
            local_delta = np.array((strafe, -vertical, forward), dtype=np.float64) * self.translation_step
            self.c2w[:3, 3] += self.c2w[:3, :3] @ local_delta
            poses.append(np.linalg.inv(self.c2w))
        return CameraChunk(
            w2c=np.ascontiguousarray(poses, dtype=np.float32),
            intrinsics=np.repeat(np.asarray(intrinsics, dtype=np.float32)[None], frame_count, axis=0),
            c2w_last=self.c2w.astype(np.float32).copy(),
        )
