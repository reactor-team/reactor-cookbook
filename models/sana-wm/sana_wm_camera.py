"""Plan native SANA camera poses on the application side of a step."""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
from sana_wm_assets import SanaWMConfig


class SanaCameraPlanner:
    """Use the pinned upstream smoother and integrator without GPU dependencies."""

    def __init__(self, config: SanaWMConfig) -> None:
        self._config = config
        name = "_sana_native_camera_control"
        path = config.source_path / "inference_video_scripts/wm/camera_control.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import camera controls from {path}")
        self._camera = importlib.util.module_from_spec(spec)
        sys.modules[name] = self._camera
        spec.loader.exec_module(self._camera)
        self.reset(None)

    def reset(self, trajectory: np.ndarray | None) -> None:
        self._integrator = self._camera.CameraPoseIntegrator(
            math.radians(self._config.pitch_limit_degrees)
        )
        self._velocity = self._camera.VelocityState()
        self._last_controls: set[str] = set()
        self._trajectory = (
            None
            if trajectory is None
            else np.matmul(
                np.linalg.inv(trajectory[0]).astype(np.float32)[None], trajectory
            ).astype(np.float32)
        )
        self._cursor = 1

    def plan_chunk(self, controls: set[str]) -> np.ndarray | None:
        if self._trajectory is not None:
            end = self._cursor + 24
            if end > len(self._trajectory):
                return None
            poses = self._trajectory[self._cursor : end].copy()
            self._cursor = end
            return poses
        target = self._camera.controls_to_target_velocity(
            controls,
            translation_speed=self._config.translation_speed,
            rotation_speed_rad=math.radians(self._config.rotation_speed_degrees),
        )
        poses = []
        for _ in range(24):
            if controls - self._last_controls:
                self._velocity.snap_to(target)
            else:
                self._velocity.step_toward(target, 1.0 / 16)
            self._last_controls = set(controls)
            poses.append(self._integrator.step(self._velocity).astype(np.float32))
        return np.stack(poses)
