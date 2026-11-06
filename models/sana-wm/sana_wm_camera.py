"""Plan native SANA camera poses on the application side of a step."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from sana_wm_assets import SanaWMConfig


def _load_module(name: str, path: Path) -> ModuleType:
    """Import one dependency-light upstream file without its package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SanaCameraPlanner:
    """Use the pinned upstream smoother and integrator without GPU dependencies."""

    def __init__(self, config: SanaWMConfig) -> None:
        self._config = config
        self._camera = _load_module(
            "_sana_native_camera_control",
            config.source_path / "inference_video_scripts/wm/camera_control.py",
        )
        # Torch-only upstream helper used by `prepare_camera` to relativize poses.
        self._cam_utils = _load_module(
            "_sana_native_cam_utils", config.source_path / "diffusion/utils/cam_utils.py"
        )
        self.reset(None)

    def reset(self, trajectory: np.ndarray | None) -> None:
        self._integrator = self._camera.CameraPoseIntegrator(
            math.radians(self._config.pitch_limit_degrees)
        )
        self._velocity = self._camera.VelocityState()
        self._last_controls: set[str] = set()
        self._trajectory = (
            None if trajectory is None else self._relative_trajectory(trajectory)
        )
        self._cursor = 1

    def _relative_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Relativize poses to frame 0 with upstream `prepare_camera`'s float32 ops."""
        import torch

        poses = torch.from_numpy(np.asarray(trajectory, dtype=np.float32)).float()
        first_inv = self._cam_utils.get_pose_inverse(poses[0:1]).squeeze(0)
        relative = torch.matmul(first_inv, poses[1:])
        return torch.cat([torch.eye(4).unsqueeze(0), relative], dim=0).numpy()

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
