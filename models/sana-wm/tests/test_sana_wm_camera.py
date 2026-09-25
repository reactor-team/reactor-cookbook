"""Audit the real pinned upstream camera path without loading model weights.

Set SANA_WM_TEST_SOURCE to the SANA checkout selected by sana_wm.yaml.
These integration tests skip explicitly when that optional checkout is absent.
"""

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sana_wm_camera import SanaCameraPlanner


@pytest.fixture
def camera_control():
    source = os.environ.get("SANA_WM_TEST_SOURCE")
    if not source:
        pytest.skip("Set SANA_WM_TEST_SOURCE to the pinned SANA checkout")
    path = Path(source) / "inference_video_scripts/wm/camera_control.py"
    spec = importlib.util.spec_from_file_location("sana_wm_test_camera_control", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def backend(camera_control):
    model = object.__new__(SanaCameraPlanner)
    model._camera = camera_control
    model._trajectory = None
    model._config = SimpleNamespace(translation_speed=0.025, rotation_speed_degrees=0.6)
    model._integrator = camera_control.CameraPoseIntegrator(math.radians(60))
    model._velocity = camera_control.VelocityState()
    model._last_controls = set()
    return model


@pytest.mark.parametrize("control,sign", [("pitch_up", 1), ("pitch_down", -1)])
def test_public_pitch_tokens_reach_correct_native_pose(camera_control, control, sign):
    model = backend(camera_control)
    for _ in range(2):
        poses = model.plan_chunk({control})
        assert poses.shape == (24, 4, 4)
        pose = poses[-1]
        assert sign * pose[1, 2] < 0
        point = np.linalg.inv(pose) @ np.array([0, 0, 10, 1])
        assert point[2] > 0
        assert sign * point[1] / point[2] > 0


@pytest.mark.parametrize("control,sign", [("pitch_up", 1), ("pitch_down", -1)])
def test_pitch_limit_and_release_keep_native_smoothing(camera_control, control, sign):
    model = backend(camera_control)
    for _ in range(8):
        model.plan_chunk({control})
    assert model._integrator.pitch == pytest.approx(sign * math.radians(60))
    velocity = model._velocity.pitch
    model.plan_chunk(set())
    assert 0 < model._velocity.pitch / velocity < 1
    assert model._integrator.pitch == pytest.approx(sign * math.radians(60))


@pytest.mark.parametrize("control,sign", [("yaw_left", -1), ("yaw_right", 1)])
def test_yaw_tokens_are_not_pitch(camera_control, control, sign):
    model = backend(camera_control)
    poses = model.plan_chunk({control})
    assert sign * poses[-1][0, 2] > 0
    assert poses[-1][1, 2] == pytest.approx(0)


def test_opposite_pitch_controls_cancel(camera_control):
    model = backend(camera_control)
    poses = model.plan_chunk({"pitch_up", "pitch_down"})
    np.testing.assert_allclose(poses[-1], np.eye(4))


def test_ten_chunks_match_native_smoother_and_integrator(camera_control):
    model = backend(camera_control)
    integrator = camera_control.CameraPoseIntegrator(math.radians(60))
    velocity = camera_control.VelocityState()
    last = set()
    for controls in [
        {"forward"},
        set(),
        {"yaw_left"},
        {"forward", "pitch_up"},
        {"pitch_down"},
        {"back"},
        {"strafe_left"},
        {"strafe_right"},
        {"yaw_right"},
        set(),
    ]:
        target = camera_control.controls_to_target_velocity(
            controls, translation_speed=0.025, rotation_speed_rad=math.radians(0.6)
        )
        expected = []
        for _ in range(24):
            if controls - last:
                velocity.snap_to(target)
            else:
                velocity.step_toward(target, 1 / 16)
            last = set(controls)
            expected.append(integrator.step(velocity).astype(np.float32))
        np.testing.assert_array_equal(model.plan_chunk(controls), np.stack(expected))


def test_finite_trajectory_normalization_and_end(camera_control):
    model = backend(camera_control)
    model._config.pitch_limit_degrees = 60
    trajectory = np.tile(np.eye(4, dtype=np.float32), (50, 1, 1))
    trajectory[:, 0, 3] = np.arange(50, dtype=np.float32) + 7
    model.reset(trajectory)
    expected = np.linalg.inv(trajectory[0])[None] @ trajectory
    np.testing.assert_array_equal(model.plan_chunk(set()), expected[1:25])
    np.testing.assert_array_equal(model.plan_chunk(set()), expected[25:49])
    assert model.plan_chunk(set()) is None
