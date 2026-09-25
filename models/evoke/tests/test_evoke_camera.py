"""Check public camera directions through OpenCV poses and landmark projection."""

import numpy as np
import pytest
from evoke_camera import CameraMotionPlanner, MotionConfig


def planner():
    return CameraMotionPlanner(MotionConfig(24.0, 1.0, 6.0))


def advance(camera, **axes):
    controls = {
        "forward": 0.0,
        "strafe": 0.0,
        "vertical": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }
    controls.update(axes)
    return camera.plan_chunk(**controls, frame_count=36)


def endpoint(*chunks):
    return chunks[-1][-1]


@pytest.mark.parametrize("pitch", [-1.0, -0.05, 0.05, 1.0])
def test_pitch_direction_and_landmark_projection(pitch):
    camera = planner()
    first = advance(camera, pitch=pitch)
    second = advance(camera, pitch=pitch)
    for pose in (endpoint(first), endpoint(first, second)):
        # OpenCV +Y is down: looking up points the optical axis toward -Y.
        assert pitch * pose[1, 2] < 0
        # A fixed landmark in front moves down in the image when looking up.
        point = np.linalg.inv(pose) @ np.array([0.0, 0.0, 10.0, 1.0])
        assert point[2] > 0
        assert pitch * point[1] / point[2] > 0
        np.testing.assert_allclose(pose[:3, 3], 0, atol=1e-6)
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        assert np.linalg.det(pose[:3, :3]) == pytest.approx(1, abs=1e-6)


def test_opposite_pitch_cancels_after_anchor():
    camera = planner()
    anchor = advance(camera)
    up = advance(camera, pitch=0.2)
    down = advance(camera, pitch=-0.2)
    np.testing.assert_allclose(endpoint(anchor, up, down), np.eye(4), atol=1e-6)


@pytest.mark.parametrize("axis,index", [("forward", 2), ("strafe", 0), ("vertical", 1)])
def test_translation_axes_keep_their_native_sign(axis, index):
    pose = endpoint(advance(planner(), **{axis: 0.2}))
    assert pose[index, 3] * (-1 if axis == "vertical" else 1) > 0
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-6)


def test_positive_yaw_turns_right_without_pitch():
    pose = endpoint(advance(planner(), yaw=0.2))
    assert pose[0, 2] > 0
    assert pose[1, 2] == pytest.approx(0, abs=1e-6)


def test_positive_roll_keeps_clockwise_camera_convention():
    pose = endpoint(advance(planner(), roll=0.2))
    assert pose[1, 0] > 0
    np.testing.assert_allclose(pose[:3, 2], [0, 0, 1], atol=1e-6)
