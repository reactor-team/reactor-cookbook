import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from lyra2_camera import Lyra2CameraPlanner


def planner():
    return Lyra2CameraPlanner(translation_per_frame=.002, rotation_degrees_per_frame=.125)


def test_native_chunk_shape_and_intrinsics():
    p = planner()
    k = np.array(((500, 0, 384), (0, 500, 224), (0, 0, 1)), np.float32)
    result = p.plan_chunk(forward=1, strafe=0, vertical=0, pitch=0, yaw=0, roll=0,
                          frame_count=80, intrinsics=k)
    assert result.w2c.shape == (80, 4, 4)
    assert result.intrinsics.shape == (80, 3, 3)
    np.testing.assert_array_equal(result.intrinsics[0], k)
    assert result.c2w_last[2, 3] > 0


def test_six_axes_are_continuous_between_chunks():
    p = planner()
    k = np.eye(3, dtype=np.float32)
    first = p.plan_chunk(forward=.5, strafe=1, vertical=.25, pitch=.2, yaw=-.4, roll=.1,
                         frame_count=80, intrinsics=k)
    second = p.plan_chunk(forward=.5, strafe=1, vertical=.25, pitch=.2, yaw=-.4, roll=.1,
                          frame_count=80, intrinsics=k)
    boundary_step = np.linalg.norm(np.linalg.inv(second.w2c[0])[:3, 3] - first.c2w_last[:3, 3])
    assert 0 < boundary_step < .01
    assert np.linalg.norm(second.c2w_last[:3, 3]) > np.linalg.norm(first.c2w_last[:3, 3])


def test_rejects_non_native_chunk_size():
    import pytest
    with pytest.raises(ValueError, match="exactly 80"):
        planner().plan_chunk(forward=0, strafe=0, vertical=0, pitch=0, yaw=0, roll=0,
                             frame_count=79, intrinsics=np.eye(3))
