import asyncio
import json
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from lyra2 import Lyra2
from lyra2_camera import (
    POSE_SCALE,
    Lyra2CameraPlanner,
    camera_path_chunk,
    load_camera_path,
)
from lyra2_model import Lyra2Result
from lyra2_types import Lyra2State
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile


def payload(frames=801, intrinsics=None):
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = 7
    poses[:, 2, 3] = np.arange(frames) * 0.01
    body = {"poses": poses.tolist()}
    if intrinsics is not None:
        body["intrinsics"] = np.asarray(intrinsics).tolist()
    return json.dumps(body).encode()


K_PATH = ((420.2020202020202, 0, 416), (0, 430.97643097643095, 240), (0, 0, 1))


def test_exact_native80_pose_slices_and_calibration():
    path = load_camera_path(payload())
    assert path.intrinsics is None
    seed = np.eye(4)
    seed[:3, 3] = [1, 2, 3]
    calibration = seed @ np.linalg.inv(path.poses[0])
    for index in range(10):
        chunk = camera_path_chunk(
            path, calibration, index, np.diag([100, 200, 1]), pose_scale=1.0
        )
        expected = np.repeat(np.eye(4)[None], 80, axis=0)
        expected[:, 0, 3] = -1
        expected[:, 1, 3] = -2
        expected[:, 2, 3] = -3 - np.arange(index * 80 + 1, index * 80 + 81) * 0.01
        np.testing.assert_allclose(chunk.w2c, expected, atol=1e-6)
        assert chunk.intrinsics.shape == (80, 3, 3)
        np.testing.assert_array_equal(chunk.intrinsics[0], np.diag([100, 200, 1]))


def test_path_intrinsics_and_official_pose_scale():
    # lyra2_custom_traj_inference.load_trajectory: float32 w2c, translation
    # *= pose_scale, and the trajectory's own K for every frame.
    path = load_camera_path(payload(intrinsics=K_PATH))
    assert path.intrinsics.shape == (801, 3, 3) and path.intrinsics.dtype == np.float32
    calibration = np.linalg.inv(path.poses[0])
    chunk = camera_path_chunk(path, calibration, 2, np.eye(3))
    raw = np.linalg.inv(calibration @ path.poses[161:241]).astype(np.float32)
    expected = raw.copy()
    expected[:, :3, 3] = raw[:, :3, 3] * np.float32(POSE_SCALE)
    np.testing.assert_array_equal(chunk.w2c, expected)
    np.testing.assert_array_equal(
        chunk.intrinsics, np.repeat(np.float32(K_PATH)[None], 80, axis=0)
    )
    per_frame = np.repeat(np.float32(K_PATH)[None], 801, axis=0)
    per_frame[:, 0, 2] = np.arange(801)
    chunk = camera_path_chunk(
        load_camera_path(payload(intrinsics=per_frame)), calibration, 1, np.eye(3)
    )
    np.testing.assert_array_equal(chunk.intrinsics[:, 0, 2], np.arange(81, 161))


@pytest.mark.parametrize(
    "intrinsics",
    [np.eye(4), np.zeros((3, 3)), np.repeat(np.eye(3)[None], 80, axis=0), [[1, 0, 0], [0, 1, 0], [0, 1, 1]]],
)
def test_invalid_intrinsics(intrinsics):
    with pytest.raises(ValueError):
        load_camera_path(payload(intrinsics=intrinsics))


@pytest.mark.parametrize("frames", [0, 1, 80, 82, 800, 802])
def test_invalid_count(frames):
    with pytest.raises(ValueError):
        load_camera_path(payload(frames))


def test_seed_with_chunk1_then_path_to_completion():
    async def run():
        app = Lyra2()
        app.state = Lyra2State()
        app.send = AsyncMock()
        app.output.flush = Mock()
        app.state._anchor = np.zeros((4, 4, 3), np.uint8)
        app.state.prompt = "scene"
        app.state._planner = Lyra2CameraPlanner(
            translation_per_frame=0.01, rotation_degrees_per_frame=0.1
        )
        await app.set_camera_path(
            UploadedFile(
                name="path.json",
                mime_type="application/json",
                data=payload(intrinsics=K_PATH),
            )
        )
        calibration = app._path_calibration.copy()
        path = load_camera_path(payload(intrinsics=K_PATH))
        np.testing.assert_array_equal(calibration, np.linalg.inv(path.poses[0]))
        with pytest.raises(CommandError):
            await app.release_camera()
        for index in range(10):
            a = await app.process_input()
            b = await app.process_input()
            assert (a.anchor is not None) == (index == 0)
            np.testing.assert_array_equal(a.w2c, b.w2c)
            np.testing.assert_array_equal(
                a.w2c, camera_path_chunk(path, calibration, index, None).w2c
            )
            np.testing.assert_array_equal(a.intrinsics[0], np.float32(K_PATH))
            if index == 0:
                np.testing.assert_array_equal(a.seed_intrinsics, np.float32(K_PATH))
            else:
                assert a.seed_intrinsics is None
            out = Lyra2Result(
                a.world_id,
                index + 1,
                np.zeros((80, 2, 2, 3), np.uint8),
                None,
                np.float32(K_PATH) if index == 0 else None,
            )
            assert await app.process_output(StepOutcome(result=out, elapsed=0.1))
        assert app._state().camera_path_complete
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(42)
        again = await app.process_input()
        assert again.anchor is not None
        np.testing.assert_array_equal(again.w2c, camera_path_chunk(path, calibration, 0, None).w2c)
        np.testing.assert_array_equal(again.seed_intrinsics, np.float32(K_PATH))

    asyncio.run(run())


def test_path_without_intrinsics_uses_seed_k_from_chunk_two():
    path = load_camera_path(payload())
    calibration = np.linalg.inv(path.poses[0])
    assert camera_path_chunk(path, calibration, 0, None).intrinsics is None
    k = np.diag([3.0, 4.0, 1.0])
    np.testing.assert_array_equal(camera_path_chunk(path, calibration, 1, k).intrinsics[5], k)
