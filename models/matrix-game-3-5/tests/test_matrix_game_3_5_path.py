import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from matrix_game_3_5 import MatrixGame35
from matrix_game_3_5_camera import CameraMotionPlanner, MotionConfig, load_camera_path
from matrix_game_3_5_model import MatrixGame35Result
from matrix_game_3_5_types import MatrixGame35State
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile


def data(frames=121):
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = 2
    poses[:, 2, 3] = np.arange(frames) * 0.1
    return json.dumps({"poses": poses.tolist()}).encode()


def test_calibration_shared_anchor_and_exact_rgb_slices():
    anchor = np.eye(4)
    anchor[:3, 3] = [3, 4, 5]
    path = load_camera_path(data(), 10, anchor)
    assert len(path.chunks) == 10
    np.testing.assert_array_equal(path.chunks[0][0], anchor)
    full = np.concatenate([chunk[1:] for chunk in path.chunks])
    np.testing.assert_allclose(full[:, 2, 3], 5 + np.arange(1, 121) * 0.1, atol=1e-6)
    for prev, current in zip(path.chunks, path.chunks[1:]):
        np.testing.assert_array_equal(prev[-1], current[0])
    np.testing.assert_array_equal(path.poses[1:], full)
    np.testing.assert_array_equal(path.poses[0], anchor)


@pytest.mark.parametrize("frames", [0, 1, 12, 14, 120, 122, 133])
def test_invalid_length(frames):
    with pytest.raises(ValueError):
        load_camera_path(data(frames), 10, np.eye(4))


def test_invalid_rotation():
    payload = json.loads(data())
    payload["poses"][1][0][0] = 2
    with pytest.raises(ValueError):
        load_camera_path(json.dumps(payload).encode(), 10, np.eye(4))


def test_path_finite_retry_reset_and_axis_refusal():
    async def run():
        app = MatrixGame35()
        app.state = MatrixGame35State()
        app._config = SimpleNamespace(max_chunks=100)
        app._camera_anchor = np.eye(4)
        app._planner = CameraMotionPlanner(np.eye(4), MotionConfig(16, 1.5, 45))
        app._selected_input = Path("upload.jpg")
        app.state.prompt = "A street"
        app.send = AsyncMock()
        app.output.flush = Mock()
        await app.set_camera_path(
            UploadedFile(name="path.json", mime_type="application/json", data=data())
        )
        with pytest.raises(CommandError):
            app.set_forward(1)
        for index in range(10):
            a = await app.process_input()
            b = await app.process_input()
            np.testing.assert_array_equal(a.trajectory, b.trajectory)
            assert (a.anchor is not None) == (index == 0)
            if a.anchor is not None:
                assert a.anchor.camera is app._camera_path.poses
            out = MatrixGame35Result(
                np.zeros((12, 2, 2, 3), np.uint8), a.world_id, index + 1, False
            )
            await app.process_output(StepOutcome(result=out, elapsed=0.01))
        assert (
            app._state_update().limit_reached and app._state_update().max_chunks == 10
        )
        with pytest.raises(ApplicationError):
            await app.process_input()
        app.reset(42)
        assert (await app.process_input()).anchor is not None

    asyncio.run(run())
