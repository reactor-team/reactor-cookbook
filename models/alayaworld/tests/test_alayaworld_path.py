"""Finite native camera windows advance only after successful outputs."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from alayaworld_camera import load_camera_path
from alayaworld_model import AlayaResult
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from test_alayaworld import app, setup_result  # noqa: F401 - reuse fixture


def upload(chunks=12):
    poses = np.tile(np.eye(4), (chunks * 32 + 8, 1, 1))
    poses[:, 0, 3] = np.arange(len(poses)) * 0.01 + 2
    return UploadedFile(
        name="path.json",
        mime_type="application/json",
        data=json.dumps({"poses": poses.tolist()}).encode(),
    )


def test_native_windows_and_initial_pose():
    trajectory = upload()
    poses = np.asarray(json.loads(trajectory.data)["poses"], dtype=np.float32)
    path = load_camera_path(trajectory.data, 12)
    anchor = np.eye(4, dtype=np.float32)
    anchor[:3, 3] = [7, 3, 4]
    expected = anchor @ np.linalg.inv(poses[0]) @ poses
    actual = np.concatenate(
        [path.window(i, anchor, 39 if i == 0 else 32) for i in range(12)]
    )
    np.testing.assert_allclose(actual, expected[1:])
    with pytest.raises(ValueError, match="window changed"):
        path.window(0, anchor, 32)


def test_twelve_chunks_stop_and_reset(app, monkeypatch):  # noqa: F811 - imported fixture
    app._selected_input = Path("anchor")
    app.state.prompt = "Scene"
    calls = []

    def generate(value):
        calls.append(value)
        if value.image is not None:
            return setup_result(value.world_id)
        return AlayaResult(
            value.world_id,
            np.zeros((32, 8, 8, 3), np.uint8),
            len(calls) - 1,
            value.prompt,
            32,
        )

    monkeypatch.setattr(app.engine, "generate", generate)

    async def run():
        reply = await app.set_camera_path(upload())
        assert (reply.chunks, reply.frames) == (12, 384)
        with pytest.raises(CommandError):
            await app.set_yaw(1)
        assert (
            await app.process_output(
                StepOutcome(result=app.generate(await app.process_input()))
            )
            is None
        )
        for index in range(12):
            value = await app.process_input()
            np.testing.assert_array_equal(
                value.trajectory, (await app.process_input()).trajectory
            )
            output = await app.process_output(StepOutcome(result=app.generate(value)))
            assert len(output.main_video) == 32
        with pytest.raises(ApplicationError, match="complete"):
            await app.process_input()
        assert app._state_update().next_chunk is None
        await app.reset(42)
        assert (await app.process_input()).image == Path("anchor")

    asyncio.run(run())


@pytest.mark.parametrize("count", [0, 8, 39, 41, 73])
def test_invalid_lengths(count):
    with pytest.raises(ValueError):
        load_camera_path(
            json.dumps({"poses": np.tile(np.eye(4), (count, 1, 1)).tolist()}).encode(),
            12,
        )
