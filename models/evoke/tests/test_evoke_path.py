"""Finite uploaded trajectories reach the worker verbatim with their anchor."""

import asyncio
import json

import numpy as np
import pytest
from evoke_camera import load_camera_path
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from test_evoke import _ready_model


def upload(chunks=10):
    poses = np.tile(np.eye(4), (36 * chunks, 1, 1))
    poses[:, 0, 3] = np.arange(len(poses)) * 0.02 + 3
    return UploadedFile(
        name="path.json",
        mime_type="application/json",
        data=json.dumps({"poses": poses.tolist()}).encode(),
    )


def test_path_keeps_uploaded_poses():
    file = upload()
    poses = np.asarray(json.loads(file.data)["poses"], dtype=np.float32)
    path = load_camera_path(file.data, 10)
    # The released launcher uses a pose track in its own frame; no re-anchoring here.
    np.testing.assert_array_equal(path.poses, poses)
    assert path.chunk_count == 10


def test_ten_chunks_stop_reset_and_controls():
    async def run():
        app, backend, _ = _ready_model()
        reply = await app.set_camera_path(upload())
        assert (reply.chunks, reply.frames) == (10, 360)
        with pytest.raises(CommandError):
            await app.set_yaw(1)
        expected = load_camera_path(upload().data, 10).poses
        for index in range(10):
            value = await app.process_input()
            # The whole path travels with the anchor; chunks carry no per-step poses.
            assert value.trajectory is None
            if index == 0:
                np.testing.assert_array_equal(value.anchor.camera_path, expected)
            else:
                assert value.anchor is None
            result = app.generate(value)
            out = await app.process_output(StepOutcome(result=result))
            assert len(out.main_video) == 36
        assert len(backend.generate_calls) == 10
        with pytest.raises(ApplicationError, match="complete"):
            await app.process_input()
        await app.reset(42)
        restarted = await app.process_input()
        np.testing.assert_array_equal(restarted.anchor.camera_path, expected)
        assert restarted.trajectory is None
        await app.start_text(prompt="Scene", seed=42)
        assert app._camera_path is None
        with pytest.raises(CommandError):
            await app.set_camera_path(upload())

    asyncio.run(run())


@pytest.mark.parametrize("count", [0, 1, 35, 37, 73])
def test_invalid_lengths(count):
    with pytest.raises(ValueError):
        load_camera_path(
            json.dumps({"poses": np.tile(np.eye(4), (count, 1, 1)).tolist()}).encode(),
            10,
        )
