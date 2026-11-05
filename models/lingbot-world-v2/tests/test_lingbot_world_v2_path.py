"""Verify finite camera uploads and native chunk boundaries without weights."""

import asyncio
import json

import numpy as np
import pytest
from lingbot_world_v2_camera import load_camera_path
from reactor_runtime import ApplicationError, CommandError, UploadedFile
from test_lingbot_world_v2 import make_model, step, upload


def test_native_rgb_tail_does_not_spend_an_extra_chunk():
    path = load_camera_path(path_data(65), max_chunks=4, chunk_latents=4)
    assert len(path.chunks) == 4 and path.output_frames == 61
    with pytest.raises(ValueError):
        load_camera_path(path_data(77), max_chunks=4, chunk_latents=4)


def path_data(frames=49):
    poses = np.tile(np.eye(4), (frames, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 2, frames)
    return json.dumps(
        {"poses": poses.tolist(), "intrinsics": [503, 503, 416, 240]}
    ).encode()


def test_three_chunks_stop_and_reset_replays():
    app = make_model()
    selected = asyncio.run(
        app.set_camera_path(
            UploadedFile(
                name="path.json", mime_type="application/json", data=path_data()
            ),
            noise_chunks=-1,
        )
    )
    assert (selected.chunks, selected.frames, selected.source_frames) == (3, 45, 49)
    assert app._state_update().camera_path_chunks == 3
    before = asyncio.run(app.process_input())
    again = asyncio.run(app.process_input())
    np.testing.assert_array_equal(before.poses, again.poses)
    inputs = []
    for expected in [13, 16, 16]:
        inp, result, _ = step(app)
        assert len(result.frames) == expected
        inputs.append(inp)
    assert inputs[0].anchor is not None and inputs[1].anchor is None
    assert inputs[0].anchor.noise_chunks == 3
    np.testing.assert_array_equal(inputs[0].anchor.intrinsics, [503, 503, 416, 240])
    assert app._state_update().next_chunk is None
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    asyncio.run(app.reset(seed=42))
    np.testing.assert_array_equal(
        asyncio.run(app.process_input()).poses, inputs[0].poses
    )
    with pytest.raises(CommandError):
        asyncio.run(
            app.set_camera(forward=1, strafe=0, vertical=0, yaw=0, pitch=0, roll=0)
        )
    asyncio.run(app.set_image(upload(), prompt="A garden."))
    assert app._camera_path is None


def test_explicit_noise_horizon_keeps_native_path_and_replays_on_reset():
    app = make_model()
    upload_path = UploadedFile(
        name="path.json", mime_type="application/json", data=path_data()
    )
    asyncio.run(app.set_camera_path(upload_path, noise_chunks=12))
    first, _, _ = step(app)
    assert first.anchor.noise_chunks == 12
    asyncio.run(app.reset(seed=42))
    assert asyncio.run(app.process_input()).anchor.noise_chunks == 12
    asyncio.run(app.set_image(upload(), prompt="A garden."))
    asyncio.run(app.set_camera_path(upload_path, noise_chunks=-1))
    assert asyncio.run(app.process_input()).anchor.noise_chunks == 3


@pytest.mark.parametrize("noise", [0, 1, 2, 257])
def test_noise_horizon_rejection_preserves_selected_camera_path(noise):
    app = make_model()
    upload_path = UploadedFile(
        name="path.json", mime_type="application/json", data=path_data()
    )
    asyncio.run(app.set_camera_path(upload_path, noise_chunks=-1))
    original = app._camera_path
    with pytest.raises(CommandError):
        asyncio.run(app.set_camera_path(upload_path, noise_chunks=noise))
    assert app._camera_path is original


@pytest.mark.parametrize(
    "frames, chunks, output",
    [(13, 1, 13), (16, 1, 13), (49, 3, 45), (253, 16, 253), (316, 19, 301)],
)
def test_native_length_rounding(frames, chunks, output):
    path = load_camera_path(path_data(frames), max_chunks=32, chunk_latents=4)
    assert len(path.chunks) == chunks and path.output_frames == output
    flat = path.chunks.reshape(-1, 4, 4)
    np.testing.assert_allclose(flat[0], np.eye(4), atol=1e-5)
    np.testing.assert_allclose(flat[1:, 0, 3], 1, atol=1e-5)


@pytest.mark.parametrize(
    "change", ["nan", "rotation", "intrinsics", "short", "oversized"]
)
def test_invalid_path_does_not_replace_selection(change):
    app = make_model()
    data = json.loads(path_data())
    if change == "nan":
        data["poses"][1][0][0] = float("nan")
    if change == "rotation":
        data["poses"][1][0][0] = 2
    if change == "intrinsics":
        data["intrinsics"][0] = 0
    if change == "short":
        data["poses"] = data["poses"][:12]
    payload = (
        b" " * (2 * 1024 * 1024 + 1)
        if change == "oversized"
        else json.dumps(data).encode()
    )
    with pytest.raises(CommandError):
        asyncio.run(
            app.set_camera_path(
                UploadedFile(
                    name="bad.json", mime_type="application/json", data=payload
                ),
                noise_chunks=-1,
            )
        )
    assert app._camera_path is None
