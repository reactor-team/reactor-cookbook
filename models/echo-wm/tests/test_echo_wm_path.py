"""Finite camera paths preserve native sampling and causal progress."""

import asyncio
import json

import numpy as np
import pytest
from echo_wm_camera import load_camera_path
from reactor_runtime import ApplicationError, CommandError, UploadedFile
from test_echo_wm import _selected_app, _step


def upload(chunks=12):
    poses = np.tile(np.eye(4), (24 * chunks + 1, 1, 1))
    poses[:, 0, 3] = np.arange(len(poses)) * 0.05
    return UploadedFile(
        name="path.json",
        mime_type="application/json",
        data=json.dumps({"poses": poses.tolist()}).encode(),
    )


def test_native_sampling_and_anchor():
    trajectory = upload()
    data = json.loads(trajectory.data)
    poses = np.asarray(data["poses"], dtype=np.float32)
    poses[:, 2, 3] += 2
    parsed = load_camera_path(json.dumps({"poses": poses.tolist()}).encode(), 12)
    expected = np.linalg.inv(poses[0]) @ poses[::8]
    np.testing.assert_allclose(parsed.chunks.reshape(-1, 4, 4), expected[1:])
    assert parsed.frames == 289


def test_finite_path_steps_reset_and_controls():
    async def run():
        app, engine, messages = _selected_app(12)
        result = await app.set_camera_path(upload())
        assert (result.chunks, result.frames) == (12, 289)
        with pytest.raises(CommandError):
            await app.set_camera_motion(forward=1, strafe=0, yaw=0, pitch=0)
        expected = load_camera_path(upload().data, 12).chunks
        for index in range(12):
            inp = await app.process_input()
            assert inp.rollout_chunks == 12
            np.testing.assert_array_equal(inp.poses, expected[index])
            # Preparing an input does not consume the uploaded trajectory.
            np.testing.assert_array_equal((await app.process_input()).poses, inp.poses)
            out = await _step(app)
            assert len(out.main_video) == (25 if index == 0 else 24)
        assert len(engine.inputs) == 12
        assert not any(type(x).__name__ == "AutomaticResetQueued" for x in messages)
        with pytest.raises(ApplicationError, match="complete"):
            await app.process_input()
        await app.reset(seed=42)
        np.testing.assert_array_equal((await app.process_input()).poses, expected[0])

    asyncio.run(run())


@pytest.mark.parametrize("count", [0, 1, 24, 26, 50])
def test_invalid_lengths(count):
    poses = np.tile(np.eye(4), (count, 1, 1))
    with pytest.raises(ValueError):
        load_camera_path(json.dumps({"poses": poses.tolist()}).encode(), 12)


def test_invalid_rotation_does_not_change_active_path():
    async def run():
        app, _, _ = _selected_app()
        await app.set_camera_path(upload())
        active = app._camera_path
        bad = json.loads(upload().data)
        bad["poses"][1][0][0] = 2
        with pytest.raises(CommandError):
            await app.set_camera_path(
                UploadedFile(
                    name="bad.json",
                    mime_type="application/json",
                    data=json.dumps(bad).encode(),
                )
            )
        assert app._camera_path is active

    asyncio.run(run())


def test_path_rollout_forwards_complete_poses_and_keeps_fov():
    async def run():
        app, engine, _ = _selected_app(12)
        await app.set_fov(fov_degrees=80.0)
        await app.set_camera_path(upload())
        # Before the path's rollout starts the field of view still applies.
        await app.set_fov(fov_degrees=75.0)
        first = await app.process_input()
        await _step(app)
        np.testing.assert_array_equal(
            first.camera_path, load_camera_path(upload().data, 12).poses
        )
        assert first.camera_path.shape == (289, 4, 4)
        assert first.fov_degrees == 75.0
        with pytest.raises(CommandError, match="field of view"):
            await app.set_fov(fov_degrees=90.0)
        await app.reset(seed=42)
        await app.set_fov(fov_degrees=90.0)

    asyncio.run(run())


def test_model_follows_path_without_per_chunk_camera(tmp_path):
    from dataclasses import replace
    from pathlib import Path

    from echo_wm_assets import read_config
    from echo_wm_model import EchoAnchor, EchoInput, EchoModel
    from test_echo_wm import RecordingBackend

    engine = EchoModel()
    backend = RecordingBackend()
    engine._backend = backend
    engine._config = replace(
        read_config(Path(__file__).parents[1] / "echo_wm.yaml"),
        runtime_dir=tmp_path,
        max_chunks=2,
    )
    path = load_camera_path(upload(2).data, 2)
    input = EchoInput(
        world_id=1,
        anchor=EchoAnchor(image=b"image", suffix=".png", seed=7),
        prompt="room",
        poses=path.chunks[0],
        fov_degrees=70.0,
        rollout_chunks=2,
        camera_path=path.poses,
    )
    engine.generate(input)
    engine.generate(replace(input, anchor=None, poses=path.chunks[1]))
    assert backend.starts[0]["camera_path"] is path.poses
    assert [camera for camera, _ in backend.cameras] == [None, None]
    held = replace(input, world_id=2, camera_path=None, rollout_chunks=None)
    engine.generate(held)
    assert backend.cameras[-1][0].latent_poses is held.poses


def test_path_condition_matches_upstream_action_condition():
    """The uploaded-path condition reproduces upstream's Action DSL tensors."""
    torch = pytest.importorskip("torch")
    from pathlib import Path
    from types import SimpleNamespace

    from echo_wm_assets import activate_source, read_config
    from reactor_runtime import get_weights_path

    config = read_config(Path(__file__).parents[1] / "echo_wm.yaml", get_weights_path())
    if not config.source_path.exists():
        pytest.skip("Pinned upstream source is required for the camera oracle")
    activate_source(config)
    import echo_wm_backend as module
    from helpers.action_camera import build_action_pt_from_string
    from helpers.action_condition import build_causal_action_condition

    action = "w-30,l-20,ja-25,none-21"
    width, height, frames = 1280, 704, 97
    speeds = dict(translation_speed=0.05, rotation_speed_deg=0.4, pitch_limit_deg=40.0)
    poses = build_action_pt_from_string(
        action, num_frames=frames, image_width=width, image_height=height,
        fov_deg=70.0, fps=24.0, **speeds,
    )["c2ws_raw"].numpy()
    # The adapter uploads float32 poses as JSON; parsing restores them exactly.
    trajectory = UploadedFile(
        name="path.json",
        mime_type="application/json",
        data=json.dumps({"poses": poses.tolist()}).encode(),
    )
    path = load_camera_path(trajectory.data, 4)
    np.testing.assert_array_equal(path.poses, poses)
    backend = module.EchoWMBackend.__new__(module.EchoWMBackend)
    backend._config = SimpleNamespace(width=width, height=height)
    backend._device = torch.device("cpu")
    backend._total_video_latents = 13
    actual = backend._path_action_condition(path.poses, 70.0)
    expected = build_causal_action_condition(
        action, num_frames=frames, width=width, height=height, fov_deg=70.0,
        device=torch.device("cpu"), fps=24.0, **speeds,
    )
    for key in ("ucpe_viewmats", "ucpe_Ks"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual[key].stride() == expected[key].stride()
