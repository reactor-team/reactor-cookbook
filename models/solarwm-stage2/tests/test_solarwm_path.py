import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from solarwm_stage2 import SolarWM
from solarwm_stage2_camera import CameraMotionPlanner, MotionConfig, load_camera_path
from solarwm_stage2_types import SolarWMState


def path(frames=117):
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 2, 3] = np.arange(frames) * 0.03
    return json.dumps(dict(poses=poses.tolist())).encode()


def test_causal_indices_and_finite_budget():
    value = load_camera_path(path(), 320)
    assert value.frames == 117 and value.chunks == 10
    np.testing.assert_allclose(
        value.poses[:, 2, 3], np.array([0, *range(1, 117, 4)]) * 0.03, atol=1e-6
    )
    with pytest.raises(ValueError):
        value.chunk(10)


@pytest.mark.parametrize("frames", [0, 8, 10, 116, 120])
def test_reject_partial_chunks(frames):
    with pytest.raises(ValueError):
        load_camera_path(path(frames), 320)


def test_bad_pose_atomic_and_ten_chunks():
    app = SolarWM()
    app.state = SolarWMState()
    app._config = SimpleNamespace(max_chunks=320)
    app.state._selected_image = np.zeros((8, 8, 3), np.uint8)
    app.state._planner = CameraMotionPlanner(MotionConfig(1, 8))

    async def run():
        uploaded = UploadedFile(
            name="path.json", mime_type="application/json", data=path()
        )
        selected = await app.set_camera_path(uploaded)
        assert selected.chunks == 10
        original = app._camera_path
        with pytest.raises(CommandError):
            await app.set_camera_path(
                UploadedFile(name="bad.json", mime_type="application/json", data=b"{}")
            )
        assert app._camera_path is original
        with pytest.raises(CommandError):
            await app.set_forward(1.0)
        for i in range(10):
            value = await app.process_input()
            again = await app.process_input()
            np.testing.assert_array_equal(value.poses, again.poses)
            np.testing.assert_array_equal(value.poses, original.chunk(i))
            assert (value.anchor is not None) == (i == 0)
            result = SimpleNamespace(
                world_id=value.world_id,
                chunk_index=i + 1,
                complete=False,
                max_chunks=320,
                frames=np.zeros((9 if i == 0 else 12, 8, 8, 3), np.uint8),
            )
            await app.process_output(StepOutcome(result=result, elapsed=0.1))
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(42)
        np.testing.assert_array_equal(
            (await app.process_input()).poses, original.chunk(0)
        )

    asyncio.run(run())


def _random_path(frames=129, seed=0):
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.4, (frames, 3))).as_matrix()
    poses[:, :3, 3] = rng.normal(0, 5, (frames, 3))
    return poses.astype(np.float32)


def test_relative_poses_rebase_on_the_first_pose():
    pytest.importorskip("scipy")
    poses = _random_path()
    value = load_camera_path(json.dumps(dict(poses=poses.tolist())).encode(), 320)
    selected = poses[[0, *range(1, 129, 4)]].astype(np.float64)
    expected = np.linalg.inv(selected[0]) @ selected
    np.testing.assert_allclose(value.poses, expected, atol=1e-4)
    np.testing.assert_array_equal(value.poses[0], np.eye(4))


def test_relative_poses_match_upstream_camera_tokens_bitwise():
    """Same FP32 operation order as upstream build_camera_tokens."""
    pytest.importorskip("scipy")
    data = pytest.importorskip("solarwm.backends.wan22.runtime.data")
    poses = _random_path(seed=1)
    value = load_camera_path(json.dumps(dict(poses=poses.tolist())).encode(), 320)
    selected = poses[data.latent_aligned_pixel_indices(len(poses))]
    np.testing.assert_array_equal(
        value.poses, data._relative_c2w_fp32(selected, "absolute_c2w")
    )


def test_camera_tokens_match_upstream_build_camera_tokens():
    """A path's tokens equal upstream's for the same RGB-rate C2W poses."""
    import io

    pytest.importorskip("scipy")
    data = pytest.importorskip("solarwm.backends.wan22.runtime.data")
    from solarwm.data.camera import CameraGuards

    from solarwm_stage2_backend import _camera_tokens

    poses = _random_path(seed=2)
    buffer = io.BytesIO()
    np.savez(buffer, c2w=poses)
    manifest = {
        "camera": {
            "convention": data.CAMERA_CONVENTION,
            "dtype": "float32",
            "finite": True,
            "magnitude_audit_seconds": 10.0,
            "max_camera_abs": 20.0,
            "max_rel_translation": 20.0,
            "shape": list(poses.shape),
        }
    }
    expected = data.build_camera_tokens(
        buffer.getvalue(),
        range(len(poses)),
        manifest,
        source_fps=16.0,
        output_fps=16.0,
        frame_sequence_length=5,
        guards=CameraGuards(None, None),
        manifest_guards=CameraGuards(20.0, 20.0),
        configured_array_key="c2w",
    )
    path = load_camera_path(json.dumps(dict(poses=poses.tolist())).encode(), 320)
    for index in range(path.chunks):
        tokens = _camera_tokens(path.chunk(index), 5, "cpu")
        window = slice(index * 15, (index + 1) * 15)
        np.testing.assert_array_equal(
            tokens["viewmats"][0].numpy(), expected["viewmats"][window].numpy()
        )
        np.testing.assert_array_equal(tokens["K"][0].numpy(), expected["K"][window].numpy())
