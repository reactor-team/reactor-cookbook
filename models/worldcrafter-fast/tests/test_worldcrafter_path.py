"""Finite camera paths follow upstream inference.py --camera-path semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome

from test_worldcrafter import ready, result, run, start
from worldcrafter_camera import FRAMES, load_camera_path
from worldcrafter_types import CameraPathSelected, StateUpdate

MODEL_DIR = Path(__file__).resolve().parents[1]


def path_poses(chunks=3):
    poses = np.tile(np.eye(4, dtype=np.float32), (FRAMES * chunks, 1, 1))
    angle = np.radians(np.arange(len(poses), dtype=np.float32) * 0.5)
    poses[:, 0, 0] = poses[:, 2, 2] = np.cos(angle)
    poses[:, 0, 2], poses[:, 2, 0] = np.sin(angle), -np.sin(angle)
    poses[:, 2, 3] = np.arange(len(poses), dtype=np.float32) * np.float32(0.03)
    return poses


def upload(poses):
    return SimpleNamespace(name="path.json", mime_type="application/json",
                           data=json.dumps({"poses": poses.tolist()}).encode())


def test_float32_path_is_kept_bit_exact():
    poses = path_poses()
    path = load_camera_path(upload(poses).data, 50)
    assert path.chunks == 3 and path.poses.dtype == np.float32
    np.testing.assert_array_equal(path.poses, poses)
    assert not path.poses.flags.writeable


def test_path_off_origin_is_kept_as_upstream_camera_npy():
    shift = np.eye(4, dtype=np.float32)
    shift[:3, 3] = (1.0, 2.0, -3.0)
    poses = shift @ path_poses()
    np.testing.assert_array_equal(load_camera_path(upload(poses).data, 50).poses, poses)


@pytest.mark.parametrize("count", [0, 1, 32, 34, 66 + 1, 33 * 3])
def test_invalid_lengths(count):
    poses = np.tile(np.eye(4), (count, 1, 1))
    with pytest.raises(ValueError):
        load_camera_path(json.dumps({"poses": poses.tolist()}).encode(), 2)


@pytest.mark.parametrize("payload", [
    b"not json", b"[]", json.dumps({"poses": [], "extra": 1}).encode(),
    json.dumps({"poses": (np.tile(np.eye(4), (33, 1, 1)) * 2).tolist()}).encode(),
    json.dumps({"poses": np.full((33, 4, 4), np.nan).tolist()}).encode(),
])
def test_invalid_payloads(payload):
    with pytest.raises(ValueError):
        load_camera_path(payload, 50)


def test_path_steps_stop_reset_replays_and_start_world_restores_move():
    app, messages = ready()
    start(app)
    poses = path_poses()
    reply = run(app.set_camera_path(upload(poses)))
    assert isinstance(reply, CameraPathSelected)
    assert (reply.world_id, reply.chunks, reply.frames) == (2, 3, 99)
    assert [m for m in messages if isinstance(m, StateUpdate)][-1].camera_path_chunks == 3
    with pytest.raises(CommandError) as error:
        run(app.move("forward", 1, 1))
    assert error.value.code == "camera_path_active"
    for index in range(3):
        value = run(app.process_input())
        assert value.local_poses is None and value.world_poses is None
        np.testing.assert_array_equal(value.path_poses, poses[:FRAMES * (index + 1)])
        assert (value.image is not None) == (index == 0) and value.gravity_align
        run(app.process_output(StepOutcome(result=result(world=2, chunk=index))))
    assert app.state._completed == 3
    with pytest.raises(ApplicationError, match="complete"):
        run(app.process_input())
    run(app.reset())
    replay = run(app.process_input())
    assert replay.world_id == 3 and replay.image is not None
    np.testing.assert_array_equal(replay.path_poses, poses[:FRAMES])
    start(app)
    assert app.state._camera_path is None
    assert run(app.process_input()).path_poses is None


def test_path_requires_image_and_invalid_upload_keeps_world():
    app, _ = ready()
    with pytest.raises(CommandError) as error:
        run(app.set_camera_path(upload(path_poses())))
    assert error.value.code == "image_required"
    start(app)
    world, queue = app.state._world_id, list(app.state._queue)
    with pytest.raises(CommandError) as error:
        run(app.set_camera_path(upload(path_poses()[:40])))
    assert error.value.code == "invalid_camera_path"
    assert app.state._world_id == world and list(app.state._queue) == queue


def test_path_longer_than_rollout_is_refused():
    app, _ = ready()
    start(app)
    app._max_chunks = 2
    with pytest.raises(CommandError, match="at most 2 chunks"):
        run(app.set_camera_path(upload(path_poses(3))))


UPSTREAM = os.environ.get("WORLDCRAFTER_UPSTREAM_SOURCE")


@pytest.mark.skipif(not UPSTREAM, reason="set WORLDCRAFTER_UPSTREAM_SOURCE to the pinned upstream checkout")
def test_worker_path_poses_match_upstream_camera_path_preparation(tmp_path):
    """Prefix steps reproduce cli.prepare_camera + inference.generate pose arrays."""
    poses = path_poses(3)
    np.save(tmp_path / "poses.npy", poses)
    script = f"""
import sys
sys.path[:0] = [{UPSTREAM!r}, {str(MODEL_DIR)!r}]
import numpy as np
from worldcrafter.gravity import align_trajectory
from worldcrafter.camera import relative_poses  # ucpe.bridge._relative_pose_chunk's kernel
import importlib.util
spec = importlib.util.spec_from_file_location("wc_worker", {str(MODEL_DIR / "worker.py")!r})
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
poses = np.load({str(tmp_path / "poses.npy")!r})
up = (np.array([0.05, -0.99, 0.12]) / np.linalg.norm([0.05, -0.99, 0.12])).tolist()
for gravity in (None, up):
    camera = align_trajectory(poses, gravity) if gravity is not None else poses
    local_ref = np.concatenate([relative_poses(camera[33 * k:33 * (k + 1)]) for k in range(3)])
    for index in range(3):
        local, world = worker.path_poses(poses[:33 * (index + 1)], gravity, index)
        assert np.array_equal(world, camera[:33 * (index + 1)].astype(np.float32)), index
        assert np.array_equal(local, local_ref[:33 * (index + 1)]), index
print("ok")
"""
    output = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == "ok"


def test_model_validates_path_prefix_shape():
    from test_worldcrafter import FakeBackend
    from worldcrafter_model import InvalidPoses, WorldCrafterInput, WorldCrafterModel
    model = WorldCrafterModel()
    model.backend, model._max_chunks = FakeBackend(), 3
    poses = path_poses()

    def step(path, **extra):
        return WorldCrafterInput(world_id=1, image=b"png", prompt="scene", seed=3, gravity_align=True,
                                 local_poses=extra.get("local"), world_poses=None, path_poses=path)

    with pytest.raises(InvalidPoses):
        model.generate(step(poses[:66]))
    with pytest.raises(InvalidPoses):
        model.generate(step(poses[:33].astype(np.float64)))
    with pytest.raises(InvalidPoses):
        model.generate(step(poses[:33], local=np.zeros((33, 3, 4), np.float32)))
    assert model.backend.steps == []
    model.generate(step(poses[:33]))
    assert len(model.backend.steps) == 1
