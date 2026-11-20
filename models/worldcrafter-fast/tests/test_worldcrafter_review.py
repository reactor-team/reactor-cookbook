"""Regression cases for shared sessions, uploads, camera inputs and media loss."""
import asyncio
import io
import json
import threading

import numpy as np
from PIL import Image
import pytest
from reactor_runtime import StepOutcome

from test_worldcrafter import ready, result, run, start
from validate_client import FrameLedger
from worldcrafter_camera import CameraPlanner
from worldcrafter_images import validate_image
from worldcrafter_types import WorldCrafterState


def test_multiple_viewer_disconnect_preserves_accepted_motion():
    app, _ = ready()
    start(app)
    first = run(app.process_input())
    run(app.process_output(StepOutcome(result=result())))
    run(app.move("forward", 1, 4))
    app.connected = asyncio.Event()
    app.connected.set()  # Runtime has already updated this for the remaining viewer.
    run(app.on_disconnected())
    assert len(app.state._queue) == 4 and not app.state._paused
    second = run(app.process_input())
    assert second.image is None
    assert len(second.world_poses) == len(first.world_poses) + 33
    assert second.world_poses[-1, 2, 3] > 0


def test_each_session_owns_its_queue_and_camera():
    a, b = WorldCrafterState(), WorldCrafterState()
    a._queue.append(("forward", 1))
    a._planner.plan("static", 0)
    assert not b._queue and not b._planner.global_chunks


def test_large_jpeg_stays_encoded_and_upload_is_off_loop(monkeypatch):
    import worldcrafter
    from types import SimpleNamespace
    image = Image.new("RGB", (4000, 4000), (10, 30, 50))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    upload = SimpleNamespace(data=encoded.getvalue())
    loop_thread = threading.get_ident()
    called = []

    def validate(value):
        called.append(threading.get_ident())
        return validate_image(value)

    monkeypatch.setattr(worldcrafter, "validate_image", validate)
    app, _ = ready()
    run(app.start_world(upload, "A courtyard", 42, True))
    assert called and called[0] != loop_thread
    assert app.state._image is upload.data
    assert len(app.state._image) == len(encoded.getvalue())


def tag(world, chunk, frame):
    return json.dumps(dict(world_id=world, chunk_index=chunk, frame_index=frame)).encode()


def test_five_completions_with_only_one_video_chunk_fail():
    ledger = FrameLedger()
    black = np.zeros((384, 640, 3), dtype=np.uint8)
    for chunk in range(5):
        ledger.expect(dict(world_id=1, chunk_index=chunk, frames=33))
    for index in range(33):
        ledger.receive(black, tag(1, 0, index))
    assert ledger.complete((1, 0)) and not ledger.complete((1, 1))
    with pytest.raises(AssertionError, match="132 missing"):
        ledger.finish()


def test_dark_frames_repeats_and_reset_boundaries():
    ledger = FrameLedger()
    black = np.zeros((384, 640, 3), dtype=np.uint8)
    ledger.receive(np.zeros((720, 1280, 3), dtype=np.uint8), b"")
    for world in (1, 2):
        ledger.expect(dict(world_id=world, chunk_index=0, frames=33))
        ledger.receive(black, b"")
        for index in range(33):
            ledger.receive(black, tag(world, 0, index))
        ledger.receive(black, tag(world, 0, 32))
    frames, report = ledger.finish()
    assert len(frames) == report["expected_frames"] == report["received_frames"] == 66
    assert report["repeated_frames"] == 2 and report["untagged_frames"] == 3
    assert all(not frame.any() for frame in frames)


def test_replayed_old_world_cannot_satisfy_reset_media():
    ledger = FrameLedger()
    black = np.zeros((384, 640, 3), dtype=np.uint8)
    for world in (1, 2):
        ledger.expect(dict(world_id=world, chunk_index=0, frames=33))
    for _ in range(2):
        for index in range(33):
            ledger.receive(black, tag(1, 0, index))
    with pytest.raises(AssertionError, match="33 missing"):
        ledger.finish()


def test_frame_metadata_matches_each_rgb_frame():
    app, _ = ready()
    start(app)
    run(app.process_input())
    output = run(app.process_output(StepOutcome(result=result())))
    tags = output.__metadata__["main_video"]
    assert tags == [dict(world_id=1, chunk_index=0, frame_index=i) for i in range(33)]


def test_camera_endpoint_sampling_and_private_snapshot():
    camera = CameraPlanner()
    local, first = camera.plan("static", 0)
    camera.align_after_anchor([0., -1., 0.])
    local, forward = camera.plan("forward", 1.)
    assert forward[-1, 2, 3] == pytest.approx(32 / 33)
    _, next_chunk = camera.plan("forward", 1.)
    assert next_chunk[66, 2, 3] == pytest.approx(1.)
    assert local.shape == (66, 3, 4)
    assert np.allclose(local[33], np.eye(4)[:3])
    assert not first.flags.writeable and np.allclose(first, np.eye(4)[:3])


@pytest.mark.parametrize("action", ["yaw_left", "yaw_right", "orbit_left", "orbit_right"])
def test_yaw_and_horizontal_orbit_preserve_calibrated_gravity(action):
    up = np.array([.2, -.95, .24])
    up /= np.linalg.norm(up)
    camera = CameraPlanner()
    camera.plan("static", 0)
    camera.align_after_anchor(up)
    _, world = camera.plan(action, 30.)
    assert np.allclose(world[-33:, :3, :3] @ up, up, atol=1e-6)
