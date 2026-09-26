"""Contract tests for the SolarWM Reactor adapter."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import CommandError, UploadedFile
from solarwm import SolarWM
from solarwm_camera import CameraMotionPlanner, MotionConfig
from solarwm_images import normalize_output_frames, validate_uploaded_image
from solarwm_types import SolarWMState


def _upload() -> UploadedFile:
    stream = io.BytesIO()
    Image.new("RGB", (1280, 720), (40, 80, 120)).save(stream, format="PNG")
    data = stream.getvalue()
    return UploadedFile(name="anchor.png", mime_type="image/png", data=data)


def test_pipeline_declares_buffer_without_fixed_fps() -> None:
    assert SolarWM.buffer_size == 12
    assert "fps" not in SolarWM.__dict__


def test_camera_first_chunk_preserves_anchor_then_moves() -> None:
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    poses = planner.plan_chunk(strafe=0, vertical=0, forward=1, pitch=0, yaw=0, roll=0)
    assert poses.shape == (3, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-6)
    assert poses[1, 2, 3] > 0
    assert poses[2, 2, 3] > poses[1, 2, 3]
    later = planner.plan_chunk(strafe=0, vertical=0, forward=1, pitch=0, yaw=0, roll=0)
    assert later[0, 2, 3] > poses[2, 2, 3]


def test_camera_normalizes_combined_motion() -> None:
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    poses = planner.plan_chunk(strafe=1, vertical=1, forward=1, pitch=1, yaw=1, roll=1)
    assert np.linalg.norm(poses[1, :3, 3]) == pytest.approx(1.0)


def test_upload_validation_and_output_normalization() -> None:
    validate_uploaded_image(_upload())
    frames = normalize_output_frames(np.zeros((12, 480, 864, 3), dtype=np.float32))
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous


def test_set_image_accepts_empty_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow uploaded-image inference with SolarWM's empty text conditioning."""
    import asyncio

    model = SolarWM()
    model.state = SolarWMState()
    model._config = SimpleNamespace(
        max_chunks=320,
        default_prompt="A realistic cinematic scene with smooth camera motion.",
    )
    model._selected_image = None
    model._seed = 42
    model._chunk_index = 0
    model._chunk_in_flight = False
    model._last_chunk_seconds = None
    model.state.prompt = ""
    model.state._restart_requested = True
    model.state._limit_reached = False
    for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll"):
        setattr(model.state, name, 0.0)
    sent = []

    async def record(message: object) -> None:
        sent.append(message)

    monkeypatch.setattr("solarwm.validate_uploaded_image", lambda _image: None)
    monkeypatch.setattr(model, "send", record)

    reply = asyncio.run(model.set_image(_upload(), ""))

    assert reply.prompt == "A realistic cinematic scene with smooth camera motion."
    assert (
        model.state.prompt == "A realistic cinematic scene with smooth camera motion."
    )
    assert model.state._restart_requested is True


def test_upload_rejects_declared_type_mismatch() -> None:
    upload = _upload()
    wrong = UploadedFile(name="anchor.jpg", mime_type="image/jpeg", data=upload.data)
    with pytest.raises(CommandError):
        validate_uploaded_image(wrong)
