"""Check Matrix's upload-gated session startup contract."""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from matrix_game_3_5 import MatrixGame35
from matrix_game_3_5_camera import CameraMotionPlanner, MotionConfig
from matrix_game_3_5_config import read_config
from matrix_game_3_5_model import (
    MatrixGame35Anchor,
    MatrixGame35Input,
    MatrixGame35Model,
    MatrixGame35Result,
    NoAnchor,
    RolloutExhausted,
)
from matrix_game_3_5_types import MatrixGame35State
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract
from upstream_backend import MatrixWorkerBackend

MODEL_DIR = Path(__file__).parents[1]

GENERIC_PROMPT = (
    "An immersive first-person view that faithfully continues the input scene, "
    "preserving its existing environment, objects, geometry, materials, lighting, "
    "and visual style as the camera moves naturally through it."
)


def _upload() -> UploadedFile:
    """Return a small valid anchor upload."""
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(payload, format="PNG")
    return UploadedFile(
        name="anchor.png",
        mime_type="image/png",
        data=payload.getvalue(),
    )


def _model() -> MatrixGame35:
    """Return a loaded-enough model for lifecycle and command checks."""
    model = MatrixGame35()
    model.state = MatrixGame35State()
    model._config = SimpleNamespace(seed=3407, max_chunks=512)
    model._default_prompt = GENERIC_PROMPT
    return model


def test_session_waits_for_an_image_selection() -> None:
    """Wait for the viewer's anchor instead of generating from the demo image."""
    model = _model()

    model.on_session_started()
    state = model._state_update()

    assert model._selected_input is None
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.completed_chunks == 0
    assert state.next_chunk is None
    assert set(ModelContract.of(MatrixGame35).commands) == {
        "reset",
        "set_forward",
        "set_image",
        "set_pitch",
        "set_prompt",
        "set_roll",
        "set_strafe",
        "set_vertical",
        "set_yaw",
    }


def test_first_upload_starts_continuous_generation() -> None:
    """Start continuous generation from the uploaded anchor."""
    model = _model()
    model.on_session_started()

    state = model.set_image(_upload(), "")

    assert state.image_source == "uploaded"
    assert state.image_name == "anchor.png"
    assert state.completed_chunks == 0
    assert state.next_chunk == 1
    assert state.prompt == GENERIC_PROMPT
    assert model.state._world_id != model.state._applied_world_id


def test_generation_controls_require_an_uploaded_image() -> None:
    """Reject controls that cannot produce a chunk before anchor selection."""
    model = _model()
    model.on_session_started()

    with pytest.raises(CommandError) as error:
        model.set_forward(1.0)

    assert error.value.code == "image_required"


def make_model():
    model = MatrixGame35()
    model.state = MatrixGame35State()
    model._config = SimpleNamespace(
        seed=42,
        max_chunks=20,
        chunk_latents=4,
        upload_intrinsics=(1, 1, 0.5, 0.5),
    )
    model.send = AsyncMock()
    model.output.flush = Mock()
    model._selected_input = Path("anchor.png")
    model.state.prompt = "A garden."
    model._planner = CameraMotionPlanner(np.eye(4), MotionConfig(16, 1.5, 45))
    model._engine = FakeModel()
    return model


class FakeModel:
    def __init__(self):
        self.inputs = []
        self.world_id = None
        self.index = 0
        self.reset_count = 0
        self.limit = 20
        self.error = None

    def generate(self, input):
        self.inputs.append(input)
        if self.error:
            raise self.error
        if input.world_id != self.world_id:
            assert input.anchor is not None
            self.world_id = input.world_id
            self.index = 0
        self.index += 1
        return MatrixGame35Result(
            np.zeros((12, 8, 8, 3), np.uint8),
            input.world_id,
            self.index,
            self.index >= self.limit,
        )

    def reset(self):
        self.reset_count += 1
        self.world_id = None
        self.index = 0


def step(model):
    input = asyncio.run(model.process_input())
    result = model.generate(input)
    output = asyncio.run(model.process_output(StepOutcome(result=result, elapsed=1)))
    return input, result, output


def test_input_waits_for_image_and_rejects_rollout_limit():
    model = make_model()
    model._selected_input = None
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    assert model._engine.inputs == []
    model._selected_input = Path("anchor.png")
    model.state._limit_reached = True
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_failure_does_not_advance_progress():
    model = make_model()
    model._engine.error = RuntimeError("failed")
    input = asyncio.run(model.process_input())
    with pytest.raises(RuntimeError, match="failed") as error:
        model.generate(input)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=error.value)))
    assert model._chunk_index == 0
    assert model.state._applied_world_id is None
    assert asyncio.run(model.process_input()).anchor is not None
    model.send.assert_not_called()


def test_controls_are_snapshotted_and_native_chunks_remain_continuous():
    model = make_model()
    previous_end = None
    for index in range(10):
        model.set_yaw(0.1 if index % 2 else -0.1)
        input, result, output = step(model)
        assert (input.anchor is not None) == (index == 0)
        assert input.trajectory.shape == (13, 4, 4)
        if previous_end is not None:
            np.testing.assert_array_equal(input.trajectory[0], previous_end)
        previous_end = input.trajectory[-1].copy()
        assert result.chunk_index == index + 1
        assert output.main_video.shape == (12, 8, 8, 3)
    assert model._chunk_index == 10
    assert len({input.world_id for input in model._engine.inputs}) == 1


def test_generate_only_reads_frozen_snapshot():
    model = make_model()
    input = asyncio.run(model.process_input())
    with pytest.raises(FrozenInstanceError):
        input.prompt = "Changed"
    model.state = model._config = model._planner = model._selected_input = None
    assert model.generate(input).world_id == input.world_id


def test_blank_prompt_refused_without_inference():
    model = make_model()
    model.state.prompt = " "
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    assert model._engine.inputs == []


def test_upload_prompt_reset_controls_and_session_cleanup():
    model = make_model()
    first, _, _ = step(model)
    model.set_prompt(" New garden ")
    second, _, _ = step(model)
    assert second.prompt == "New garden" and second.anchor is None
    assert second.world_id == first.world_id
    model.reset(seed=123)
    third, _, _ = step(model)
    assert third.anchor.seed == 123 and third.world_id != first.world_id
    image = _upload()
    model.set_image(image, "A forest")
    fourth, _, _ = step(model)
    assert fourth.anchor.image == image.data and fourth.anchor.suffix == ".png"
    assert fourth.world_id != third.world_id
    for axis in ("forward", "strafe", "vertical", "pitch", "yaw", "roll"):
        assert getattr(model, "set_" + axis)(0.2).next_chunk == 2
    asyncio.run(model.on_disconnected())
    assert all(
        getattr(model.state, axis) == 0
        for axis in ("forward", "strafe", "vertical", "pitch", "yaw", "roll")
    )
    model.on_session_ended()
    assert model._engine.reset_count == 1 and model._selected_input is None
    assert model.state._applied_world_id is None


def test_cap_requires_explicit_restart_and_preserves_message_order():
    model = make_model()
    model._engine.limit = 1
    step(model)
    assert model.state._limit_reached
    assert [call.args[0].__class__.__name__ for call in model.send.call_args_list] == [
        "RolloutLimitReached",
        "StateUpdate",
    ]
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    with pytest.raises(CommandError):
        model.set_yaw(0.2)
    model.set_prompt("Next world")
    assert model.state._limit_reached
    model.reset(seed=-1)
    assert asyncio.run(model.process_input()).anchor is not None


def test_model_native_count_seed_prompt_cap_and_cleanup():
    engine = MatrixGame35Model()
    engine._backend = Mock()
    engine._max_chunks = 2
    engine._backend.generate_chunk.return_value = np.zeros((12, 8, 8, 3), np.uint8)
    input = MatrixGame35Input(1, None, "Garden", np.tile(np.eye(4), (13, 1, 1)))
    with pytest.raises(NoAnchor):
        engine.generate(input)
    anchor = MatrixGame35Anchor(b"image", ".png", 42)
    assert engine.generate(replace(input, anchor=anchor)).chunk_index == 1
    final = engine.generate(replace(input, prompt="Forest"))
    assert final.complete and final.chunk_index == 2
    engine._backend.reset.assert_called_once_with(
        seed=42, anchor_image=b"image", suffix=".png", prompt="Garden"
    )
    assert engine._backend.generate_chunk.call_args.args[0] is input.trajectory
    assert engine._backend.generate_chunk.call_args.args[1:] == (42, "Forest")
    with pytest.raises(RolloutExhausted):
        engine.generate(input)
    assert engine.generate(replace(input, world_id=2, anchor=anchor)).chunk_index == 1
    engine.reset()
    engine._backend.end_session.assert_called_once()
    with pytest.raises(NoAnchor):
        engine.generate(input)


@pytest.mark.parametrize("failure", [False, True])
def test_backend_materializes_and_cleans_upload(tmp_path, failure):
    backend = MatrixWorkerBackend.__new__(MatrixWorkerBackend)
    backend._root = tmp_path
    backend._request_id = 0
    observed = []

    def request(command, **payload):
        path = Path(payload["anchor_image"])
        observed.append(path)
        assert path.suffix == ".png" and path.read_bytes() == b"image"
        if failure:
            raise RuntimeError("worker failed")
        return {}

    backend._request = request
    if failure:
        with pytest.raises(RuntimeError):
            backend.reset(42, b"image", "Garden", ".png")
    else:
        backend.reset(42, b"image", "Garden", ".png")
    assert observed and not observed[0].exists()


def test_model_backend_and_config_import_without_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('reactor_runtime'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import matrix_game_3_5_model
import upstream_backend
import matrix_game_3_5_config
""",
        ],
        check=True,
    )


def test_config_uses_explicit_weights_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIX_GAME_3_5_PATH", raising=False)
    config = read_config(MODEL_DIR / "matrix_game_3_5.yaml", tmp_path)
    assert tmp_path in config.source_path.parents
    assert config.max_chunks == 512


@pytest.mark.parametrize("shape", [(12, 8, 8, 4), (11, 8, 8, 3), (12, 8, 3)])
def test_invalid_native_output_does_not_count_as_completed(shape):
    model = make_model()
    engine = MatrixGame35Model()
    engine._backend = Mock()
    engine._max_chunks = 20
    engine._backend.generate_chunk.return_value = np.zeros(shape, np.uint8)
    model._engine = engine
    with pytest.raises(RuntimeError) as error:
        model.generate(asyncio.run(model.process_input()))
    with pytest.raises(RuntimeError):
        asyncio.run(model.process_output(StepOutcome(error=error.value)))
    assert engine._chunk_index == 0
    assert model._chunk_index == 0 and model.state._applied_world_id is None
    model.send.assert_not_called()
