"""Test native step boundaries without loading GPU weights."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome
from sana_wm import SanaWM
from sana_wm_model import SanaModel, TrajectoryCompleteError
from sana_wm_types import SanaWMState


def test_image_and_calibration_uploads_are_separate(monkeypatch):
    import sana_wm
    from reactor_runtime.interface.model.contract import ModelContract

    contract = ModelContract.of(SanaWM)
    assert "set_intrinsics" in contract.commands
    import inspect

    assert "intrinsics" not in inspect.signature(SanaWM.set_image).parameters
    model, _ = ready()
    monkeypatch.setattr(sana_wm, "_validate_image", lambda value: None)
    monkeypatch.setattr(sana_wm, "_validate_intrinsics", lambda value: None)
    calibration = object()
    image = SimpleNamespace(name="uploaded.png")

    async def run():
        await model.set_intrinsics(calibration)
        model.state.prompt = "Cars from the previous sample"
        await model.set_image(image, prompt="")
        assert model._intrinsics_input is calibration
        assert model._pending_intrinsics is None
        assert "Cars" not in model.state.prompt
        await model.set_image(image, prompt="A lake")
        assert model._intrinsics_input is None
        assert model.state.prompt == "A lake"

    asyncio.run(run())


class Backend:
    def __init__(self):
        self.chunk_index = 0
        self.trajectory_frames = None
        self.resets = []
        self.controls = []

    def reset(self, image, prompt, seed, **kwargs):
        self.resets.append((image, prompt, seed, kwargs))
        self.chunk_index = 0

    def generate_chunk(self, poses):
        self.controls.append(poses)
        self.chunk_index += 1
        return np.zeros((24, 8, 8, 3), dtype=np.uint8)

    def end_session(self):
        self.chunk_index = 0


class Planner:
    def __init__(self):
        self.controls = []
        self.exhausted = False

    def reset(self, trajectory):
        self.trajectory = trajectory

    def plan_chunk(self, controls):
        self.controls.append(controls)
        return (
            None if self.exhausted else np.tile(np.eye(4, dtype=np.float32), (24, 1, 1))
        )


def ready():
    model = SanaWM()
    model.state = SanaWMState()
    model._config = SimpleNamespace(max_chunks=512)
    model._engine = SanaModel()
    model._engine._backend = Backend()
    model._engine._config = model._config
    model._camera = Planner()
    model.on_session_started()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    return model, messages


def test_waits_for_explicit_image():
    model, _ = ready()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_snapshot_and_continuous_native_chunks():
    model, _ = ready()
    model._selected_image = Path("anchor.png")
    model.state.prompt = "A lake"
    model.state._world_id = 1
    model.state._held_controls = {"forward"}

    async def run():
        for index in range(10):
            snapshot = await model.process_input()
            model.state._held_controls = {"yaw_left"}
            result = model.generate(snapshot)
            output = await model.process_output(StepOutcome(result=result))
            assert output.main_video.shape == (24, 8, 8, 3)
            assert model._chunk_index == index + 1
        assert len(model._engine._backend.resets) == 1
        assert model._camera.controls[0] == {"forward"}
        assert model._camera.controls[1] == {"yaw_left"}

    asyncio.run(run())


def test_trajectory_completion_and_error_cleanup():
    model, messages = ready()
    model._selected_image = Path("anchor.png")

    async def run():
        await model.process_output(
            StepOutcome(result=model.generate(await model.process_input()))
        )
        model._camera.exhausted = True
        snapshot = await model.process_input()
        with pytest.raises(TrajectoryCompleteError) as raised:
            model.generate(snapshot)
        output = await model.process_output(StepOutcome(error=raised.value))
        assert output is None
        assert model.state._trajectory_exhausted
        assert not model._generating and not model._chunk_in_flight
        with pytest.raises(ApplicationError):
            await model.process_input()
        with pytest.raises(RuntimeError, match="failure"):
            await model.process_output(StepOutcome(error=RuntimeError("failure")))
        assert not model._generating and not model._chunk_in_flight

    asyncio.run(run())
    assert any(type(message).__name__ == "TrajectoryExhausted" for message in messages)


def test_automatic_restart_commits_effects_in_output(monkeypatch):
    model, messages = ready()
    model._selected_image = Path("anchor.png")
    model._chunk_index = 512
    model.state._applied_world_id = model.state._world_id
    model.state._held_controls = {"forward"}
    flushes = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    async def run():
        snapshot = await model.process_input()
        assert not messages and not flushes
        assert snapshot.anchor is not None
        assert snapshot.world_id == model._automatic_world_id
        assert not model._camera.controls[-1]
        await model.process_output(StepOutcome(result=model.generate(snapshot)))
        assert model._chunk_index == 1
        assert not model.state._held_controls
        assert flushes == [None]

    asyncio.run(run())


def test_frozen_forwarding_does_not_read_app_state():
    from dataclasses import FrozenInstanceError

    model, _ = ready()
    model._selected_image = Path("anchor.png")
    model.state.prompt = "A lake"
    input = asyncio.run(model.process_input())
    with pytest.raises(FrozenInstanceError):
        input.world_id = 999
    model.state = None
    model._camera = model._config = None
    assert model.generate(input).prompt == "A lake"


def test_anchor_sent_until_success_then_only_id_and_ten_actions():
    model, _ = ready()
    model._selected_image = Path("anchor.png")
    model.state.prompt = "A lake"

    async def run():
        for index, control in enumerate(
            [
                "forward",
                "back",
                "yaw_left",
                "yaw_right",
                "pitch_up",
                "pitch_down",
                "strafe_left",
                "strafe_right",
                "forward",
                "back",
            ]
        ):
            await model.release_controls()
            await model.set_control(control, True)
            input = await model.process_input()
            assert (input.anchor is not None) == (index == 0)
            assert model._camera.controls[-1] == {control}
            result = model.generate(input)
            await model.process_output(StepOutcome(result=result))
            assert result.chunk_index == index + 1
        assert len(model._engine._backend.resets) == 1

    asyncio.run(run())


def test_model_error_does_not_acknowledge_or_count(monkeypatch):
    model, messages = ready()
    model._selected_image = Path("anchor.png")
    input = asyncio.run(model.process_input())

    def fail(input):
        raise RuntimeError("failure")

    monkeypatch.setattr(model._engine, "generate", fail)
    with pytest.raises(RuntimeError) as raised:
        model.generate(input)
    with pytest.raises(RuntimeError):
        asyncio.run(model.process_output(StepOutcome(error=raised.value)))
    assert model._chunk_index == 0
    assert model.state._applied_world_id is None
    assert messages == []
    assert not model._generating
    assert asyncio.run(model.process_input()).anchor is not None


def test_upload_prompt_calibration_reset_and_session_cleanup():
    import io

    from PIL import Image
    from reactor_runtime import UploadedFile

    model, _ = ready()
    data = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(data, "PNG")
    image = UploadedFile(name="image.png", mime_type="image/png", data=data.getvalue())
    data = io.BytesIO()
    np.save(data, np.array([10, 10, 4, 4], np.float32))
    calibration = UploadedFile(
        name="intrinsics.npy",
        mime_type="application/octet-stream",
        data=data.getvalue(),
    )

    async def run():
        await model.set_intrinsics(calibration)
        await model.set_image(image, "A lake")
        input = await model.process_input()
        assert input.anchor.image == image.data
        assert input.anchor.intrinsics == calibration.data
        await model.process_output(StepOutcome(result=model.generate(input)))
        await model.set_prompt("A mountain")
        input = await model.process_input()
        assert input.anchor.prompt == "A mountain"
        await model.process_output(StepOutcome(result=model.generate(input)))
        await model.reset(77)
        assert (await model.process_input()).anchor.seed == 77
        model.on_session_ended()
        assert model._selected_image is None
        assert model._intrinsics_input is None
        assert model._chunk_index == 0
        assert model._engine._world_id is None

    asyncio.run(run())


def test_model_guards_and_progress():
    from dataclasses import replace

    from sana_wm_model import NoAnchor, RolloutExhausted, SanaAnchor, SanaInput

    model, _ = ready()
    engine = model._engine
    engine._config = SimpleNamespace(max_chunks=2)
    input = SanaInput(1, None, np.tile(np.eye(4, dtype=np.float32), (24, 1, 1)))
    with pytest.raises(NoAnchor):
        engine.generate(input)

    first = engine.generate(
        replace(input, anchor=SanaAnchor(b"image", "A lake", 42, b"calibration", 49))
    )
    assert first.chunk_index == 1
    with pytest.raises(TrajectoryCompleteError) as raised:
        engine.generate(replace(input, poses=None))
    assert raised.value.chunk_index == 1 and raised.value.trajectory_frames == 49
    assert engine.generate(input).chunk_index == 2
    with pytest.raises(RolloutExhausted):
        engine.generate(input)
    engine.reset()
    with pytest.raises(NoAnchor):
        engine.generate(input)


def test_backend_failure_does_not_advance_model_progress(monkeypatch):
    from sana_wm_model import SanaAnchor, SanaInput

    model, _ = ready()
    engine = model._engine
    input = SanaInput(
        1, SanaAnchor(b"image", "Scene", 42, None, 0), np.tile(np.eye(4), (24, 1, 1))
    )

    def fail(poses):
        raise RuntimeError("unexpected decoded chunk shape")

    monkeypatch.setattr(engine._backend, "generate_chunk", fail)
    with pytest.raises(RuntimeError, match="decoded chunk shape"):
        engine.generate(input)
    assert engine._chunk_index == 0


def test_runtime_free_model_dependency_graph():
    import subprocess
    import sys

    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("reactor_runtime"):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import sana_wm_model
import sana_wm_backend
import sana_wm_assets
assert sana_wm_model.SanaModel() is not None
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )


def test_trajectory_commands_and_validation():
    import io

    from reactor_runtime import CommandError, UploadedFile

    model, _ = ready()

    def upload(array):
        data = io.BytesIO()
        np.save(data, array)
        return UploadedFile(
            name="poses.npy", mime_type="application/octet-stream", data=data.getvalue()
        )

    async def run():
        with pytest.raises(CommandError):
            await model.reset(-1)
        model._selected_image = Path("anchor.png")
        with pytest.raises(CommandError):
            await model.set_prompt(" ")
        with pytest.raises(CommandError):
            await model.set_camera_trajectory(upload(np.zeros((24, 4, 4))))
        trajectory = np.tile(np.eye(4, dtype=np.float32), (49, 1, 1))
        message = await model.set_camera_trajectory(upload(trajectory))
        assert message.available_chunks == 2
        assert (await model.process_input()).anchor.trajectory_frames == 49
        with pytest.raises(CommandError):
            await model.set_control("forward", True)
        await model.use_interactive_controls()
        assert model._trajectory is None
        assert (await model.process_input()).anchor.trajectory_frames == 0
        await model.set_control("yaw_left", True)
        await model.on_disconnected()
        assert not model.state._held_controls

    asyncio.run(run())
