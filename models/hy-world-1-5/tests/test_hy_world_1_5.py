"""Focused schema, control, and lifecycle tests for the HY-World 1.5 adapter."""

from __future__ import annotations

import asyncio
import importlib
import io
import subprocess
import sys
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pytest import MonkeyPatch
from reactor_runtime import ApplicationError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

RECIPE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(RECIPE_DIR))

pipeline_module = importlib.import_module("hy_world_1_5")
camera_module = importlib.import_module("hy_world_1_5_camera")
types_module = importlib.import_module("hy_world_1_5_types")
assets_module = importlib.import_module("hy_world_1_5_assets")

HYWorld15 = pipeline_module.HYWorld15
CameraControl = camera_module.CameraControl
NativeCameraPlanner = camera_module.NativeCameraPlanner
HYWorld15Output = types_module.HYWorld15Output
HYWorld15State = types_module.HYWorld15State
assemble_base_model = assets_module.assemble_base_model


class _Backend:
    def __init__(self) -> None:
        self.resets: list[tuple[str, int]] = []
        self.calls: list[tuple[Any, str]] = []
        self.index = 0
        self.ended = 0

    def reset(self, *, image: Image.Image, prompt: str, seed: int) -> None:
        self.resets.append((prompt, seed))
        self.index = 0

    def generate_chunk(self, camera: Any, prompt: str) -> np.ndarray:
        self.calls.append((camera, prompt))
        count = 13 if self.index == 0 else 16
        self.index += 1
        return np.zeros((count, 8, 8, 3), dtype=np.uint8)

    def end_session(self) -> None:
        self.ended += 1


def _image_upload() -> UploadedFile:
    stream = io.BytesIO()
    Image.new("RGB", (32, 18), color=(10, 20, 30)).save(stream, format="PNG")
    return UploadedFile(name="world.png", mime_type="image/png", data=stream.getvalue())


def _ready_model() -> tuple[Any, _Backend]:
    model = HYWorld15()
    model.state = HYWorld15State()
    model._config = SimpleNamespace(max_chunks=512, seed=1)
    backend = _Backend()
    model._engine._backend = backend
    model._engine._config = model._config
    model._planner = NativeCameraPlanner()
    model._examples = ()
    model.on_session_started()

    async def discard(_message: Any) -> None:
        return None

    model.send = discard
    return model, backend


def test_contract_is_atomic_and_documents_command_results() -> None:
    """Expose one camera setter and complete success and failure semantics."""
    contract = ModelContract.of(HYWorld15)

    assert set(contract.commands) == {
        "set_prompt",
        "set_camera",
        "release_camera",
        "reset",
        "set_image",
        "random_image",
    }
    assert all("Emits" in command.description for command in contract.commands.values())
    assert all(
        field.info.description
        for command in contract.commands.values()
        for field in command.command.__command_fields__.values()
    )
    assert "fps" not in HYWorld15.__dict__
    assert HYWorld15.buffer_size == 16


def test_native_camera_keeps_anchor_and_dual_action_alignment() -> None:
    """Match upstream latent cadence, motion scale, intrinsics, and action labels."""
    planner = NativeCameraPlanner()
    first = planner.plan(CameraControl(forward=1.0, strafe=0.0, pitch=0.0, yaw=0.0))

    np.testing.assert_allclose(first.viewmats[0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(
        first.viewmats[:, 2, 3], [0.0, -0.08, -0.16, -0.24], atol=1e-6
    )
    np.testing.assert_array_equal(first.actions, [0, 9, 9, 9])
    np.testing.assert_allclose(first.intrinsics[0, 0, 0], 0.5050505)
    np.testing.assert_allclose(first.intrinsics[0, 1, 1], 0.89786756)

    diagonal = planner.plan(CameraControl(forward=1.0, strafe=1.0, pitch=1.0, yaw=1.0))
    np.testing.assert_array_equal(diagonal.actions, [50, 50, 50, 50])

    planner.reset()
    deadzone = planner.plan(
        CameraControl(forward=0.001, strafe=0.0, pitch=0.01, yaw=0.0)
    )
    np.testing.assert_array_equal(deadzone.actions, [0, 0, 0, 0])


def test_model_layout_links_remain_valid_after_weights_root_moves() -> None:
    """Keep assembled encoder paths portable across container bind mounts."""
    with tempfile.TemporaryDirectory() as directory:
        weights = Path(directory) / "weights"
        base_model = weights / "models/base"
        qwen = weights / "models/qwen"
        byt5 = weights / "models/byt5"
        glyph = weights / "models/glyph"
        vision = weights / "models/vision"
        for path in (qwen, byt5, glyph, vision):
            path.mkdir(parents=True)
        config = SimpleNamespace(
            base_model=SimpleNamespace(path=base_model),
            qwen=SimpleNamespace(path=qwen),
            byt5=SimpleNamespace(path=byt5),
            glyph=SimpleNamespace(path=glyph),
            flux_vision=SimpleNamespace(path=vision),
        )

        assemble_base_model(config)

        destinations = (
            base_model / "text_encoder/llm",
            base_model / "text_encoder/byt5-small",
            base_model / "text_encoder/Glyph-SDXL-v2",
            base_model / "vision_encoder/siglip",
        )
        assert all(path.is_symlink() for path in destinations)
        assert all(not Path(path.readlink()).is_absolute() for path in destinations)


def test_image_selection_queues_and_generates_first_chunk(
    monkeypatch: MonkeyPatch,
) -> None:
    """Generate continuously from the fresh world queued by image selection."""
    model, backend = _ready_model()
    flushes: list[None] = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    reply = asyncio.run(model.set_image(_image_upload(), "A quiet road"))

    assert reply.source == "uploaded"
    assert reply.prompt == "A quiet road"
    assert model.state._world_id != model.state._applied_world_id

    async def generate_first_chunk() -> Any:
        snapshot = await model.process_input()
        return await model.process_output(StepOutcome(result=model.generate(snapshot)))

    output = asyncio.run(generate_first_chunk())
    assert isinstance(output, HYWorld15Output)
    assert output.main_video.shape == (13, 8, 8, 3)
    assert flushes == [None]
    assert backend.resets == [("A quiet road", 1)]
    assert len(backend.calls) == 1
    assert model._chunk_index == 1


def test_rollout_reset_flushes_pending_media(monkeypatch: MonkeyPatch) -> None:
    """Discard frames queued by the world that a reset replaces."""
    model, _backend = _ready_model()
    flushes: list[None] = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))
    asyncio.run(model.set_image(_image_upload(), "A quiet road"))
    flushes.clear()

    asyncio.run(model.reset(-1))

    assert flushes == [None]


def test_prompt_applies_at_next_chunk_boundary() -> None:
    """Sample a queued prompt exactly at the next chunk boundary."""
    model, backend = _ready_model()
    asyncio.run(model.set_image(_image_upload(), "First prompt"))

    async def drain_first_chunk() -> None:
        snapshot = await model.process_input()
        await model.process_output(StepOutcome(result=model.generate(snapshot)))

    asyncio.run(drain_first_chunk())
    asyncio.run(model.set_prompt("Second prompt"))

    async def generate_next_chunk() -> Any:
        snapshot = await model.process_input()
        return await model.process_output(StepOutcome(result=model.generate(snapshot)))

    output = asyncio.run(generate_next_chunk())
    assert isinstance(output, HYWorld15Output)
    assert output.main_video.shape == (16, 8, 8, 3)
    assert [prompt for _, prompt in backend.calls] == ["First prompt", "Second prompt"]
    assert model._chunk_index == 2


def test_step_snapshot_waiting_limit_and_error_cleanup() -> None:
    """Gate unavailable worlds and keep generation independent of shared inputs."""
    model, backend = _ready_model()

    async def run() -> None:
        with pytest.raises(ApplicationError):
            await model.process_input()
        await model.set_image(_image_upload(), "First prompt")
        snapshot = await model.process_input()
        model.state.prompt = "Second prompt"
        await model.process_output(StepOutcome(result=model.generate(snapshot)))
        assert backend.calls[0][1] == "First prompt"
        assert model._active_prompt == "First prompt"
        model.state._limit_reached = True
        with pytest.raises(ApplicationError):
            await model.process_input()
        model._generating = True
        with pytest.raises(RuntimeError, match="failed"):
            await model.process_output(StepOutcome(error=RuntimeError("failed")))
        assert not model._generating
        assert model._chunk_index == 1

    asyncio.run(run())


class _Engine:
    """Record plain model inputs and return scripted native-length results."""

    def __init__(self) -> None:
        self.inputs = []
        self.world_id = None
        self.index = 0
        self.resets = 0
        self.error = None

    def generate(self, input):
        self.inputs.append(input)
        if self.error is not None:
            raise self.error
        if input.world_id != self.world_id:
            self.index = 0
            self.world_id = input.world_id
        self.index += 1
        return pipeline_module.HYWorld15Result(
            frames=np.zeros((13 if self.index == 1 else 16, 8, 8, 3), np.uint8),
            world_id=input.world_id,
            chunk_index=self.index,
            prompt=input.prompt,
            complete=self.index == 10,
        )

    def reset(self):
        self.resets += 1


def _fake_app():
    app, _ = _ready_model()
    engine = _Engine()
    app._engine = engine
    messages = []

    async def send(message):
        messages.append(message)

    app.send = send
    return app, engine, messages


def test_all_refusals_never_call_model():
    app, engine, _ = _fake_app()

    async def run():
        with pytest.raises(ApplicationError, match="Select an image"):
            await app.process_input()
        await app.set_image(_image_upload(), "World")
        app.state.prompt = " "
        with pytest.raises(ApplicationError, match="non-empty prompt"):
            await app.process_input()
        app.state.prompt = "World"
        app.state._limit_reached = True
        with pytest.raises(ApplicationError, match="rollout limit"):
            await app.process_input()
        assert not engine.inputs

    asyncio.run(run())


def test_generate_reads_only_frozen_input():
    app, engine, _ = _fake_app()

    async def run():
        await app.set_image(_image_upload(), "World")
        input = await app.process_input()
        with pytest.raises(FrozenInstanceError):
            input.prompt = "changed"
        app.state = None
        app._planner = None
        app._config = None
        app._selected_input = None
        result = app.generate(input)
        assert engine.inputs == [input]
        assert result.prompt == "World"

    asyncio.run(run())


def test_ten_actions_anchor_ack_limit_and_runtime_timing():
    app, engine, messages = _fake_app()

    async def run():
        await app.set_image(_image_upload(), "World")
        for index in range(10):
            await app.set_camera(
                forward=(index % 3 - 1) / 2, strafe=0, pitch=0, yaw=(index % 5 - 2) / 4
            )
            input = await app.process_input()
            assert (input.anchor is not None) == (index == 0)
            assert input.viewmats.shape == (4, 4, 4)
            messages.clear()
            output = await app.process_output(
                StepOutcome(result=app.generate(input), elapsed=1.25)
            )
            assert output.main_video.shape[0] == (13 if index == 0 else 16)
            assert messages[0].generation_seconds == 1.25
            assert app._chunk_index == index + 1
        assert len({input.world_id for input in engine.inputs}) == 1
        assert app.state._limit_reached
        assert type(messages[-2]).__name__ == "RolloutLimitReached"
        with pytest.raises(ApplicationError):
            await app.process_input()

    asyncio.run(run())


def test_failed_step_does_not_ack_or_complete():
    app, engine, messages = _fake_app()
    engine.error = RuntimeError("GPU failed")

    async def run():
        await app.set_image(_image_upload(), "World")
        input = await app.process_input()
        messages.clear()
        try:
            app.generate(input)
        except RuntimeError as error:
            with pytest.raises(RuntimeError, match="GPU failed"):
                await app.process_output(StepOutcome(error=error))
        assert app.state._applied_world_id is None
        assert app._chunk_index == 0
        assert not app._generating and not messages
        assert (await app.process_input()).anchor is not None

    asyncio.run(run())


def test_prompt_reset_upload_and_session_cleanup():
    app, engine, _ = _fake_app()

    async def step():
        input = await app.process_input()
        await app.process_output(StepOutcome(result=app.generate(input)))
        return input

    async def run():
        await app.set_image(_image_upload(), "First")
        first = await step()
        assert isinstance(first.anchor.image, bytes)
        await app.set_prompt("Second")
        second = await step()
        assert second.world_id == first.world_id and second.anchor is None
        assert second.prompt == "Second"
        await app.reset(42)
        reset = await step()
        assert reset.world_id != first.world_id and reset.anchor.seed == 42
        np.testing.assert_allclose(reset.viewmats[0], np.eye(4))
        await app.set_image(_image_upload(), "Third")
        replaced = await step()
        assert replaced.world_id != reset.world_id
        await app.set_camera(1, 1, 1, 1)
        await app.release_camera()
        assert app.state._forward == app.state._yaw == 0
        app.on_session_ended()
        assert engine.resets == 1 and app._selected_input is None
        assert app.state._applied_world_id is None

    asyncio.run(run())


def test_model_owns_world_counter_and_native_prompt_updates():
    from hy_world_1_5_model import (
        HYWorld15Anchor,
        HYWorld15Input,
        HYWorld15Model,
        NoAnchor,
        RolloutExhausted,
    )

    engine = HYWorld15Model()
    backend = _Backend()
    engine._backend = backend
    engine._config = SimpleNamespace(max_chunks=2)
    camera = NativeCameraPlanner().plan(CameraControl(0, 0, 0, 0))

    def input(world, anchor=None, prompt="World"):
        return HYWorld15Input(
            world, anchor, prompt, camera.viewmats, camera.intrinsics, camera.actions
        )

    with pytest.raises(NoAnchor):
        engine.generate(input(1))
    anchor = HYWorld15Anchor(_image_upload().data, 42)
    first = engine.generate(input(1, anchor))
    second = engine.generate(input(1, prompt="Changed"))
    assert first.chunk_index == 1 and second.complete
    assert backend.resets == [("World", 42)]
    assert backend.calls[-1][1] == "Changed"
    np.testing.assert_array_equal(backend.calls[0][0].actions, camera.actions)
    with pytest.raises(RolloutExhausted):
        engine.generate(input(1))
    assert len(backend.resets) == 1
    fresh = engine.generate(input(2, anchor))
    assert fresh.chunk_index == 1 and fresh.frames.shape[0] == 13
    engine.reset()
    assert backend.ended == 1
    with pytest.raises(NoAnchor):
        engine.generate(input(2))


def test_model_dependency_graph_does_not_import_runtime():
    source = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "reactor_runtime" or name.startswith("reactor_runtime."):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import hy_world_1_5_model
import hy_world_1_5_backend
"""
    subprocess.run([sys.executable, "-c", source], cwd=RECIPE_DIR, check=True)


def test_builtin_image_selection_and_path_anchor(tmp_path):
    app, backend = _ready_model()
    path = tmp_path / "world.png"
    path.write_bytes(_image_upload().data)
    app._examples = (assets_module.ExampleImage(path, "Built-in world"),)

    async def run():
        reply = await app.random_image()
        assert reply.source == "built_in"
        input = await app.process_input()
        assert input.anchor.image == path
        await app.process_output(StepOutcome(result=app.generate(input)))
        assert backend.resets == [("Built-in world", 1)]
        assert (await app.process_input()).anchor is None

    asyncio.run(run())
