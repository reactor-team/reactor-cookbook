"""Exercise application boundaries and native bookkeeping without GPU weights."""

import asyncio
import io
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from alayaworld import AlayaWorld
from alayaworld_model import (
    AlayaInput,
    AlayaResult,
    AlayaWorldModel,
    InvalidTrajectoryError,
    NoAnchorError,
    RolloutExhaustedError,
)
from alayaworld_types import AlayaWorldState
from PIL import Image
from reactor_runtime import ApplicationError, StepOutcome, UploadedFile


def test_recipe_uses_pre_optimization_inference(tmp_path):
    from alayaworld_assets import read_config

    root = Path(__file__).parents[1]
    config = read_config(root / "alayaworld.yaml", tmp_path)
    assert config.attention_backend == "pytorch"
    assert config.bank_taehv is False
    assert config.taehv_path is None
    assert config.taehv_source_path is None
    assert config.warmup_chunks == 0
    assert config.compile_mode == "default"
    assert config.flex_attention is True
    manifest = yaml.safe_load((root / "reactor.yaml").read_text())
    assert not any("flash-attn" in step for step in manifest["build"].get("run", []))


@pytest.fixture
def app(monkeypatch):
    app = AlayaWorld()
    app.state = AlayaWorldState()
    app._config = SimpleNamespace(
        max_chunks_per_rollout=512,
        seed=7,
        strafe_units_per_second=0.126,
        vertical_units_per_second=0.261,
        forward_units_per_second=1.905,
        pitch_degrees_per_second=4.039,
        yaw_degrees_per_second=9.375,
        roll_degrees_per_second=4.094,
    )
    app.messages = []

    async def send(message):
        app.messages.append(message)

    monkeypatch.setattr(app, "send", send)
    monkeypatch.setattr(app.output, "flush", lambda: None)
    return app


def setup_result(world_id=1):
    return AlayaResult(world_id, None, 0, "Scene", 39, np.eye(4, dtype=np.float32))


def test_refusal_never_reaches_model(app, monkeypatch):
    monkeypatch.setattr(app.engine, "generate", lambda _: pytest.fail("model called"))
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())


def test_generate_only_forwards_input(app, monkeypatch):
    snapshot = AlayaInput(3, "snapshot", 7, Path("anchor"))
    marker = object()
    received = []
    monkeypatch.setattr(
        app.engine, "generate", lambda value: received.append(value) or marker
    )
    app.state.prompt = "unrelated"
    assert app.generate(snapshot) is marker
    assert received == [snapshot]


def test_anchor_once_and_ten_continuous_actions(app, monkeypatch):
    app._selected_input = Path("anchor")
    app.state._world_id = 1
    snapshots = []

    def generate(value):
        snapshots.append(value)
        if value.image is not None:
            return setup_result(value.world_id)
        return AlayaResult(
            value.world_id,
            np.zeros((32, 8, 8, 3), np.uint8),
            len(snapshots) - 1,
            value.prompt,
            32,
        )

    monkeypatch.setattr(app.engine, "generate", generate)

    async def run():
        snapshot = await app.process_input()
        assert snapshot.image == Path("anchor") and snapshot.trajectory is None
        assert (
            await app.process_output(StepOutcome(result=app.generate(snapshot))) is None
        )
        assert app._ar_index == 0
        for index in range(10):
            app.state.prompt = f"Scene {index}"
            app.state.strafe = index / 10
            snapshot = await app.process_input()
            assert snapshot.image is None
            assert snapshot.trajectory.shape == (39 if index == 0 else 32, 4, 4)
            with pytest.raises(FrozenInstanceError):
                snapshot.prompt = "mutated"
            app.state.prompt = "later input"
            result = app.generate(snapshot)
            assert result.active_prompt == f"Scene {index}"
            output = await app.process_output(StepOutcome(result=result, elapsed=0.25))
            assert output.main_video.shape == (32, 8, 8, 3)
        assert app._ar_index == 10
        assert sum(item.image is not None for item in snapshots) == 1
        await app.reset(-1)
        assert (await app.process_input()).image == Path("anchor")

    asyncio.run(run())


def test_error_does_not_acknowledge_or_count(app):
    app._reset_in_flight = app._chunk_in_flight = True
    app.state._world_id = 5
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(
            app.process_output(StepOutcome(error=RuntimeError("native failure")))
        )
    assert app.state._applied_world_id is None and app._ar_index == 0
    assert not app._reset_in_flight and not app._chunk_in_flight
    assert app.messages == []


def test_failed_chunk_does_not_advance_camera(app):
    app._selected_input = Path("anchor")
    app.state._world_id = 1
    asyncio.run(app.process_output(StepOutcome(result=setup_result())))
    before = app._camera._current_c2w.copy()
    app.state.forward = 1
    asyncio.run(app.process_input())
    with pytest.raises(RuntimeError):
        asyncio.run(app.process_output(StepOutcome(error=RuntimeError("failed"))))
    np.testing.assert_array_equal(app._camera._current_c2w, before)
    assert app._ar_index == 0


def test_model_error_propagates_through_both_hooks(app, monkeypatch):
    failure = RuntimeError("failed native turn")

    def generate(_):
        raise failure

    monkeypatch.setattr(app.engine, "generate", generate)
    app._selected_input = Path("anchor")
    snapshot = asyncio.run(app.process_input())
    try:
        app.generate(snapshot)
    except RuntimeError as error:
        outcome = StepOutcome(error=error)
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(app.process_output(outcome))
    assert raised.value is failure
    assert app._ar_index == 0 and app.state._applied_world_id is None


def test_manual_reset_flushes_and_queues_seed(app, monkeypatch):
    app._selected_input = Path("anchor")
    app.state.forward = 1
    flushes = []
    monkeypatch.setattr(app.output, "flush", lambda: flushes.append(True))
    previous = app.state._world_id
    asyncio.run(app.reset(19))
    assert flushes == [True] and app._seed == 19
    assert app.state._world_id == previous + 1 and app.state.forward == 0


def test_upload_resolves_to_bytes_and_prompt_preserves_world(app):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16)).save(stream, format="PNG")
    image = UploadedFile(
        name="scene.png", mime_type="image/png", data=stream.getvalue()
    )

    async def run():
        await app.set_image(image, " A scene ")
        snapshot = await app.process_input()
        assert snapshot.image == image.data and isinstance(snapshot.image, bytes)
        world_id = snapshot.world_id
        await app.set_prompt(" Updated scene ")
        assert app.state._world_id == world_id
        assert app.state.prompt == "Updated scene"

    asyncio.run(run())


def test_cap_requests_new_world_and_session_cleanup(app, monkeypatch):
    app._selected_input = Path("anchor")
    app._config.max_chunks_per_rollout = 1
    app.state._world_id = 1
    asyncio.run(app.process_output(StepOutcome(result=setup_result())))
    result = AlayaResult(1, np.zeros((32, 8, 8, 3), np.uint8), 1, "Scene", 32)
    asyncio.run(app.process_output(StepOutcome(result=result)))
    assert app.state._world_id == 2 and app.state._applied_world_id == 1
    assert any(
        getattr(m, "trigger", "") == "automatic_chunk_limit" for m in app.messages
    )
    reset = []
    monkeypatch.setattr(app.engine, "reset", lambda: reset.append(True))
    app.on_session_ended()
    assert reset == [True] and app._selected_input is None and app._camera is None
    assert app._ar_index == 0 and app.state._applied_world_id is None


@pytest.fixture
def native_model(monkeypatch):
    model = AlayaWorldModel()
    model._config = SimpleNamespace(
        max_spatial_frames=320, recent_spatial_frames=160, max_chunks_per_rollout=512
    )
    model._engine = SimpleNamespace(encode_caption=lambda prompt: prompt)
    model._alaya_pipeline = SimpleNamespace(
        cfg=SimpleNamespace(sample=SimpleNamespace(temporal_stride=8)),
        generate=lambda *_: object(),
        finalize=lambda *_: None,
    )
    model._cache = SimpleNamespace(
        history=object(),
        K=4,
        target_start=lambda index: 20 + index * 4,
    )
    monkeypatch.setattr(model, "_write_camera_trajectory", lambda *_: None)
    monkeypatch.setattr(
        model, "_decode_new_frames", lambda *_: np.zeros((32, 8, 8, 3), np.uint8)
    )
    return model


def test_native_world_setup_and_ten_chunks(native_model, monkeypatch):
    model = native_model
    starts = []

    def initialize(prompt, seed, image):
        starts.append((prompt, seed, image))
        model._ar_index = 0
        model._active_prompt = prompt
        return np.eye(4, dtype=np.float32)

    monkeypatch.setattr(model, "_reset_rollout", initialize)
    result = model.generate(AlayaInput(1, "Scene", 7, b"image"))
    assert result.frames is None and result.frames_wanted == 39
    assert not hasattr(result, "input")
    for index in range(10):
        result = model.generate(
            AlayaInput(1, "Scene", 7, trajectory=np.zeros((32, 4, 4)))
        )
        assert result.completed_chunks == index + 1 and result.frames_wanted == 32
    assert len(starts) == 1
    model.generate(AlayaInput(2, "New", 9, b"next"))
    assert len(starts) == 2 and model._ar_index == 0
    model.reset()
    assert model._world_id is None and model._cache is None and model._ar_index == 0


@pytest.mark.parametrize("failure", ["generate", "decode", "shape"])
def test_native_failure_does_not_count(native_model, monkeypatch, failure):
    def fail(*_):
        raise RuntimeError("native failure")

    if failure == "generate":
        monkeypatch.setattr(native_model._alaya_pipeline, "generate", fail)
    elif failure == "decode":
        monkeypatch.setattr(native_model, "_decode_new_frames", fail)
    else:
        monkeypatch.setattr(
            native_model,
            "_decode_new_frames",
            lambda *_: np.zeros((32, 8, 8), np.uint8),
        )
    with pytest.raises(RuntimeError):
        native_model._generate_chunk("Scene", np.zeros((32, 4, 4)))
    assert native_model._ar_index == 0


def test_failed_initialization_does_not_acknowledge(native_model, monkeypatch):
    def fail(*_):
        raise RuntimeError("setup failure")

    monkeypatch.setattr(native_model, "_reset_rollout", fail)
    with pytest.raises(RuntimeError):
        native_model.generate(AlayaInput(8, "Scene", 7, b"image"))
    assert native_model._world_id is None and native_model._ar_index == 0


def test_model_rejects_missing_anchor_and_trajectory(native_model):
    with pytest.raises(NoAnchorError):
        native_model.generate(AlayaInput(1, "Scene", 7))
    native_model._world_id = 1
    with pytest.raises(InvalidTrajectoryError):
        native_model.generate(AlayaInput(1, "Scene", 7))
    assert native_model._ar_index == 0


def test_model_owns_rollout_limit(native_model):
    native_model._world_id = 1
    native_model._ar_index = native_model._config.max_chunks_per_rollout
    with pytest.raises(RolloutExhaustedError):
        native_model.generate(
            AlayaInput(1, "Scene", 7, trajectory=np.zeros((32, 4, 4)))
        )
    assert native_model._ar_index == 512


def test_model_dependency_graph_has_no_runtime():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('reactor_runtime'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from alayaworld_model import AlayaWorldModel
AlayaWorldModel()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
