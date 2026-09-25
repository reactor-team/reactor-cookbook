"""Test LingBot-World v1 session and image-selection contracts."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

import lingbot_world_v1
from lingbot_world_v1 import LingBotWorldV1
from lingbot_world_v1_camera import CameraMotionPlanner, MotionConfig
from lingbot_world_v1_model import (
    AnchorImage,
    LingbotV1Input,
    LingbotV1Model,
    LingbotV1Result,
    NoAnchor,
)
from lingbot_world_v1_types import (
    CameraMotionChanged,
    ImageSelected,
    LingBotWorldState,
    RolloutLimitReached,
    StateUpdate,
)


def _world() -> tuple[Any, list[Any]]:
    sample = SimpleNamespace(
        image=Path("sample.jpg"),
        intrinsics=Path("intrinsics.npy"),
        prompt="A calm lakeside world",
    )
    config: Any = SimpleNamespace(seed=42, samples=(sample,), max_chunks=320)
    world = LingBotWorldV1()
    world.state = LingBotWorldState()
    world._config = config
    world._default_prompt = sample.prompt
    messages: list[Any] = []

    async def record(message: Any) -> None:
        messages.append(message)

    world.send = record
    return world, messages


def test_session_waits_for_an_explicit_image_selection() -> None:
    """Expose an empty idle world until upload or random selection succeeds."""
    world, _ = _world()

    world.on_session_started()
    state = world._state_update()

    assert world._selected_input is None
    assert world._selected_intrinsics is None
    assert state.prompt == ""
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.next_chunk is None
    assert state.next_chunk_frames is None


def test_first_upload_uses_the_default_public_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow upload to initialize a world before any built-in image is selected."""
    world, messages = _world()
    world.on_session_started()
    monkeypatch.setattr(
        lingbot_world_v1, "validate_uploaded_image", lambda _image: None
    )
    upload = UploadedFile(name="anchor.png", mime_type="image/png", data=b"image")

    reply = asyncio.run(world.set_image(upload, ""))

    assert isinstance(reply, ImageSelected)
    assert reply.source == "uploaded"
    assert reply.filename == "anchor.png"
    assert reply.prompt == "A calm lakeside world"
    assert world._selected_input is upload
    assert world._selected_intrinsics == Path("intrinsics.npy")
    state = messages[-1]
    assert isinstance(state, StateUpdate)
    assert state.image_source == "uploaded"
    assert state.image_name == "anchor.png"
    assert state.next_chunk == 1
    assert state.next_chunk_frames == 9


def test_camera_change_replies_and_broadcasts_complete_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm a control change and broadcast the resulting durable state."""
    world, messages = _world()
    world.on_session_started()
    monkeypatch.setattr(
        lingbot_world_v1, "validate_uploaded_image", lambda _image: None
    )
    upload = UploadedFile(name="anchor.png", mime_type="image/png", data=b"image")
    asyncio.run(world.set_image(upload, ""))

    reply = asyncio.run(world.set_yaw(0.75))

    assert isinstance(reply, CameraMotionChanged)
    assert reply.yaw == 0.75
    assert reply.applies_to_chunk == 1
    assert isinstance(messages[-1], StateUpdate)
    assert messages[-1].yaw == 0.75


def test_camera_controls_require_an_image() -> None:
    """Reject motion that has no selected world to control."""
    world, _ = _world()
    world.on_session_started()

    with pytest.raises(CommandError):
        asyncio.run(world.set_forward(1.0))


class FakeModel:
    """Stand in for the model half: record inputs, return native chunk shapes."""

    def __init__(self) -> None:
        self.inputs: list[LingbotV1Input] = []
        self.world_id: int | None = None
        self.chunk_index = 0
        self.resets = 0

    def generate(self, input: LingbotV1Input) -> LingbotV1Result:
        self.inputs.append(input)
        if input.world_id != self.world_id:
            assert input.anchor is not None, "a fresh world must carry its anchor"
            self.world_id = input.world_id
            self.chunk_index = 0
        self.chunk_index += 1
        frames = np.zeros((9 if self.chunk_index == 1 else 12, 8, 8, 3), dtype=np.uint8)
        return LingbotV1Result(
            frames=frames, world_id=input.world_id, chunk_index=self.chunk_index
        )

    def reset(self) -> None:
        self.resets += 1
        self.world_id = None
        self.chunk_index = 0


def _loaded_world() -> tuple[Any, FakeModel, list[Any]]:
    world, messages = _world()
    model = FakeModel()
    world._engine = model
    world._planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    world.on_session_started()
    asyncio.run(world.random_image())
    return world, model, messages


def _anchors(model: FakeModel) -> list[AnchorImage]:
    return [input.anchor for input in model.inputs if input.anchor is not None]


async def _step(world: Any) -> Any:
    input = await world.process_input()
    result = world.generate(input)
    return await world.process_output(StepOutcome(result=result, elapsed=0.1))


def test_refused_step_does_not_touch_model() -> None:
    world, _ = _world()
    world.on_session_started()
    with pytest.raises(ApplicationError):
        asyncio.run(world.process_input())


def test_generate_reads_only_the_input_snapshot() -> None:
    world, model, _ = _loaded_world()
    input = asyncio.run(world.process_input())
    with pytest.raises(FrozenInstanceError):
        input.prompt = "mutated"
    world.state = None
    result = world.generate(input)
    assert result.frames.shape == (9, 8, 8, 3)
    assert model.inputs[0] is input
    assert input.anchor is not None
    assert input.anchor.image == Path("sample.jpg")
    assert input.anchor.intrinsics == Path("intrinsics.npy")
    assert input.anchor.seed == 42
    assert input.poses.shape == (3, 4, 4)


def test_anchor_crosses_once_per_world() -> None:
    world, model, _ = _loaded_world()
    asyncio.run(_step(world))
    asyncio.run(_step(world))
    assert [input.anchor is not None for input in model.inputs] == [True, False]
    assert world.state._applied_world_id == world.state._world_id


def test_uploaded_anchor_crosses_as_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    world, model, _ = _loaded_world()
    monkeypatch.setattr(
        lingbot_world_v1, "validate_uploaded_image", lambda _image: None
    )
    upload = UploadedFile(name="anchor.png", mime_type="image/png", data=b"image")
    asyncio.run(world.set_image(upload, ""))
    asyncio.run(_step(world))
    anchor = _anchors(model)[-1]
    assert anchor.image == b"image"
    assert anchor.suffix == ".png"


def test_ten_continuous_actions_preserve_rollout() -> None:
    world, model, messages = _loaded_world()

    async def run() -> None:
        for index in range(10):
            reply = await world.set_yaw(0.25 if index % 2 == 0 else -0.25)
            assert reply.applies_to_chunk == index + 1
            output = await _step(world)
            assert output.main_video.shape[0] == (9 if index == 0 else 12)
            assert world._chunk_index == index + 1
        await world.set_prompt("The same lake at sunset")
        await _step(world)

    asyncio.run(run())
    assert len(_anchors(model)) == 1
    assert len(model.inputs) == 11
    assert model.inputs[-1].prompt == "The same lake at sunset"
    assert messages[-1].next_chunk == 12
    assert "fps" not in vars(LingBotWorldV1)
    assert "inference" not in vars(LingBotWorldV1)


def test_reset_restarts_at_first_native_chunk() -> None:
    world, model, _ = _loaded_world()
    asyncio.run(_step(world))
    asyncio.run(world.set_forward(1.0))
    asyncio.run(world.reset(123))
    output = asyncio.run(_step(world))
    assert output.main_video.shape[0] == 9
    anchors = _anchors(model)
    assert len(anchors) == 2
    assert anchors[-1].seed == 123
    assert model.inputs[-1].world_id != model.inputs[0].world_id
    assert world._chunk_index == 1
    assert world.state.forward == 0
    np.testing.assert_array_equal(model.inputs[-1].poses[0], np.eye(4))


def test_limit_emits_last_chunk_then_refuses_until_reset() -> None:
    world, model, messages = _loaded_world()
    world._config.max_chunks = 2
    asyncio.run(_step(world))
    output = asyncio.run(_step(world))
    assert output.main_video.shape[0] == 12
    assert sum(isinstance(message, RolloutLimitReached) for message in messages) == 1
    with pytest.raises(ApplicationError):
        asyncio.run(world.process_input())
    with pytest.raises(CommandError):
        asyncio.run(world.set_prompt("new prompt"))
    assert len(model.inputs) == 2
    asyncio.run(world.reset(-1))
    assert asyncio.run(_step(world)).main_video.shape[0] == 9


def test_chunk_time_is_the_runtime_measurement() -> None:
    world, _, messages = _loaded_world()
    input = asyncio.run(world.process_input())
    result = world.generate(input)
    asyncio.run(world.process_output(StepOutcome(result=result, elapsed=0.37)))
    assert world._last_chunk_seconds == 0.37
    assert messages[-1].last_chunk_seconds == 0.37


def test_model_failure_propagates_without_claiming_a_completed_chunk() -> None:
    world, _, messages = _loaded_world()
    count = len(messages)
    error = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        asyncio.run(world.process_output(StepOutcome(error=error, elapsed=0.1)))
    assert world._chunk_index == 0
    assert world.state._applied_world_id is None
    assert len(messages) == count


def test_session_end_releases_rollout_and_clears_selection() -> None:
    world, model, _ = _loaded_world()
    asyncio.run(_step(world))
    world.on_session_ended()
    assert model.resets == 1
    assert world._selected_input is None
    assert world._chunk_index == 0


class FakeBackend:
    """Stand in for native GPU inference inside the model half."""

    def __init__(self) -> None:
        self.resets: list[dict[str, Any]] = []
        self.chunks: list[tuple[np.ndarray, str]] = []
        self.ended = 0

    def reset(self, **kwargs: Any) -> None:
        self.resets.append(kwargs)

    def generate_chunk(self, poses: np.ndarray, prompt: str) -> np.ndarray:
        self.chunks.append((poses, prompt))
        return np.zeros((12, 8, 8, 3), dtype=np.uint8)

    def end_session(self) -> None:
        self.ended += 1


def _model() -> tuple[LingbotV1Model, FakeBackend]:
    model = LingbotV1Model()
    backend = FakeBackend()
    model._backend = backend
    return model, backend


def _input(world_id: int, anchor: AnchorImage | None) -> LingbotV1Input:
    return LingbotV1Input(
        world_id=world_id,
        anchor=anchor,
        prompt="a lake",
        poses=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
    )


_ANCHOR = AnchorImage(
    image=b"png-bytes", suffix=".png", intrinsics=Path("intrinsics.npy"), seed=7
)


def test_model_starts_a_world_on_an_unseen_id_and_counts_its_chunks() -> None:
    model, backend = _model()
    first = model.generate(_input(1, _ANCHOR))
    second = model.generate(_input(1, None))
    assert len(backend.resets) == 1
    assert backend.resets[0] == {
        "seed": 7,
        "anchor_image": b"png-bytes",
        "suffix": ".png",
        "intrinsics": Path("intrinsics.npy"),
        "prompt": "a lake",
    }
    assert (first.world_id, first.chunk_index) == (1, 1)
    assert (second.world_id, second.chunk_index) == (1, 2)
    assert len(backend.chunks) == 2


def test_model_refuses_an_unseen_world_without_an_anchor() -> None:
    model, backend = _model()
    with pytest.raises(NoAnchor):
        model.generate(_input(1, None))
    assert backend.resets == []
    assert backend.chunks == []


def test_model_reset_forgets_the_world_so_the_anchor_crosses_again() -> None:
    model, backend = _model()
    model.generate(_input(1, _ANCHOR))
    model.reset()
    assert backend.ended == 1
    with pytest.raises(NoAnchor):
        model.generate(_input(1, None))
    result = model.generate(_input(1, _ANCHOR))
    assert result.chunk_index == 1
    assert len(backend.resets) == 2


class ProcessModel(LingbotV1Model):
    """Exercise the real process boundary with the native GPU backend faked."""

    def load(self, record: Path) -> None:
        self._backend = FakeBackend()
        record.write_text(json.dumps({"pid": os.getpid(), "rank": self.rank}))


def test_distributed_worker_keeps_ten_chunks_and_propagates_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reactor_runtime.distributed import DistributedRunner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    record = tmp_path / "worker.json"
    runner = DistributedRunner(ProcessModel, load_kwargs={"record": record})
    runner.start()
    try:
        worker = json.loads(record.read_text())
        assert worker["pid"] != os.getpid()
        assert worker["rank"] == 0
        for index in range(10):
            result = runner.generate(_input(1, _ANCHOR if index == 0 else None))
            assert result.world_id == 1
            assert result.chunk_index == index + 1
            assert result.frames.dtype == np.uint8
            assert result.frames.flags.c_contiguous
        runner.reset()
        with pytest.raises(NoAnchor):
            runner.generate(_input(1, None))
        assert runner.healthy
        assert runner.generate(_input(2, _ANCHOR)).chunk_index == 1
    finally:
        runner.shutdown()
    assert not runner.healthy


def test_model_dependency_graph_imports_without_runtime() -> None:
    source = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import lingbot_world_v1_model
"""
    subprocess.run(
        [sys.executable, "-c", source], check=True, cwd=Path(__file__).parents[1]
    )


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_app_starts_one_runner_with_plain_load_settings(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    calls = []

    class FakeRunner:
        def __init__(self, worker_cls, **kwargs):
            calls.append((worker_cls, kwargs))

        def start(self):
            calls.append("started")

    monkeypatch.setattr(lingbot_world_v1, "DistributedRunner", FakeRunner)
    monkeypatch.setattr(lingbot_world_v1, "prepare_runtime", lambda config: None)
    config_path = Path(__file__).parents[1] / "lingbot_world_v1.yaml"
    config = replace(lingbot_world_v1.read_config(config_path), world_size=world_size)
    monkeypatch.setattr(lingbot_world_v1, "read_config", lambda path: config)
    world = LingBotWorldV1()
    world.load(config_path)
    worker_cls, options = calls[0]
    assert worker_cls is LingbotV1Model
    assert options["world_size"] == world_size
    settings = options["load_kwargs"]["settings"]
    assert settings.context_latents == 21
    assert settings.max_chunks == 320
    assert calls[1] == "started"


@pytest.mark.parametrize("world_size", ["0", "3", "8", "true", "'2'"])
def test_config_rejects_unsupported_world_sizes(
    tmp_path: Path, world_size: str
) -> None:
    config_path = tmp_path / "model.yaml"
    source = (Path(__file__).parents[1] / "lingbot_world_v1.yaml").read_text()
    config_path.write_text(source.replace("world_size: 1", f"world_size: {world_size}"))
    with pytest.raises(ValueError, match="world_size"):
        lingbot_world_v1.read_config(config_path)


def test_model_passes_runner_rank_to_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot_world_v1_model

    calls = []
    backend = FakeBackend()

    def factory(settings, *, rank, world_size):
        calls.append((settings, rank, world_size))
        return backend

    monkeypatch.setattr(lingbot_world_v1_model, "LingBotBackend", factory)
    model = LingbotV1Model()
    model.rank, model.world_size = 3, 4
    settings = object()
    model.load(settings)
    assert calls == [(settings, 3, 4)]
