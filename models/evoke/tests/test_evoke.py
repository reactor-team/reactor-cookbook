"""Test the EVOKE Reactor contract and chunk boundary without loading weights."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import evoke_config
import numpy as np
import pytest
import yaml
from evoke import Evoke
from evoke_camera import CameraMotionPlanner, MotionConfig
from evoke_images import validate_uploaded_image, validate_uploaded_pose
from evoke_model import (
    EvokeAnchor,
    EvokeInput,
    EvokeModel,
    EvokeResult,
    NoAnchor,
    RolloutExhausted,
)
from evoke_types import CommandApplied, EvokeOutput, EvokeState, StateUpdate
from reactor_runtime import ApplicationError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

EXAMPLE_DIR = Path(__file__).parents[1]
STABILITY_PROMPT = evoke_config.read_config(EXAMPLE_DIR / "evoke.yaml").stability_prompt


def test_session_waits_for_explicit_conditioning():
    model, backend, _ = _ready_model()
    model._config = SimpleNamespace(
        seed=42, stability_prompt=STABILITY_PROMPT, max_chunks=512
    )
    model.on_session_started()
    assert model._media is None and model._input_source == "none"
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    assert not backend.reset_calls and not backend.generate_calls


CACHE_ENVIRONMENT = {
    "UV_CACHE_DIR": ".cache/uv",
    "UV_PYTHON_INSTALL_DIR": ".cache/python",
    "XDG_CACHE_HOME": ".cache/xdg",
    "HF_HOME": ".cache/huggingface",
    "TORCH_HOME": ".cache/torch",
    "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
    "TRITON_CACHE_DIR": ".cache/triton",
    "CUTE_DSL_CACHE_DIR": ".cache/cute-dsl",
    "TMPDIR": ".cache/tmp",
}


class _Backend:
    def __init__(self) -> None:
        self.reset_calls: list[dict[str, Any]] = []
        self.generate_calls: list[tuple[np.ndarray | None, int, str]] = []
        self.mode = "i2v"
        self.index = 0
        self.ended = 0

    def reset(self, **values: Any) -> None:
        self.reset_calls.append(values)
        self.mode = values["mode"]
        self.index = 0

    def generate_chunk(
        self,
        trajectory: np.ndarray | None,
        *,
        seed: int,
        prompt: str,
    ) -> np.ndarray:
        self.generate_calls.append((trajectory, seed, prompt))
        count = 33 if self.mode == "t2v" and self.index == 0 else 36
        self.index += 1
        return np.zeros((count, 384, 640, 3), dtype=np.uint8)

    def end_session(self):
        self.ended += 1


def _ready_model() -> tuple[Any, _Backend, list[Any]]:
    model = Evoke()
    model.state = EvokeState()
    model.state.prompt = "A coral reef"
    model._config = SimpleNamespace(max_chunks=512)
    model._stability_prompt = STABILITY_PROMPT
    model._media = Path("image.jpg")
    model._input_source = "built_in"
    model._input_name = "image.jpg"
    model._planner = CameraMotionPlanner(
        MotionConfig(
            fps=24,
            translation_units_per_second=1.0,
            rotation_degrees_per_second=6.0,
        )
    )
    backend = _Backend()
    model._engine._backend = backend
    model._engine._max_chunks = 512
    messages: list[Any] = []

    async def record(message: Any) -> None:
        messages.append(message)

    model.send = record
    return model, backend, messages


def test_contract_documents_commands_messages_and_video_track() -> None:
    """Expose every control and response through a polished Reactor schema."""
    contract = ModelContract.of(Evoke)
    assert set(contract.commands) == {
        "reset",
        "set_forward",
        "set_image",
        "set_pitch",
        "set_prompt",
        "set_reference_video",
        "set_roll",
        "set_strafe",
        "set_vertical",
        "set_yaw",
        "start_text",
    }
    assert all(
        getattr(contract.lifecycle, hook) is not None
        for hook in ("session_started", "session_ended", "connected", "disconnected")
    )
    assert all("Emits" in command.description for command in contract.commands.values())
    assert all(
        field.info.description
        for command in contract.commands.values()
        for field in command.command.__command_fields__.values()
    )

    document = contract.render_schema().to_openapi()
    assert document["x-reactor"]["tracks"] == [
        {"name": "main_video", "kind": "video", "direction": "out"}
    ]
    assert set(document["webhooks"]) == {
        "command_applied",
        "rollout_restarted",
        "state_update",
    }
    assert all(
        webhook["post"]["summary"].startswith("Emitted ")
        for webhook in document["webhooks"].values()
    )
    for name in ("StateUpdate", "CommandApplied", "RolloutRestarted"):
        properties = document["components"]["schemas"][name]["properties"]
        assert all(
            property_schema.get("description")
            for property_schema in properties.values()
        )


def test_reactor_manifest_declares_generated_gpu_build() -> None:
    """Build the GPU recipe from the versioned Reactor manifest."""
    document = yaml.safe_load((EXAMPLE_DIR / "reactor.yaml").read_text())

    assert document["$schema"] == "reactor/v1"
    assert document["model"]["resources"]["gpu"]["count"] == 1
    assert document["runtime"]["weights_path"] == "~/.cache/reactor_registry/evoke"
    assert document["build"]["runtime_version"] == "3.5.0"
    assert document["build"]["python_requirements"] == "requirements.txt"
    assert "git" in document["build"]["system_packages"]
    assert not (EXAMPLE_DIR / "Dockerfile").exists()


def test_prepare_runtime_defaults_caches_to_weights_volume(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Keep every large download and generated kernel cache on persistent storage."""
    weights_root = tmp_path / "weights"
    base = evoke_config.read_config(EXAMPLE_DIR / "evoke.yaml")
    config = replace(
        base,
        source_path=weights_root / "Evoke",
        worker_python=weights_root / "Evoke" / evoke_config.WORKER_PYTHON,
    )
    for variable in CACHE_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(evoke_config, "ensure_source_checkout", lambda _: None)
    monkeypatch.setattr(evoke_config, "ensure_worker_environment", lambda _: None)
    monkeypatch.setattr(evoke_config, "_ensure_model_assets", lambda _: None)
    monkeypatch.setattr(evoke_config, "_validate_runtime_paths", lambda _: None)

    evoke_config.prepare_runtime(config)

    for variable, relative in CACHE_ENVIRONMENT.items():
        expected = weights_root / relative
        assert os.environ[variable] == str(expected)
        assert expected.is_dir()


def test_camera_chunks_are_absolute_and_continuous() -> None:
    """Continue six-axis absolute camera poses across native chunks."""
    planner = CameraMotionPlanner(
        MotionConfig(
            fps=24,
            translation_units_per_second=1.0,
            rotation_degrees_per_second=6.0,
        )
    )
    first = planner.plan_chunk(
        strafe=0.0,
        vertical=0.0,
        forward=1.0,
        pitch=0.0,
        yaw=0.5,
        roll=0.0,
        frame_count=36,
    )
    second = planner.plan_chunk(
        strafe=1.0,
        vertical=0.0,
        forward=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        frame_count=36,
    )

    assert first.shape == second.shape == (36, 4, 4)
    np.testing.assert_allclose(first[0], np.eye(4), atol=1e-6)
    assert np.linalg.norm(second[0, :3, 3] - first[-1, :3, 3]) < 0.05
    assert np.isfinite(np.concatenate([first, second])).all()


def test_camera_change_replies_and_broadcasts_complete_state() -> None:
    """Confirm a control change and broadcast the resulting durable state."""
    model, _, messages = _ready_model()

    reply = asyncio.run(model.set_yaw(0.75))

    assert isinstance(reply, CommandApplied)
    assert reply.action == "set_yaw"
    assert reply.applies_to_chunk == 1
    assert "yaw=0.75" in reply.detail
    assert isinstance(messages[-1], StateUpdate)
    assert messages[-1].yaw == 0.75


def test_inference_requests_exactly_one_native_chunk() -> None:
    """Preserve one worker call per 36-frame camera-conditioned turn."""
    model, backend, messages = _ready_model()

    async def collect() -> list[Any]:
        snapshot = await model.process_input()
        output = await model.process_output(
            StepOutcome(result=model.generate(snapshot))
        )
        return [output]

    outputs = asyncio.run(collect())

    assert len(backend.reset_calls) == 1
    assert len(backend.generate_calls) == 1
    trajectory, seed, prompt = backend.generate_calls[0]
    assert trajectory is not None and trajectory.shape == (36, 4, 4)
    assert seed == 42
    assert prompt == "A coral reef"
    assert all(isinstance(output, EvokeOutput) for output in outputs)
    assert outputs[0].main_video.shape == (36, 384, 640, 3)
    assert any(isinstance(message, StateUpdate) for message in messages)


def test_rollout_restart_flushes_pending_media(monkeypatch: Any) -> None:
    """Discard frames queued by the world that a restart replaces."""
    model, _, _ = _ready_model()
    flushes: list[None] = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    model._request_restart()

    assert flushes == [None]


def test_public_example_image_passes_upload_validation() -> None:
    """Accept the bundled public image through the same uploaded-byte path."""
    path = EXAMPLE_DIR / "example_images/evoke-coral-reef.jpg"
    upload = UploadedFile(
        name=path.name, mime_type="image/jpeg", data=path.read_bytes()
    )

    validate_uploaded_image(upload)


def test_image_upload_without_prompt_uses_neutral_stability_condition() -> None:
    """Keep omitted text free from object-specific example semantics."""
    model, _, messages = _ready_model()
    path = EXAMPLE_DIR / "example_images/evoke-coral-reef.jpg"
    upload = UploadedFile(
        name=path.name, mime_type="image/jpeg", data=path.read_bytes()
    )

    reply = asyncio.run(model.set_image(upload, "", -1))

    assert reply.action == "set_image"
    assert model.state.prompt == STABILITY_PROMPT
    assert isinstance(messages[-1], StateUpdate)
    assert messages[-1].prompt == STABILITY_PROMPT
    assert messages[-1].input_source == "uploaded"


def test_empty_set_prompt_restores_stability_condition() -> None:
    """Allow a viewer to restore the documented neutral fallback condition."""
    model, _, messages = _ready_model()

    reply = asyncio.run(model.set_prompt(""))

    assert reply.detail == "Neutral stability prompt restored"
    assert model.state.prompt == STABILITY_PROMPT
    assert messages[-1].prompt == STABILITY_PROMPT


def test_pose_upload_accepts_upstream_matrix_shapes() -> None:
    """Accept finite camera-to-world matrices and a calibrated intrinsic matrix."""
    content = io.BytesIO()
    np.savez(
        content,
        cam_c2w=np.repeat(np.eye(4)[None], 2, axis=0),
        intrinsics=np.eye(3),
    )
    upload = UploadedFile(
        name="pose.npz",
        mime_type="application/x-npz",
        data=content.getvalue(),
    )

    validate_uploaded_pose(upload)


def test_step_snapshot_preserves_conditioning_and_errors() -> None:
    """Use the sampled prompt throughout generation and propagate failures."""
    model, backend, _ = _ready_model()

    async def run() -> None:
        snapshot = await model.process_input()
        model.state.prompt = "A different world"
        await model.process_output(StepOutcome(result=model.generate(snapshot)))
        assert backend.generate_calls[0][2] == "A coral reef"
        assert model._chunk_index == 1
        with pytest.raises(RuntimeError, match="failed"):
            await model.process_output(StepOutcome(error=RuntimeError("failed")))
        assert model._chunk_index == 1

    asyncio.run(run())


@pytest.mark.parametrize("max_chunks", [512, 2048])
def test_automatic_restart_effects_belong_to_output(
    monkeypatch: Any, max_chunks: int
) -> None:
    """Snapshot a neutral fresh rollout without sending or flushing in input."""
    model, backend, messages = _ready_model()
    model._config = SimpleNamespace(max_chunks=max_chunks)
    model._engine._max_chunks = max_chunks
    model._chunk_index = max_chunks - 1
    model.state._applied_world_id = model.state._world_id
    model._engine._world_id = model.state._world_id
    model._engine._chunk_index = max_chunks - 1
    model._engine._seed = 42
    model.state.forward = 1.0
    flushes = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    async def run() -> None:
        last = await model.process_input()
        await model.process_output(StepOutcome(result=model.generate(last)))
        assert model._chunk_index == max_chunks
        messages.clear()
        snapshot = await model.process_input()
        assert not messages and not flushes
        assert snapshot.anchor is not None
        np.testing.assert_allclose(
            snapshot.trajectory, np.repeat(np.eye(4)[None], 36, axis=0)
        )
        await model.process_output(StepOutcome(result=model.generate(snapshot)))
        assert model._chunk_index == 1
        assert len(backend.reset_calls) == 1
        assert flushes == [None]
        assert model.state.forward == 0.0

    asyncio.run(run())


def test_default_restart_bound_matches_config_fallback(tmp_path: Path) -> None:
    """Keep the shipped restart horizon and an omitted setting consistent."""
    config_path = EXAMPLE_DIR / "evoke.yaml"
    assert evoke_config.read_config(config_path).max_chunks == 2048
    document = yaml.safe_load(config_path.read_text())
    document["stream"].pop("max_chunks")
    fallback = tmp_path / "evoke.yaml"
    fallback.write_text(yaml.safe_dump(document))
    assert evoke_config.read_config(fallback).max_chunks == 2048


@pytest.mark.parametrize("completed", [511, 512, 1024, 2047])
def test_extended_rollout_does_not_restart_early(completed: int) -> None:
    """Keep the same world and held controls until the configured boundary."""
    model, backend, messages = _ready_model()
    model._config = SimpleNamespace(max_chunks=2048)
    model._chunk_index = completed
    model.state._applied_world_id = model.state._world_id
    model._engine._world_id = model.state._world_id
    model._engine._chunk_index = completed
    model._engine._max_chunks = 2048
    model.state.pitch = 0.25

    async def run() -> None:
        snapshot = await model.process_input()
        assert snapshot.anchor is None
        assert not np.allclose(snapshot.trajectory[-1], np.eye(4))
        await model.process_output(StepOutcome(result=model.generate(snapshot)))
        assert model._chunk_index == completed + 1
        assert not backend.reset_calls
        assert model.state.pitch == 0.25
        assert not any(
            type(message).__name__ == "RolloutRestarted" for message in messages
        )

    asyncio.run(run())


class _Engine:
    def __init__(self):
        self.inputs = []
        self.world = None
        self.index = 0
        self.seed = 0
        self.mode = ""
        self.resets = 0
        self.error = None

    def generate(self, input):
        self.inputs.append(input)
        if self.error:
            raise self.error
        if input.world_id != self.world:
            self.world = input.world_id
            self.index = 0
            self.seed = input.anchor.seed
            self.mode = input.anchor.mode
        self.index += 1
        count = 33 if self.mode == "t2v" and self.index == 1 else 36
        return EvokeResult(
            np.zeros((count, 8, 8, 3), np.uint8),
            self.world,
            self.index,
            self.seed,
            False,
        )

    def reset(self):
        self.resets += 1


def _fake_app():
    app, _, messages = _ready_model()
    engine = _Engine()
    app._engine = engine
    return app, engine, messages


def _upload():
    path = EXAMPLE_DIR / "example_images/evoke-coral-reef.jpg"
    return UploadedFile(name=path.name, mime_type="image/jpeg", data=path.read_bytes())


def test_generate_reads_only_frozen_input():
    app, engine, _ = _fake_app()
    input = asyncio.run(app.process_input())
    with pytest.raises(FrozenInstanceError):
        input.prompt = "Changed"
    app.state = app._config = app._planner = app._media = None
    result = app.generate(input)
    assert engine.inputs == [input]
    assert result.chunk_index == 1


def test_refusal_and_failure_never_ack_or_complete():
    app, engine, messages = _fake_app()

    async def run():
        app._media = None
        with pytest.raises(ApplicationError):
            await app.process_input()
        assert not engine.inputs
        await app.set_image(_upload(), "Reef", 42)
        input = await app.process_input()
        messages.clear()
        engine.error = RuntimeError("worker failed")
        try:
            app.generate(input)
        except RuntimeError as error:
            with pytest.raises(RuntimeError, match="worker failed"):
                await app.process_output(StepOutcome(error=error))
        assert app._chunk_index == 0 and not messages
        assert app.state._applied_world_id is None
        assert (await app.process_input()).anchor is not None

    asyncio.run(run())


def test_ten_actions_send_conditioning_once_and_preserve_camera_arrays():
    app, engine, messages = _fake_app()

    async def run():
        await app.set_image(_upload(), "Reef", 42)
        for index in range(10):
            await app.set_forward((index % 3 - 1) / 2)
            await app.set_yaw((index % 5 - 2) / 4)
            input = await app.process_input()
            assert (input.anchor is not None) == (index == 0)
            assert input.trajectory.shape == (36, 4, 4)
            messages.clear()
            output = await app.process_output(StepOutcome(result=app.generate(input)))
            assert output.main_video.shape == (36, 8, 8, 3)
            assert app._chunk_index == index + 1
            assert isinstance(messages[-1], StateUpdate)
        assert len({input.world_id for input in engine.inputs}) == 1

    asyncio.run(run())


def test_prompt_reset_image_and_session_cleanup():
    app, engine, _ = _fake_app()

    async def step():
        input = await app.process_input()
        await app.process_output(StepOutcome(result=app.generate(input)))
        return input

    async def run():
        await app.set_image(_upload(), "First", 42)
        first = await step()
        assert (
            isinstance(first.anchor.media, bytes)
            and first.anchor.media_suffix == ".jpg"
        )
        await app.set_prompt("Second")
        second = await step()
        assert second.world_id == first.world_id and second.anchor is None
        assert second.prompt == "Second"
        await app.reset(123)
        reset = await step()
        assert reset.world_id != first.world_id and reset.anchor.seed == 123
        np.testing.assert_allclose(reset.trajectory[0], np.eye(4))
        await app.set_image(_upload(), "Third", -1)
        replaced = await step()
        assert replaced.world_id != reset.world_id
        app.on_session_ended()
        assert engine.resets == 1 and app._media is None and app._pose is None
        assert app.state._applied_world_id is None

    asyncio.run(run())


def test_text_and_reference_video_have_plain_conditioning():
    app, _engine, _ = _fake_app()
    content = io.BytesIO()
    np.savez(
        content, cam_c2w=np.repeat(np.eye(4)[None], 150, axis=0), intrinsics=np.eye(3)
    )
    pose = UploadedFile(
        name="pose.npz", mime_type="application/x-npz", data=content.getvalue()
    )
    video = UploadedFile(
        name="clip.mp4", mime_type="video/mp4", data=b"fake video for input contract"
    )

    async def run():
        await app.start_text("Ocean", 7)
        input = await app.process_input()
        assert input.trajectory is None and input.anchor.media is None
        output = await app.process_output(StepOutcome(result=app.generate(input)))
        assert output.main_video.shape[0] == 33
        input = await app.process_input()
        assert input.anchor is None
        assert (
            await app.process_output(StepOutcome(result=app.generate(input)))
        ).main_video.shape[0] == 36
        await app.set_reference_video(video, pose, "Waves", 30, 720, 1280, 9)
        input = await app.process_input()
        assert input.anchor.media == video.data and input.anchor.pose == pose.data
        assert (
            input.anchor.media_suffix == ".mp4" and input.anchor.pose_suffix == ".npz"
        )
        assert input.anchor.source_fps == 30 and input.anchor.seed == 9
        assert input.trajectory.shape == (36, 4, 4)

    asyncio.run(run())


def test_model_owns_world_count_seed_and_cap():
    model = EvokeModel()
    backend = _Backend()
    model._backend = backend
    model._max_chunks = 2
    poses = np.repeat(np.eye(4)[None], 36, axis=0)
    anchor = EvokeAnchor("i2v", b"encoded", ".png", None, "", 123, 30, 720, 1280)
    with pytest.raises(NoAnchor):
        model.generate(EvokeInput(1, None, "World", poses))
    first = model.generate(EvokeInput(1, anchor, "World", poses))
    second = model.generate(EvokeInput(1, None, "Changed", poses))
    assert first.chunk_index == 1 and second.complete and second.seed == 123
    assert len(backend.reset_calls) == 1 and backend.generate_calls[-1][2] == "Changed"
    assert backend.reset_calls[0]["media"] == b"encoded"
    with pytest.raises(RolloutExhausted):
        model.generate(EvokeInput(1, None, "World", poses))
    assert len(backend.reset_calls) == 1
    fresh = model.generate(EvokeInput(2, anchor, "World", poses))
    assert fresh.chunk_index == 1
    model.reset()
    assert backend.ended == 1
    with pytest.raises(NoAnchor):
        model.generate(EvokeInput(2, None, "World", poses))


def test_backend_materializes_plain_bytes_and_cleans_uploads(tmp_path):
    from upstream_backend import EvokeWorkerBackend

    backend = EvokeWorkerBackend.__new__(EvokeWorkerBackend)
    backend._root = tmp_path
    backend._session_uploads = []
    backend._request_id = 0
    path = backend._materialize(b"encoded video", "media", ".mp4")
    assert path.suffix == ".mp4" and path.read_bytes() == b"encoded video"
    assert backend._materialize(path, "media", "") == path
    assert backend._materialize(None, "pose", "") is None
    backend._clear_session_uploads()
    assert not path.exists()


def test_model_and_config_import_without_runtime():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "reactor_runtime" or name.startswith("reactor_runtime."):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import evoke_model
import evoke_config
"""
    subprocess.run([sys.executable, "-c", code], cwd=EXAMPLE_DIR, check=True)
