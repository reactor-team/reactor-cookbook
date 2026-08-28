"""Test the EVOKE Reactor contract and chunk boundary without loading weights."""

from __future__ import annotations

import asyncio
import io
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from reactor_runtime import UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

import evoke_config
from evoke import Evoke
from evoke_camera import CameraMotionPlanner, MotionConfig
from evoke_images import validate_uploaded_image, validate_uploaded_pose
from evoke_types import CommandApplied, EvokeOutput, EvokeState, StateUpdate

EXAMPLE_DIR = Path(__file__).parents[1]
STABILITY_PROMPT = evoke_config.read_config(EXAMPLE_DIR / "evoke.yaml").stability_prompt

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

    def reset(self, **values: Any) -> None:
        self.reset_calls.append(values)

    def generate_chunk(
        self,
        trajectory: np.ndarray | None,
        *,
        seed: int,
        prompt: str,
    ) -> np.ndarray:
        self.generate_calls.append((trajectory, seed, prompt))
        return np.zeros((36, 384, 640, 3), dtype=np.uint8)


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
    model._backend = backend
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
    assert document["build"]["runtime_version"] == "3.2.5"
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
        generator = model.inference()
        output = await anext(generator)
        await generator.aclose()
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
