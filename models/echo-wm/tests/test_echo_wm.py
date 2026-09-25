"""Test Echo-WM's schema, camera fidelity, uploads, and chunk boundary."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from echo_wm import EchoWM
from echo_wm_assets import activate_source, read_config
from echo_wm_attention import FlashAttention4, set_attention_backend
from echo_wm_camera import EchoCameraPlanner, MotionConfig
from echo_wm_images import validate_uploaded_image
from echo_wm_model import (
    EchoAnchor,
    EchoInput,
    EchoModel,
    EchoResult,
    NoAnchor,
    RolloutExhausted,
)
from echo_wm_schema import EchoWMOutput, EchoWMState
from PIL import Image
from reactor_runtime import (
    ApplicationError,
    CommandError,
    StepOutcome,
    UploadedFile,
    get_weights_path,
)
from reactor_runtime.interface.model.contract import ModelContract


class FakeModel:
    """Record application inputs without importing the upstream GPU stack."""

    def __init__(self, max_chunks: int = 512) -> None:
        self.inputs: list[EchoInput] = []
        self.world_id: int | None = None
        self.chunk_index = 0
        self.max_chunks = max_chunks
        self.resets = 0

    def generate(self, input: EchoInput) -> EchoResult:
        self.inputs.append(input)
        if input.world_id != self.world_id:
            assert input.anchor is not None
            self.world_id = input.world_id
            self.chunk_index = 0
        self.chunk_index += 1
        first = self.chunk_index == 1
        return EchoResult(
            video=np.zeros((25 if first else 24, 8, 8, 3), dtype=np.uint8),
            audio=np.zeros((1, 50_000 if first else 48_000), dtype=np.int16),
            world_id=input.world_id,
            chunk_index=self.chunk_index,
            prompt=input.prompt,
            complete=self.chunk_index >= self.max_chunks,
            profile={"denoise_seconds": 0.1},
        )

    def reset(self) -> None:
        self.resets += 1
        self.world_id = None
        self.chunk_index = 0


def _app(max_chunks: int = 512) -> tuple[EchoWM, FakeModel, list[Any]]:
    config = replace(
        read_config(Path(__file__).parents[1] / "echo_wm.yaml"), max_chunks=max_chunks
    )
    app = EchoWM()
    app.state = EchoWMState()
    app._config = config
    engine = FakeModel(max_chunks)
    app._engine = engine
    app._planner = EchoCameraPlanner(
        MotionConfig(
            config.fps,
            config.translation_speed,
            config.rotation_speed_degrees,
            config.pitch_speed_degrees,
            config.pitch_limit_degrees,
        )
    )
    messages: list[Any] = []

    async def send(message: Any) -> None:
        messages.append(message)

    app.send = send
    app.on_session_started()
    return app, engine, messages


def _selected_app(max_chunks: int = 512) -> tuple[EchoWM, FakeModel, list[Any]]:
    app, engine, messages = _app(max_chunks)
    app._selected_image = Path("anchor.png")
    app.state.prompt = "A quiet room with distant music."
    app._request_reset()
    return app, engine, messages


async def _step(app: EchoWM, elapsed: float = 0.37) -> EchoWMOutput:
    result = app.generate(await app.process_input())
    return await app.process_output(StepOutcome(result=result, elapsed=elapsed))


def test_refusals_never_reach_model() -> None:
    app, engine, _ = _app()
    with pytest.raises(ApplicationError, match="image"):
        asyncio.run(_step(app))
    app._selected_image = Path("anchor.png")
    with pytest.raises(ApplicationError, match="prompt"):
        asyncio.run(_step(app))
    assert engine.inputs == []


def test_generate_only_forwards_frozen_input() -> None:
    app, engine, _ = _selected_app()
    snapshot = asyncio.run(app.process_input())
    with pytest.raises(FrozenInstanceError):
        snapshot.prompt = "changed"
    app.state = None
    app._planner = None
    app._config = None
    result = app.generate(snapshot)
    assert engine.inputs == [snapshot]
    assert result.world_id == snapshot.world_id
    assert snapshot.poses.shape == (3, 4, 4)
    assert snapshot.poses.dtype == np.float32


def test_ten_camera_actions_keep_one_world_and_one_anchor() -> None:
    app, engine, messages = _selected_app()

    async def run() -> None:
        for index in range(10):
            await app.set_camera_motion(
                forward=0.0, strafe=0.0, pitch=0.0, yaw=0.25 if index % 2 else -0.25
            )
            output = await _step(app)
            assert output.main_video.shape[0] == (25 if index == 0 else 24)
            assert output.main_audio.shape[-1] == (50_000 if index == 0 else 48_000)

    asyncio.run(run())
    assert [i.anchor is not None for i in engine.inputs] == [True] + [False] * 9
    assert len({i.world_id for i in engine.inputs}) == 1
    assert app._chunk_index == 10
    chunks = [m for m in messages if type(m).__name__ == "ChunkCompleted"]
    assert len(chunks) == 10
    assert all(m.generation_seconds == 0.37 for m in chunks)
    assert chunks[-1].yaw == 0.25
    assert not np.array_equal(engine.inputs[0].poses, engine.inputs[1].poses)


def test_reset_prompt_and_upload_each_create_a_world(tmp_path: Path) -> None:
    app, engine, _ = _selected_app()
    asyncio.run(_step(app))
    asyncio.run(app.reset(seed=123))
    asyncio.run(_step(app))
    assert engine.inputs[-1].anchor.seed == 123
    np.testing.assert_array_equal(
        engine.inputs[-1].poses, np.tile(np.eye(4), (3, 1, 1))
    )
    asyncio.run(app.set_prompt(prompt="A sunny garden."))
    asyncio.run(_step(app))
    assert engine.inputs[-1].prompt == "A sunny garden."
    path = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "blue").save(path)
    data = path.read_bytes()
    asyncio.run(
        app.set_image(
            UploadedFile(name=path.name, mime_type="image/png", data=data),
            prompt="A blue scene.",
            seed=5,
        )
    )
    asyncio.run(_step(app))
    assert engine.inputs[-1].anchor.image == data
    assert engine.inputs[-1].anchor.suffix == ".png"
    assert len({i.world_id for i in engine.inputs}) == 4
    assert all(i.anchor is not None for i in engine.inputs)
    assert app._chunk_index == 1


def test_automatic_restart_preserves_controls_and_emits_final_chunk() -> None:
    app, engine, messages = _selected_app(max_chunks=2)
    app.state._forward = 0.5
    first = asyncio.run(_step(app))
    second = asyncio.run(_step(app))
    assert first.main_video.shape[0] == 25
    assert second.main_video.shape[0] == 24
    assert [type(m).__name__ for m in messages[-3:]] == [
        "ChunkCompleted",
        "AutomaticResetQueued",
        "StateUpdate",
    ]
    assert app._chunk_index == 2
    assert app._state_update().reset_queued
    assert app._state_update().next_chunk == 1
    assert app.state._forward == 0.5
    third = asyncio.run(_step(app))
    assert third.main_video.shape[0] == 25
    assert engine.inputs[-1].world_id != engine.inputs[0].world_id
    assert engine.inputs[-1].anchor is not None


def test_model_failure_does_not_acknowledge_world_or_count_chunk() -> None:
    app, engine, messages = _selected_app()
    snapshot = asyncio.run(app.process_input())

    def fail(input: EchoInput) -> EchoResult:
        raise RuntimeError("GPU failure")

    engine.generate = fail
    with pytest.raises(RuntimeError, match="GPU failure") as error:
        app.generate(snapshot)
    with pytest.raises(RuntimeError, match="GPU failure"):
        asyncio.run(app.process_output(StepOutcome(error=error.value, elapsed=0.5)))
    assert app._chunk_index == 0
    assert app.state._applied_world_id is None
    assert messages == []
    assert not app._generating
    assert asyncio.run(app.process_input()).anchor is not None


def test_session_end_releases_model_and_selection() -> None:
    app, engine, _ = _selected_app()
    asyncio.run(_step(app))
    app.on_session_ended()
    assert engine.resets == 1
    assert app._selected_image is None
    assert app._chunk_index == 0
    assert app.state._applied_world_id is None


class RecordingBackend:
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.cameras: list[Any] = []
        self.ends = 0
        self.last_profile = {"denoise_seconds": 0.2}

    def reset(self, **kwargs: Any) -> None:
        self.starts.append(kwargs)
        self.image_bytes = kwargs["image"].read_bytes()

    def generate_chunk(
        self, camera: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        self.cameras.append((camera, kwargs))
        return np.zeros((24, 8, 8, 3), dtype=np.uint8), np.zeros(
            (1, 48_000), dtype=np.int16
        )

    def end_session(self, **kwargs: Any) -> None:
        self.ends += 1


def test_model_owns_world_counter_and_reset(tmp_path: Path) -> None:
    engine = EchoModel()
    backend = RecordingBackend()
    engine._backend = backend
    engine._config = replace(
        read_config(Path(__file__).parents[1] / "echo_wm.yaml"),
        runtime_dir=tmp_path,
        max_chunks=2,
    )
    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    anchor = EchoAnchor(image=b"encoded-image", suffix=".png", seed=7)
    input = EchoInput(
        world_id=1, anchor=anchor, prompt="room", poses=poses, fov_degrees=70.0
    )
    with pytest.raises(NoAnchor):
        engine.generate(replace(input, anchor=None))
    assert backend.starts == []
    first = engine.generate(input)
    second = engine.generate(
        replace(input, anchor=None, prompt="not applied mid-world", fov_degrees=80.0)
    )
    assert (first.chunk_index, second.chunk_index) == (1, 2)
    assert first.world_id == second.world_id == 1
    assert second.prompt == "room" and second.complete
    assert len(backend.starts) == 1
    assert backend.image_bytes == b"encoded-image"
    assert not backend.starts[0]["image"].exists()
    assert backend.cameras[-1][1]["fov_degrees"] == 80.0
    assert backend.cameras[-1][0].latent_poses is poses
    with pytest.raises(RolloutExhausted):
        engine.generate(replace(input, anchor=None))
    assert len(backend.cameras) == 2
    fresh = engine.generate(replace(input, world_id=2))
    assert fresh.chunk_index == 1 and len(backend.starts) == 2
    engine.reset()
    assert backend.ends == 1
    with pytest.raises(NoAnchor):
        engine.generate(replace(input, world_id=2, anchor=None))


def test_model_imports_without_runtime() -> None:
    code = """
import builtins
original = builtins.__import__
def isolated(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError('model imports runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = isolated
from echo_wm_model import EchoModel
EchoModel()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )


def test_public_contract_is_atomic_and_audiovisual() -> None:
    """Expose only explicit controls and both generated media tracks."""
    contract = ModelContract.of(EchoWM)
    assert set(contract.commands) == {
        "random_image",
        "release_camera",
        "reset",
        "set_camera_motion",
        "set_fov",
        "set_image",
        "set_prompt",
    }
    assert EchoWM.fps == 24
    assert EchoWM.buffer_size == 24
    assert set(EchoWMOutput.__tracks__) == {"main_video", "main_audio"}
    assert EchoWMOutput.__tracks__["main_audio"].rate == 48_000


def test_camera_chunks_match_upstream_integration() -> None:
    """Match the released DSL across controls that change between chunks."""
    config_path = Path(__file__).parents[1] / "echo_wm.yaml"
    config = read_config(config_path, get_weights_path())
    if not (config.wm_root / "helpers" / "action_camera.py").is_file():
        pytest.skip("Pinned upstream camera reference is not installed")
    activate_source(config)
    from helpers.action_camera import action_string_to_c2w

    config = MotionConfig(
        fps=24.0,
        translation_speed=0.05,
        rotation_speed_degrees=0.4,
        pitch_speed_degrees=0.2,
        pitch_limit_degrees=40.0,
    )
    planner = EchoCameraPlanner(config)
    first = planner.plan_chunk(
        forward=1.0,
        strafe=0.0,
        pitch=0.0,
        yaw=-1.0,
        frame_count=24,
    )
    second = planner.plan_chunk(
        forward=0.0,
        strafe=-1.0,
        pitch=1.0,
        yaw=0.0,
        frame_count=24,
    )
    reference = action_string_to_c2w(
        [["w", "j"]] * 24 + [["a", "i"]] * 24,
        translation_speed=config.translation_speed,
        rotation_speed_deg=config.rotation_speed_degrees,
        pitch_speed_deg=config.pitch_speed_degrees,
        pitch_limit_deg=config.pitch_limit_degrees,
        fps=config.fps,
    )
    expected = reference[[8, 16, 24, 32, 40, 48]]
    actual = np.concatenate([first.latent_poses, second.latent_poses])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_upload_validation_checks_bytes_and_codec(tmp_path: Path) -> None:
    """Accept a real image and reject bytes disguised as an image."""
    image_path = tmp_path / "anchor.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(image_path)
    validate_uploaded_image(
        UploadedFile(
            name=image_path.name,
            mime_type="image/png",
            data=image_path.read_bytes(),
        )
    )
    with pytest.raises(CommandError):
        validate_uploaded_image(
            UploadedFile(
                name="fake.png",
                mime_type="image/png",
                data=b"not an image",
            )
        )


def test_blank_upload_prompt_uses_configured_default(tmp_path: Path) -> None:
    """Use the configured image-neutral default for a blank upload prompt."""
    image_path = tmp_path / "anchor.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(image_path)
    config = read_config(Path(__file__).parents[1] / "echo_wm.yaml")
    model = EchoWM()
    model.state = EchoWMState()
    model.state.prompt = "Prompt paired with the previously selected image."
    model._config = config
    model._seed = config.seed

    message = asyncio.run(
        model.set_image(
            UploadedFile(
                name=image_path.name,
                mime_type="image/png",
                data=image_path.read_bytes(),
            ),
            prompt="   ",
            seed=-1,
        )
    )

    assert model.state.prompt == config.default_upload_prompt
    assert message.prompt == config.default_upload_prompt
    assert "previously selected" not in message.prompt


def test_inference_yields_one_synchronized_native_chunk(tmp_path: Path) -> None:
    """Bind one backend call to one batched audio-video Reactor output."""

    class FakeBackend:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.chunk_calls = 0

        def reset(self, **_: object) -> None:
            self.reset_calls += 1

        def generate_chunk(
            self, *_: object, **__: object
        ) -> tuple[np.ndarray, np.ndarray]:
            self.chunk_calls += 1
            return (
                np.zeros((25, 704, 1280, 3), dtype=np.uint8),
                np.zeros((1, 50_000), dtype=np.int16),
            )

        @property
        def last_profile(self) -> dict[str, float]:
            return {
                "denoise_seconds": 0.1,
                "cache_commit_seconds": 0.02,
                "video_decode_seconds": 0.03,
                "audio_decode_seconds": 0.04,
                "cuda_total_seconds": 0.19,
            }

        def end_session(self, *, release_cuda_cache: bool = True) -> None:
            return

    config_path = Path(__file__).parents[1] / "echo_wm.yaml"
    config = read_config(config_path)
    model = EchoWM()
    model.state = EchoWMState()
    model._config = config
    backend = FakeBackend()
    model._engine = EchoModel()
    model._engine._backend = backend
    model._engine._config = config
    model._planner = EchoCameraPlanner(
        MotionConfig(
            fps=config.fps,
            translation_speed=config.translation_speed,
            rotation_speed_degrees=config.rotation_speed_degrees,
            pitch_speed_degrees=config.pitch_speed_degrees,
            pitch_limit_degrees=config.pitch_limit_degrees,
        )
    )
    model._selected_image = tmp_path / "anchor.png"
    model.state.prompt = "A quiet room with distant music."
    model.state._world_id = 1
    model._seed = 42

    async def generate() -> tuple[EchoWMOutput, EchoWMOutput]:
        from reactor_runtime import StepOutcome

        first = await model.process_output(
            StepOutcome(result=model.generate(await model.process_input()))
        )
        for _ in range(9):
            second = await model.process_output(
                StepOutcome(result=model.generate(await model.process_input()))
            )
        assert isinstance(first, EchoWMOutput)
        assert isinstance(second, EchoWMOutput)
        return first, second

    output, _ = asyncio.run(generate())
    video = cast(np.ndarray, output.main_video)
    audio = cast(np.ndarray, output.main_audio)
    assert video.shape == (25, 704, 1280, 3)
    assert audio.shape == (1, 50_000)
    assert backend.reset_calls == 1
    assert backend.chunk_calls == 10
    assert model._chunk_index == 10


def test_rollout_reset_flushes_pending_media() -> None:
    """Discard queued media when a fresh rollout replaces the active world."""

    class FakeOutput:
        def __init__(self) -> None:
            self.flushes = 0

        def flush(self) -> None:
            self.flushes += 1

    model = EchoWM()
    model.state = EchoWMState()
    output = FakeOutput()
    model.output = cast(Any, output)

    model._request_reset()

    assert model.state._world_id != model.state._applied_world_id
    assert output.flushes == 1


def test_flash_attention_4_preserves_masked_upstream_path() -> None:
    """Use FA4 only where its unmasked call represents upstream semantics."""
    torch = pytest.importorskip("torch")
    calls: list[str] = []

    def flash(query: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        calls.append("flash")
        return query

    def fallback(*_: object) -> torch.Tensor:
        calls.append("pytorch")
        return torch.ones((1, 2, 8), dtype=torch.bfloat16)

    attention = FlashAttention4(flash, fallback, torch)
    query = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 8)
    key = torch.zeros((1, 3, 8), dtype=torch.bfloat16)
    value = torch.zeros((1, 3, 8), dtype=torch.bfloat16)
    unmasked = attention(query, key, value, heads=2)
    masked = attention(query, key, value, heads=2, mask=torch.ones((2, 3)))

    torch.testing.assert_close(unmasked, query)
    torch.testing.assert_close(masked, torch.ones_like(query))
    assert calls == ["flash", "pytorch"]


def test_attention_backend_changes_only_upstream_attention_modules() -> None:
    """Patch each selected attention module and leave unrelated modules intact."""
    torch = pytest.importorskip("torch")

    class Selected(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_function: object = "before"

    root = torch.nn.Sequential(Selected(), torch.nn.Linear(2, 2), Selected())
    callable_backend = object()
    changed = set_attention_backend(root, callable_backend, Selected)

    assert changed == 2
    assert cast(Selected, root[0]).attention_function is callable_backend
    assert cast(Selected, root[2]).attention_function is callable_backend


def test_video_decode_tiling_is_disabled_for_b200() -> None:
    """Decode each visible chunk directly when the requested B200 has ample memory."""
    config = read_config(Path(__file__).parents[1] / "echo_wm.yaml")

    assert config.video_decode_tiling is False
