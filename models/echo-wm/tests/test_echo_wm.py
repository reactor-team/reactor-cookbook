"""Test Echo-WM's schema, camera fidelity, uploads, and chunk boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image
from reactor_runtime import CommandError, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from echo_wm import EchoWM
from echo_wm_assets import activate_source, read_config
from echo_wm_attention import FlashAttention4, set_attention_backend
from echo_wm_camera import EchoCameraPlanner, MotionConfig
from echo_wm_images import validate_uploaded_image
from echo_wm_schema import EchoWMOutput, EchoWMState


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
    activate_source(read_config(config_path))
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
    model._backend = FakeBackend()
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
    model.state._reset_requested = True
    model._seed = 42

    async def generate() -> tuple[EchoWMOutput, EchoWMOutput]:
        outputs = model.inference()
        first = await anext(outputs)
        second = await anext(outputs)
        assert isinstance(first, EchoWMOutput)
        assert isinstance(second, EchoWMOutput)
        return first, second

    output, _ = asyncio.run(generate())
    video = cast(np.ndarray, output.main_video)
    audio = cast(np.ndarray, output.main_audio)
    assert video.shape == (25, 704, 1280, 3)
    assert audio.shape == (1, 50_000)
    assert model._backend.reset_calls == 1
    assert model._backend.chunk_calls == 2


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

    assert model.state._reset_requested is True
    assert output.flushes == 1


def test_flash_attention_4_preserves_masked_upstream_path() -> None:
    """Use FA4 only where its unmasked call represents upstream semantics."""
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
