"""Contract and rollout-boundary tests for the YUME-1.5 adapter."""

from __future__ import annotations

import asyncio
import io
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from yume import Yume15
from yume_assets import read_config
from yume_controls import conditioned_prompt
from yume_images import prepare_image
from yume_model import NoAnchor, YumeInput, YumeModel
from yume_types import YumeOutput, YumeState


@pytest.mark.parametrize(
    "damaged",
    [
        "diffusion_pytorch_model.safetensors",
        "config.json",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/special_tokens_map.json",
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_asset_download_repairs_each_required_file(
    tmp_path, monkeypatch, damaged, empty
):
    import sys
    from dataclasses import replace

    import yume_assets as assets

    config = assets.read_config(Path(__file__).parents[1] / "yume.yaml", tmp_path)
    config = replace(config, source_path=tmp_path / "source")
    (config.source_path / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=config.source_revision),
    )
    root = config.checkpoint_path
    files = [
        "diffusion_pytorch_model.safetensors",
        "config.json",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/special_tokens_map.json",
    ]

    def populate(**kwargs):
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    populate()
    target = root / damaged
    if empty:
        target.write_bytes(b"")
    else:
        target.unlink()

    def download_snapshot(**kwargs):
        # Simulate Hub reusing stale metadata for an existing zero-byte file.
        populate()
        if empty and not kwargs.get("force_download"):
            target.write_bytes(b"")

    download = Mock(side_effect=download_snapshot)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )

    assets.prepare_assets(config)
    assert download.call_count == (2 if empty else 1)
    if empty:
        assert download.call_args.kwargs["force_download"] is True
        assert download.call_args.kwargs["allow_patterns"] == [damaged]
    assert download.call_args.kwargs["revision"] == config.checkpoint_revision

    download.reset_mock()
    assets.prepare_assets(config)
    download.assert_not_called()
    target.unlink()
    download.side_effect = None
    with pytest.raises(RuntimeError, match="missing or empty"):
        assets.prepare_assets(config)


def uploaded_image() -> UploadedFile:
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 30, 40)).save(stream, format="PNG")
    return UploadedFile(name="start.png", mime_type="image/png", data=stream.getvalue())


def test_contract_has_only_real_upstream_controls() -> None:
    contract = ModelContract.of(Yume15)
    assert set(contract.commands) == {
        "release_controls",
        "reset",
        "set_key_state",
        "set_image",
        "set_prompt",
        "set_text_scene",
        "set_video_scene",
    }
    assert "fps" not in Yume15.__dict__
    assert Yume15.buffer_size == 29
    assert set(YumeOutput.__tracks__) == {"main_video"}


def test_native_fast_context_is_fixed() -> None:
    config = read_config(Path(__file__).parents[1] / "yume.yaml")
    assert config.frames_per_chunk == 32
    assert config.latent_frames_per_chunk == 8
    assert config.sample_steps == 4


def test_conditioning_explicitly_distinguishes_stationary_and_motion() -> None:
    stationary = conditioned_prompt("A city street", "none", "none")
    moving = conditioned_prompt("A city street", "forward", "pan_left")

    assert "Actual distance moved:0" in stationary
    assert "Angular change rate (turn speed):0" in stationary
    assert "View rotation speed:0" in stationary
    assert "pushes forward" not in stationary
    assert "Actual distance moved:4" in moving
    assert "Angular change rate (turn speed):4" in moving
    assert "View rotation speed:4" in moving


def test_upload_is_decodable() -> None:
    prepare_image(uploaded_image())


def test_blank_image_prompt_uses_neutral_configured_prompt() -> None:
    model = Yume15()
    model._engine = YumeModel()
    model.state = YumeState()
    model._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    model.output = cast(Any, type("Output", (), {"flush": lambda self: None})())

    message = asyncio.run(model.set_image(uploaded_image(), "   ", 42))

    assert model.state.prompt == model._config.default_upload_prompt
    assert message.prompt == model._config.default_upload_prompt


def test_one_turn_is_one_chunk_and_prompt_can_change_without_reset(
    tmp_path: Path,
) -> None:
    class FakeBackend:
        resets = 0
        calls = 0

        def reset(self, **_: object) -> None:
            self.resets += 1

        def generate_chunk(
            self, *, prompt: str, movement: str, view: str
        ) -> tuple[np.ndarray, str]:
            self.calls += 1
            return np.zeros((29, 8, 8, 3), dtype=np.uint8)

        def end_session(self) -> None:
            return

    model = Yume15()
    model._engine = YumeModel()
    model.state = YumeState()
    model._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    backend = FakeBackend()
    model._engine._backend = backend
    model.state._seed = 42
    asyncio.run(model.set_image(uploaded_image(), "A forest trail", 42))

    async def generate() -> tuple[np.ndarray, np.ndarray]:
        first = await model.process_output(
            StepOutcome(
                result=model.generate(await model.process_input()), elapsed=1.25
            )
        )
        assert isinstance(first, YumeOutput)
        await model.set_prompt("Rain begins")
        second = await model.process_output(
            StepOutcome(
                result=model.generate(await model.process_input()), elapsed=1.25
            )
        )
        assert isinstance(second, YumeOutput)
        return cast(np.ndarray, first.main_video), cast(np.ndarray, second.main_video)

    first, second = asyncio.run(generate())
    assert first.shape == second.shape == (29, 8, 8, 3)
    assert backend.resets == 1
    assert backend.calls == 2


def test_reset_flushes_media() -> None:
    class FakeOutput:
        flushes = 0

        def flush(self) -> None:
            self.flushes += 1

    model = Yume15()
    model._engine = YumeModel()
    model.state = YumeState()
    output = FakeOutput()
    model.output = cast(Any, output)
    model._request_reset()
    assert output.flushes == 1
    assert model.state._world_id != model.state._applied_world_id
    assert model.state._pressed_keys == frozenset()


def test_held_keys_combine_and_release_independently() -> None:
    model = Yume15()
    model._engine = YumeModel()
    model.state = YumeState()
    model.state._mode = "text_to_video"
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.set_key_state("a", True))
    asyncio.run(model.set_key_state("arrow_up", True))
    assert model._resolve_controls(model.state._pressed_keys) == (
        "forward_left",
        "tilt_up",
    )
    asyncio.run(model.set_key_state("a", False))
    assert model._resolve_controls(model.state._pressed_keys) == ("forward", "tilt_up")


def test_refusal_never_calls_model() -> None:
    app = Yume15()
    app._engine = YumeModel()
    app.state = YumeState()
    app._engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app._engine.generate.assert_not_called()


def test_generate_forwards_frozen_input_without_reading_state() -> None:
    app = Yume15()
    app._engine = YumeModel()
    value = YumeInput(1, None, "forest", "none", "none")
    with pytest.raises(FrozenInstanceError):
        value.prompt = "changed"
    result = object()
    app._engine = SimpleNamespace(
        generate=lambda actual: result if actual is value else None
    )
    app.state = None
    assert app.generate(value) is result


def test_ten_continuous_hook_steps_send_anchor_once() -> None:
    app = Yume15()
    app._engine = YumeModel()
    app.state = YumeState()
    app.send = AsyncMock()
    anchors = []

    class Backend:
        def reset(self, **kwargs):
            anchors.append(kwargs)

        def generate_chunk(self, **kwargs):
            return np.zeros((29, 8, 8, 3), np.uint8)

        def end_session(self):
            pass

    app._engine._backend = Backend()

    async def drive():
        await app.set_text_scene("forest", 42)
        for index in range(1, 11):
            value = await app.process_input()
            assert (value.anchor is not None) == (index == 1)
            result = app.generate(value)
            assert result.chunk_index == index
            await app.process_output(StepOutcome(result=result, elapsed=0.5))
        assert len(anchors) == 1
        states = [
            c.args[0]
            for c in app.send.call_args_list
            if type(c.args[0]).__name__ == "StateUpdate"
        ]
        assert all(not state.limit_reached for state in states[:-1])
        await app.reset(43)
        assert (await app.process_input()).anchor is not None

    asyncio.run(drive())


def test_model_error_reaches_output_without_counting_step() -> None:
    app = Yume15()
    app._engine = YumeModel()
    app.state = YumeState()
    app._engine._backend = Mock()
    with pytest.raises(NoAnchor) as caught:
        app.generate(YumeInput(1, None, "forest", "none", "none"))
    with pytest.raises(NoAnchor):
        asyncio.run(app.process_output(StepOutcome(error=caught.value)))
    assert app.state._chunk_index == app._engine._chunk_index == 0


def test_bad_video_is_command_error():
    from reactor_runtime import CommandError

    from yume_images import prepare_video

    with pytest.raises(CommandError, match="decoded"):
        prepare_video(UploadedFile("bad.mp4", "video/mp4", b"invalid"))


def test_new_session_restores_config_seed():
    app = Yume15()
    app._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    app.state = YumeState()
    app.state._seed = 123
    app.on_session_started()
    assert app.state._seed == 42


def test_application_fake_model_final_chunk_and_refusal():
    from yume_model import YumeResult

    app = Yume15()
    app.state = YumeState()
    app._engine = Mock()
    app.send = AsyncMock()
    asyncio.run(app.set_text_scene("forest", 42))
    step = asyncio.run(app.process_input())
    result = YumeResult(step.world_id, 10, np.zeros((29, 2, 2, 3), np.uint8), True)
    app._engine.generate.return_value = result
    assert app.generate(step) is result
    app.send.reset_mock()
    asyncio.run(app.process_output(StepOutcome(result=result, elapsed=1.0)))
    assert [type(c.args[0]).__name__ for c in app.send.call_args_list] == [
        "ChunkCompleted",
        "RolloutLimitReached",
        "StateUpdate",
    ]
    with pytest.raises(ApplicationError, match="complete"):
        asyncio.run(app.process_input())
    asyncio.run(app.reset(-1))
    assert asyncio.run(app.process_input()).anchor is not None


def test_model_cap_and_reset_are_explicit():
    from yume_model import RolloutExhausted, YumeAnchor

    engine = YumeModel()
    backend = Mock()
    backend.generate_chunk.return_value = np.zeros((29, 2, 2, 3), np.uint8)
    engine._backend = backend
    engine._max_chunks = 1
    step = YumeInput(1, YumeAnchor("text_to_video", None, 42), "forest", "none", "none")
    assert engine.generate(step).complete
    with pytest.raises(RolloutExhausted):
        engine.generate(YumeInput(1, None, "forest", "none", "none"))
    assert backend.generate_chunk.call_count == 1
    engine.reset()
    backend.end_session.assert_called_once()
    assert engine.generate(step).chunk_index == 1


def test_image_anchor_is_prepared_once():
    app = Yume15()
    app.state = YumeState()
    app._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    asyncio.run(app.set_image(uploaded_image(), "forest", 42))
    anchor = asyncio.run(app.process_input()).anchor
    assert anchor.media.shape == (704, 1280, 3)
    assert anchor.media.dtype == np.uint8
    assert anchor.media is asyncio.run(app.process_input()).anchor.media


def test_video_preparation_is_bounded(monkeypatch):
    import av

    from yume_images import prepare_video

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, video):
            for i in range(33):
                yield SimpleNamespace(
                    width=16, height=16, to_image=lambda: Image.new("RGB", (16, 16))
                )
            raise AssertionError("decoded more than the required 33 frames")

    monkeypatch.setattr(av, "open", lambda _: Container())
    result = prepare_video(UploadedFile("video.mp4", "video/mp4", b"fixture"))
    assert result.shape == (33, 704, 1280, 3)


def test_model_imports_without_runtime():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith('reactor_runtime'): raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import yume_model, yume_controls
yume_model.YumeModel()
""",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_first_checkout_pins_revision_before_loading_weights(tmp_path, monkeypatch):
    from dataclasses import replace

    import yume_assets

    config = replace(
        read_config(Path(__file__).parents[1] / "yume.yaml"),
        source_path=tmp_path / "source",
        checkpoint_path=tmp_path / "weights",
    )
    config.checkpoint_path.mkdir()
    for name in [
        "diffusion_pytorch_model.safetensors",
        "config.json",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/special_tokens_map.json",
    ]:
        path = config.checkpoint_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if "clone" in args:
            (config.source_path / ".git").mkdir(parents=True)
        return SimpleNamespace(stdout=config.source_revision)

    monkeypatch.setattr(yume_assets.subprocess, "run", run)
    yume_assets.prepare_assets(config)
    assert calls[0][1] == "clone"
    assert calls[1][-3:] == ["checkout", "--detach", config.source_revision]
    assert calls[2][-2:] == ["rev-parse", "HEAD"]


def test_video_scene_passes_prepared_cpu_anchor(monkeypatch):
    import yume

    app = Yume15()
    app.state = YumeState()
    app._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    frames = np.zeros((33, 704, 1280, 3), np.uint8)
    decode = Mock(return_value=frames)
    monkeypatch.setattr(yume, "prepare_video", decode)
    media = UploadedFile("scene.mp4", "video/mp4", b"fixture")
    asyncio.run(app.set_video_scene(media, "forest", 42))
    step = asyncio.run(app.process_input())
    assert step.anchor.media is frames and step.anchor.mode == "video_to_video"
    decode.assert_called_once_with(media)
