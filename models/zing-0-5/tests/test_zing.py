"""Contract and upstream-fidelity tests for the Zing adapter."""

from __future__ import annotations

import asyncio
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from zing import Zing
from zing_assets import read_config
from zing_images import prepare_image
from zing_model import LOCAL_ATTN_SIZE, SINK_SIZE, ZingModel
from zing_types import ZingOutput, ZingState


@pytest.mark.parametrize("value", ["~/source", "relative/source", "/absolute/source"])
def test_config_paths_expand_home_and_ignore_cwd(tmp_path, monkeypatch, value):
    import yaml

    import zing_assets as assets

    path = Path(__file__).parents[1] / "zing.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["source"]["path"] = value
    raw["assets"]["path"] = value
    monkeypatch.setattr(assets.yaml, "safe_load", lambda text: raw)
    monkeypatch.chdir(tmp_path)
    config = assets.read_config(path, tmp_path)
    expected = Path(value).expanduser()
    assert config.source_path == (path.parent / expected).resolve()
    assert config.asset_path == (tmp_path / expected).resolve()


@pytest.mark.parametrize(
    "damaged",
    [
        "generator/model.pt",
        "pretrained/vae/config.json",
        "pretrained/vae/diffusion_pytorch_model.safetensors",
        "pretrained/text_encoder/config.json",
        "pretrained/text_encoder/model.safetensors.index.json",
        "pretrained/text_encoder/model-00001-of-00003.safetensors",
        "pretrained/text_encoder/model-00002-of-00003.safetensors",
        "pretrained/text_encoder/model-00003-of-00003.safetensors",
        "pretrained/tokenizer/tokenizer_config.json",
        "pretrained/tokenizer/tokenizer.json",
        "pretrained/tokenizer/spiece.model",
        "pretrained/tokenizer/special_tokens_map.json",
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_asset_download_repairs_each_required_file(
    tmp_path, monkeypatch, damaged, empty
):
    import sys
    from dataclasses import replace

    import zing_assets as assets

    config = assets.read_config(Path(__file__).parents[1] / "zing.yaml", tmp_path)
    config = replace(config, source_path=tmp_path / "source")
    (config.source_path / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=config.source_revision),
    )
    root = config.asset_path
    files = [
        "generator/model.pt",
        "pretrained/vae/config.json",
        "pretrained/vae/diffusion_pytorch_model.safetensors",
        "pretrained/text_encoder/config.json",
        "pretrained/text_encoder/model.safetensors.index.json",
        "pretrained/text_encoder/model-00001-of-00003.safetensors",
        "pretrained/text_encoder/model-00002-of-00003.safetensors",
        "pretrained/text_encoder/model-00003-of-00003.safetensors",
        "pretrained/tokenizer/tokenizer_config.json",
        "pretrained/tokenizer/tokenizer.json",
        "pretrained/tokenizer/spiece.model",
        "pretrained/tokenizer/special_tokens_map.json",
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
    monkeypatch.setenv("HF_TOKEN", "preferred")
    monkeypatch.setenv("HF_KEY", "legacy")
    assets.prepare_assets(config)
    assert download.call_count == (2 if empty else 1)
    if empty:
        assert download.call_args.kwargs["force_download"] is True
        assert download.call_args.kwargs["allow_patterns"] == [damaged]
    assert download.call_args.kwargs["revision"] == config.asset_revision
    assert download.call_args.kwargs["token"] == "preferred"
    download.reset_mock()
    assets.prepare_assets(config)
    download.assert_not_called()
    target.unlink()
    download.side_effect = None
    with pytest.raises(RuntimeError, match="missing or empty"):
        assets.prepare_assets(config)


def test_contract_covers_text_image_and_all_native_controls() -> None:
    contract = ModelContract.of(Zing)
    assert set(contract.commands) == {
        "example_image",
        "release_controls",
        "reset",
        "set_image",
        "set_key",
        "set_prompt",
    }
    assert "fps" not in Zing.__dict__
    assert Zing.buffer_size == 16
    assert set(ZingOutput.__tracks__) == {"main_video"}


def test_released_cache_and_chunk_geometry_are_preserved() -> None:
    config = read_config(
        Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test")
    )
    source_override = os.environ.get("ZING_TEST_SOURCE_PATH")
    if source_override:
        config = replace(config, source_path=Path(source_override))
    import sys

    sys.path.insert(0, str(config.source_path / "src"))
    load_config = pytest.importorskip("zing_v0_5.config").load_config
    upstream = load_config(config.source_path / "config" / "zing.yaml")
    assert upstream.generator.local_attn_size == LOCAL_ATTN_SIZE == 97
    assert upstream.generator.sink_size == SINK_SIZE == 9
    assert upstream.inference.frames_per_block == 4
    assert upstream.vae.temporal_scale == 4


def test_all_eight_keys_can_be_held_and_released() -> None:
    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model.state.prompt = "world"
    model.state._completed_chunks = 2

    async def mutate() -> None:
        for key in ("w", "a", "s", "d", "i", "j", "k", "l"):
            await model.set_key(cast(Any, key), True)
        assert model.state._pressed_keys == frozenset("wasdijkl")
        result = await model.release_controls()
        assert result.released_keys == sorted("wasdijkl")

    asyncio.run(mutate())


def test_prompt_switch_does_not_reset_an_active_rollout() -> None:
    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model.state.prompt = "first"
    model.state._applied_world_id = model.state._world_epoch
    model.state._completed_chunks = 3
    model.state._active_prompt = "first"
    message = asyncio.run(model.set_prompt("second"))
    assert message.applies_to_chunk == 4
    assert message.resets_rollout is False
    assert model.state._applied_world_id == model.state._world_epoch


def test_image_upload_is_real_and_moderated(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (64, 32), (20, 40, 60)).save(path)
    prepare_image(
        UploadedFile(name=path.name, mime_type="image/png", data=path.read_bytes())
    )
    assert (
        ModelContract.of(Zing)
        .commands["set_image"]
        .command.__command_fields__["image"]
        .info.moderate
    )


def test_new_session_waits_for_user_input() -> None:
    class IdleBackend:
        def reset(self, **_: object) -> None:
            raise AssertionError("idle session must not reset the backend")

        def generate_chunk(self, **_: object) -> np.ndarray:
            raise AssertionError("idle session must not generate")

        def cache_frames(self) -> int:
            return 0

        def end_session(self) -> None:
            pass

    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model._config = read_config(
        Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test")
    )
    model.engine.backend = IdleBackend()
    model.on_session_started()
    assert model.state.prompt == ""
    assert not model._state_update().reset_queued
    assert model.state._conditioning == "none"
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_one_backend_call_maps_to_one_reactor_output() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self, **_: object) -> None:
            pass

        def generate_chunk(self, **_: object) -> np.ndarray:
            self.calls += 1
            return np.zeros((16, 704, 1248, 3), dtype=np.uint8)

        def cache_frames(self) -> int:
            return 4 * self.calls

        def end_session(self) -> None:
            pass

    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model.state.prompt = "A traversable courtyard"
    model.state._conditioning = "text"
    model._config = read_config(
        Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test")
    )
    backend = FakeBackend()
    model.engine.backend = backend
    model.engine.config = model._config

    async def run() -> None:
        result = model.generate(await model.process_input())
        output = await model.process_output(StepOutcome(result=result, elapsed=0.25))
        assert cast(np.ndarray, output.main_video).shape == (16, 704, 1248, 3)

    asyncio.run(run())
    assert backend.calls == 1


def test_ten_chunks_preserve_world_anchor_and_native_cache(tmp_path):
    class Backend:
        calls = 0
        resets = 0

        def reset(self, **kwargs):
            assert kwargs["image"] is not None
            self.resets += 1

        def generate_chunk(self, **kwargs):
            self.calls += 1
            return np.zeros((16, 8, 8, 3), dtype=np.uint8)

        def cache_frames(self):
            return self.calls * 4

    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model.state._conditioning = "uploaded"
    model.state._image = np.zeros((8, 8, 3), dtype=np.uint8)
    model._config = read_config(Path(__file__).parents[1] / "zing.yaml", tmp_path)
    model.engine.config = model._config
    model.engine.backend = backend = Backend()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send

    async def run():
        for index in range(1, 11):
            input = await model.process_input()
            assert (input.image is not None) == (index == 1)
            result = model.generate(input)
            assert result.index == index
            await model.process_output(StepOutcome(result=result, elapsed=0.25))

    asyncio.run(run())
    assert (backend.resets, backend.calls) == (1, 10)
    assert model.state._completed_chunks == 10
    assert all(
        m.generation_seconds == 0.25
        for m in messages
        if type(m).__name__ == "ChunkCompleted"
    )
    states = [m for m in messages if type(m).__name__ == "StateUpdate"]
    assert len(states) == 10
    assert [m.completed_chunks for m in states] == list(range(1, 11))


def test_generate_reads_only_input_and_failure_never_completes():
    model = Zing()
    model.engine = ZingModel()
    marker = object()
    model.engine = SimpleNamespace(generate=lambda input: input)
    model.state = None
    assert model.generate(marker) is marker
    model.state = ZingState()

    def fail(input):
        raise ValueError("native failure")

    model.engine = SimpleNamespace(generate=fail)
    with pytest.raises(ValueError) as failure:
        model.generate(marker)
    with pytest.raises(ValueError, match="native failure"):
        asyncio.run(model.process_output(StepOutcome(error=failure.value)))
    assert model.state._completed_chunks == 0
    assert model.state._applied_world_id is None


def test_refusal_and_frozen_unacknowledged_anchor():
    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model.engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.engine.generate.assert_not_called()
    model.state._conditioning = "uploaded"
    model.state._image = np.zeros((8, 8, 3), np.uint8)
    step = asyncio.run(model.process_input())
    with pytest.raises(FrozenInstanceError):
        step.world_id = 99
    assert asyncio.run(model.process_input()).image is step.image


def test_native_failure_preserves_completed_index():
    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model.state._conditioning = "text"
    model.engine.config = SimpleNamespace(max_chunks=32)
    model.engine.backend = backend = Mock()
    backend.generate_chunk.return_value = np.zeros((16, 8, 8, 3), np.uint8)
    backend.cache_frames.return_value = 5
    result = model.generate(asyncio.run(model.process_input()))
    asyncio.run(model.process_output(StepOutcome(result=result, elapsed=0.1)))
    assert model.state._completed_chunks == model.engine.index == 1
    backend.generate_chunk.side_effect = RuntimeError("native failure")
    model.send = Mock()
    with pytest.raises(RuntimeError) as error:
        model.generate(asyncio.run(model.process_input()))
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(model.process_output(StepOutcome(error=error.value)))
    assert model.state._completed_chunks == model.engine.index == 1
    model.send.assert_not_called()


def test_limit_retains_world_and_releases_controls():
    """Exhaustion freezes the backend until an explicit reset creates a new epoch."""

    class Backend:
        resets = 0
        calls = 0

        def reset(self, **kwargs):
            self.resets += 1

        def generate_chunk(self, **kwargs):
            self.calls += 1
            return np.zeros((16, 8, 8, 3), dtype=np.uint8)

        def cache_frames(self):
            return self.calls * 4

    model = Zing()
    model.engine = ZingModel()
    model.state = ZingState()
    model._config = replace(
        read_config(Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test")),
        max_chunks=2,
    )
    model.engine.backend = backend = Backend()
    model.engine.config = model._config
    model.on_session_started()
    messages = []

    async def record(message):
        messages.append(message)

    model.send = record

    async def run():
        async def step():
            result = model.generate(await model.process_input())
            return await model.process_output(StepOutcome(result=result, elapsed=0.1))

        await model.set_prompt("A courtyard")
        epoch = model.state._world_epoch
        assert await step() is not None
        await model.set_key("w", True)
        assert await step() is not None
        assert model._state_update().limit_reached
        assert model.state._world_epoch == epoch
        assert model.state._completed_chunks == 2
        assert not model.state._pressed_keys
        for _ in range(3):
            with pytest.raises(ApplicationError):
                await step()
        assert (backend.resets, backend.calls) == (1, 2)
        assert (
            sum(type(message).__name__ == "RolloutLimitReached" for message in messages)
            == 1
        )
        with pytest.raises(CommandError):
            await model.set_key("w", True)
        await model.set_key("w", False)
        await model.release_controls()
        await model.reset(42)
        assert model.state._world_epoch == epoch + 1
        assert not model.state._limit_reached
        assert await step() is not None
        assert (backend.resets, backend.calls) == (2, 3)

    asyncio.run(run())


def test_native_key_values_and_fixed_memory_geometry():
    from zing_model import NATIVE_KEYS, action_values

    assert (LOCAL_ATTN_SIZE, SINK_SIZE) == (97, 9)
    for index, key in enumerate(NATIVE_KEYS):
        expected = [0.0] * 8
        expected[index] = 1.0
        assert action_values([key]) == expected


def test_reset_without_scene_is_rejected():
    model = Zing()
    model.state = ZingState()
    with pytest.raises(CommandError, match="Select"):
        asyncio.run(model.reset(42))
    assert model.state._world_epoch == 0


def test_missing_example_keeps_current_selection(tmp_path, monkeypatch):
    import zing

    monkeypatch.setattr(zing, "__file__", str(tmp_path / "zing.py"))
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "retained"
    model._config = SimpleNamespace(source_path=tmp_path, example_prompt="new")
    with pytest.raises(CommandError, match="unavailable"):
        asyncio.run(model.example_image())
    assert model.state.prompt == "retained"


def test_fake_model_final_message_order():
    from unittest.mock import AsyncMock

    from zing_model import ZingResult

    model = Zing()
    model.state = ZingState()
    model.engine = Mock()
    model._config = SimpleNamespace(max_chunks=1)
    model.send = AsyncMock()
    asyncio.run(model.set_prompt("forest"))
    asyncio.run(model.set_key("k", True))
    step = asyncio.run(model.process_input())
    result = ZingResult(step.world_id, np.zeros((16, 2, 2, 3), np.uint8), 1, True)
    model.engine.generate.return_value = result
    assert model.generate(step) is result
    model.send.reset_mock()
    asyncio.run(model.process_output(StepOutcome(result=result)))
    messages = [call.args[0] for call in model.send.call_args_list]
    assert [type(m).__name__ for m in messages] == [
        "ChunkCompleted",
        "RolloutLimitReached",
        "StateUpdate",
    ]
    assert messages[0].action_keys == ["k"]
    assert messages[-1].pressed_keys == []


def test_model_failure_limit_and_reset():
    from zing_model import NotSeeded, RolloutComplete, ZingInput

    model = ZingModel()
    model.config = SimpleNamespace(max_chunks=1)
    model.backend = Mock()
    with pytest.raises(NotSeeded):
        model.generate(ZingInput(1, None, "forest", 42, frozenset(), True))
    model.backend.generate_chunk.return_value = np.zeros((16, 2, 2, 3), np.uint8)
    step = ZingInput(1, None, "forest", 42, frozenset())
    assert model.generate(step).complete
    with pytest.raises(RolloutComplete):
        model.generate(step)
    assert model.backend.generate_chunk.call_count == 1
    model.reset()
    assert model.generate(step).index == 1


def test_model_graph_without_runtime():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith('reactor_runtime'):raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
import zing_model
zing_model.ZingModel()
""",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
    )
