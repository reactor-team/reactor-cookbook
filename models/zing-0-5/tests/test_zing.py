"""Contract and upstream-fidelity tests for the Zing adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from zing import Zing
from zing_assets import activate_source, read_config
from zing_images import validate_image
from zing_types import ZingOutput, ZingState


def test_contract_covers_text_image_and_all_native_controls() -> None:
    contract = ModelContract.of(Zing)
    assert set(contract.commands) == {
        "example_image", "release_controls", "reset", "set_image", "set_key", "set_prompt"
    }
    assert "fps" not in Zing.__dict__
    assert Zing.buffer_size == 16
    assert set(ZingOutput.__tracks__) == {"main_video"}


def test_released_cache_and_chunk_geometry_are_preserved() -> None:
    config = read_config(Path(__file__).parents[1] / "zing.yaml")
    activate_source(config)
    from zing_v0_5.config import load_config
    upstream = load_config(config.source_path / "config" / "zing.yaml")
    assert upstream.generator.local_attn_size == config.local_attn_size == 97
    assert upstream.generator.sink_size == config.sink_size == 9
    assert upstream.inference.frames_per_block == 4
    assert upstream.vae.temporal_scale == 4


def test_all_eight_keys_can_be_held_and_released() -> None:
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "world"
    model._completed_chunks = 2
    async def mutate() -> None:
        for key in ("w", "a", "s", "d", "i", "j", "k", "l"):
            await model.set_key(cast(Any, key), True)
        assert model.state._pressed_keys == frozenset("wasdijkl")
        result = await model.release_controls()
        assert result.released_keys == sorted("wasdijkl")
    asyncio.run(mutate())


def test_prompt_switch_does_not_reset_an_active_rollout() -> None:
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "first"
    model.state._reset_requested = False
    model._completed_chunks = 3
    model._active_prompt = "first"
    message = asyncio.run(model.set_prompt("second"))
    assert message.applies_to_chunk == 4
    assert message.resets_rollout is False
    assert model.state._reset_requested is False


def test_image_upload_is_real_and_moderated(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (64, 32), (20, 40, 60)).save(path)
    validate_image(UploadedFile(name=path.name, mime_type="image/png", data=path.read_bytes()))


def test_new_session_waits_for_user_input() -> None:
    class IdleBackend:
        def reset(self, **_: object) -> None:
            raise AssertionError("idle session must not reset the backend")
        def generate_chunk(self, **_: object) -> np.ndarray:
            raise AssertionError("idle session must not generate")
        def cache_frames(self) -> int: return 0
        def end_session(self) -> None: pass
    model = Zing()
    model.state = ZingState()
    model._config = read_config(Path(__file__).parents[1] / "zing.yaml")
    model._backend = IdleBackend()
    model.on_session_started()
    assert model.state.prompt == ""
    assert model.state._reset_requested is False
    assert model._conditioning == "none"
    assert asyncio.run(anext(model.inference())) is None


def test_one_backend_call_maps_to_one_reactor_output() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0
        def reset(self, **_: object) -> None: pass
        def generate_chunk(self, **_: object) -> np.ndarray:
            self.calls += 1
            return np.zeros((16, 704, 1248, 3), dtype=np.uint8)
        def cache_frames(self) -> int: return 4 * self.calls
        def end_session(self) -> None: pass
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "A traversable courtyard"
    model.state._reset_requested = True
    model._config = read_config(Path(__file__).parents[1] / "zing.yaml")
    backend = FakeBackend()
    model._backend = backend
    async def run() -> None:
        outputs = model.inference()
        output = None
        while output is None:
            output = await anext(outputs)
        assert cast(np.ndarray, output.main_video).shape == (16, 704, 1248, 3)
    asyncio.run(run())
    assert backend.calls == 1
