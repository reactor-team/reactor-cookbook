"""Verify native Runtime 3.5 chunk boundaries without loading model weights."""

import asyncio
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from abot_world import ABotWorld
from abot_world_assets import ExampleScene
from abot_world_model import (
    ABotAnchor,
    ABotInput,
    ABotResult,
    ABotWorldModel,
    NoAnchor,
    RolloutExhausted,
)
from abot_world_types import ABotWorldState
from reactor_runtime import ApplicationError, ReactorApp, StepOutcome, UploadedFile


def test_step_waits_for_image() -> None:
    model = ABotWorld()
    model.state = ABotWorldState()
    assert isinstance(model, ReactorApp)
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_ten_steps_snapshot_controls_and_keep_rollout() -> None:
    model = ABotWorld()
    model.state = ABotWorldState()
    model._selected_input = Path("anchor.png")
    model.state.prompt = "A stable landscape"
    model._config = SimpleNamespace(max_chunks=512, max_chunks_per_rollout=512)
    engine = _Engine()
    model._engine = engine

    async def discard(*_):
        pass

    model.send = discard

    async def run():
        for index in range(10):
            model.state.prompt = f"Scene {index}"
            key = ["W", "A", "S", "D", "I", "J", "K", "L", "W", "D"][index]
            model.state._activated_keys = frozenset({key})
            snapshot = await model.process_input()
            assert isinstance(snapshot, ABotInput)
            assert snapshot.action[key]
            assert (snapshot.anchor is not None) == (index == 0)
            assert not model.state._activated_keys
            with pytest.raises(FrozenInstanceError):
                snapshot.prompt = "mutated"
            model.state.prompt = "later input"
            result = model.generate(snapshot)
            output = await model.process_output(StepOutcome(result=result))
            assert output.main_video.shape == (9 if index == 0 else 12, 8, 8, 3)
            assert engine.inputs[-1].prompt == f"Scene {index}"
            assert model._active_prompt == f"Scene {index}"
            assert model._sampled_keys == frozenset({key})
        assert model._chunk_index == 10

    asyncio.run(run())
    assert len(engine.inputs) == 10
    assert len({input.world_id for input in engine.inputs}) == 1


def test_generation_error_releases_busy_flags() -> None:
    model = ABotWorld()
    model.state = ABotWorldState()
    model._reset_in_flight = True
    model._chunk_in_flight = True
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(
            model.process_output(StepOutcome(error=RuntimeError("native failure")))
        )
    assert not model._reset_in_flight
    assert not model._chunk_in_flight


class _Engine:
    def __init__(self, limit=512):
        self.inputs = []
        self.world_id = None
        self.index = 0
        self.limit = limit
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
        return ABotResult(
            np.zeros((9 if self.index == 1 else 12, 8, 8, 3), np.uint8),
            self.world_id,
            self.index,
            frozenset(key for key, active in input.action.items() if active),
            input.prompt,
            self.index >= self.limit,
        )

    def reset(self):
        self.resets += 1


def _app():
    app = ABotWorld()
    app.state = ABotWorldState()
    app._config = SimpleNamespace(max_chunks=512, seed=42, examples=())
    app.on_session_started()
    engine = _Engine()
    app._engine = engine
    messages = []

    async def send(message):
        messages.append(message)

    app.send = send
    return app, engine, messages


def _upload():
    import io

    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 50, 100)).save(output, format="PNG")
    return UploadedFile(name="scene.png", mime_type="image/png", data=output.getvalue())


def test_refusals_never_call_engine():
    app, engine, _ = _app()
    with pytest.raises(ApplicationError, match="Select an image"):
        asyncio.run(app.process_input())
    app._selected_input = Path("anchor.png")
    app.state._limit_reached = True
    with pytest.raises(ApplicationError, match="rollout limit"):
        asyncio.run(app.process_input())
    assert not engine.inputs


def test_generate_reads_only_input():
    app, engine, _ = _app()
    app._selected_input = Path("anchor.png")
    input = asyncio.run(app.process_input())
    app.state = app._config = app._selected_input = None
    result = app.generate(input)
    assert engine.inputs == [input] and result.chunk_index == 1


def test_failure_does_not_ack_or_complete():
    app, engine, messages = _app()
    app._selected_input = Path("anchor.png")
    engine.error = RuntimeError("native failure")

    async def run():
        input = await app.process_input()
        try:
            app.generate(input)
        except RuntimeError as error:
            with pytest.raises(RuntimeError, match="native failure"):
                await app.process_output(StepOutcome(error=error))
        assert app.state._applied_world_id is None
        assert app._chunk_index == 0 and not messages
        assert not app._chunk_in_flight and not app._reset_in_flight
        assert (await app.process_input()).anchor is not None

    asyncio.run(run())


def test_upload_live_prompt_reset_taps_and_session_cleanup():
    app, engine, _ = _app()

    async def step():
        input = await app.process_input()
        await app.process_output(StepOutcome(result=app.generate(input)))
        return input

    async def run():
        await app.set_image(_upload(), "First")
        await app.set_key_state("W", True)
        await app.set_key_state("W", False)
        first = await step()
        assert first.anchor.image == _upload().data and first.anchor.suffix == ".png"
        assert first.action["W"] and not app.state._activated_keys
        await app.set_prompt("Second")
        second = await step()
        assert second.world_id == first.world_id and second.anchor is None
        assert second.prompt == "Second" and not second.action["W"]
        await app.reset(123)
        reset = await step()
        assert reset.anchor.seed == 123 and reset.world_id != first.world_id
        assert app._chunk_index == 1
        await app.set_key_state("J", True)
        await app.release_controls()
        assert not app.state._pressed_keys
        app.on_session_ended()
        assert engine.resets == 1 and app._selected_input is None
        assert app.state._applied_world_id is None

    asyncio.run(run())


def test_builtin_image_and_rollout_limit(tmp_path):
    app, engine, messages = _app()
    path = tmp_path / "scene.png"
    path.write_bytes(_upload().data)
    app._config = SimpleNamespace(
        max_chunks=1, seed=42, examples=(ExampleScene(path, "Example"),)
    )
    engine.limit = 1

    async def run():
        await app.random_image()
        input = await app.process_input()
        assert input.anchor.image == path
        messages.clear()
        await app.process_output(StepOutcome(result=app.generate(input)))
        assert app.state._limit_reached
        assert [type(m).__name__ for m in messages] == [
            "RolloutLimitReached",
            "StateUpdate",
        ]
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(-1)
        assert not app.state._limit_reached

    asyncio.run(run())


class _Pipeline:
    def __init__(self):
        self.events = []
        self.kv_cache1 = None
        self.index = 0
        self.vae = SimpleNamespace(
            model=SimpleNamespace(clear_cache=lambda: self.events.append("vae_clear")),
            taehv=SimpleNamespace(reset=lambda: self.events.append("taehv_reset")),
        )

    def set_prompts(self, prompts, **kwargs):
        self.events.append(("prompt", prompts))

    def set_ref_latent_mask_from_exists_paths(self, **kwargs):
        self.events.append("reference_mask")

    def reset_stream(self, **kwargs):
        self.events.append("reset_stream")
        self.kv_cache1 = object()
        self.index = 0

    def set_first_frame_latent(self, path, **kwargs):
        self.events.append(("image", Path(path), Path(path).read_bytes()))

    def set_act(self, action, **kwargs):
        self.events.append(("action", action, kwargs))

    def generate_next_block(self, noise):
        self.events.append("generate")
        return noise

    def decode_block_and_write(self, latent, capture):
        for _ in range(9 if self.index == 0 else 12):
            capture.append_data(np.zeros((8, 8, 3), np.uint8))
        self.index += 1


def test_native_model_world_cache_prompt_and_cleanup(tmp_path):
    model = ABotWorldModel()
    pipeline = _Pipeline()
    model._pipeline = pipeline
    model._device = "fake_cuda"
    model._config = SimpleNamespace(
        max_chunks=2, height=8, width=8, checkpoint=SimpleNamespace(path=tmp_path)
    )
    model._weights_root = tmp_path
    model._latent_shape = (1, 3, 4, 2, 2)
    model._modules = {
        "set_seed": lambda seed: pipeline.events.append(("seed", seed)),
        "torch": SimpleNamespace(
            bfloat16="bf16", randn=lambda shape, **kwargs: np.zeros(shape)
        ),
    }
    anchor = ABotAnchor(_upload().data, ".png", 123)
    with pytest.raises(NoAnchor):
        model.generate(ABotInput(1, None, "World", {}))
    first = model.generate(ABotInput(1, anchor, "World", {"W": True}))
    cache = pipeline.kv_cache1
    second = model.generate(ABotInput(1, None, "New prompt", {"J": True}))
    assert first.frames.shape[0] == 9 and second.frames.shape[0] == 12
    assert second.complete and second.prompt == "New prompt"
    assert second.sampled_keys == frozenset({"J"}) and pipeline.kv_cache1 is cache
    assert pipeline.events.count("reset_stream") == 1
    images = [
        event
        for event in pipeline.events
        if isinstance(event, tuple) and event[0] == "image"
    ]
    assert images[0][2] == anchor.image and not images[0][1].exists()
    with pytest.raises(RolloutExhausted):
        model.generate(ABotInput(1, None, "World", {}))
    fresh = model.generate(ABotInput(2, anchor, "World", {}))
    assert fresh.chunk_index == 1 and fresh.frames.shape[0] == 9
    model.reset()
    assert pipeline.events[-2:] == ["vae_clear", "taehv_reset"]
    with pytest.raises(NoAnchor):
        model.generate(ABotInput(2, None, "World", {}))


def test_model_import_graph_is_runtime_free():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "reactor_runtime" or name.startswith("reactor_runtime."):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import abot_world_model
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )
