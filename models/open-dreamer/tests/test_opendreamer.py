"""Verify native step-loop conditioning, caches and transient controls without a GPU."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import opendreamer_model
import pytest
from opendreamer import OpenDreamer
from opendreamer_types import OpenDreamerState
from opendreamer_utils import ConditioningActions, RolloutConditioning
from reactor_runtime import ApplicationError, StepOutcome


def ready_model(monkeypatch):
    """Create a two-frame conditioning sequence and deterministic mock JAX calls."""
    model = OpenDreamer()
    model.state = OpenDreamerState()
    model._config = SimpleNamespace(seed=42, conditioning_frames=2)
    model._model._config = model._config
    model._model_frame_shape = (8, 8, 3)
    model.output = SimpleNamespace(flush=lambda: None)

    async def send(message):
        pass

    model.send = send
    model._model._latent_shape = (1, 1, 1, 1)
    model._model._empty_dynamics_cache = 0
    model._model._empty_tokenizer_cache = 0
    model._model._deps = {
        "jax": SimpleNamespace(
            random=SimpleNamespace(
                PRNGKey=lambda seed: seed, split=lambda seed: (seed + 1, seed + 2)
            ),
            block_until_ready=lambda value: value,
        ),
        "jnp": np,
        "action_type": SimpleNamespace,
        "mouse_to_categorical": lambda x, y: x + y,
    }
    model._model._key_to_index = {
        "key.keyboard.w": 0,
        "mouse.0": 1,
        "mouse.wheel_neg": 2,
        "mouse.wheel_pos": 3,
    }
    conditioning = RolloutConditioning(
        np.zeros((2, 8, 8, 3), dtype=np.uint8),
        ConditioningActions(
            binary=np.zeros((1, 2, 4)), categorical=np.zeros((1, 2)), continuous=None
        ),
    )
    model._demos = {"demo_1": conditioning}
    model._model._conditioning_source = "demo_1"
    calls = []

    def observe(tokenizer, dynamics, frame, action, dc, tc):
        calls.append(("observe", dc, tc))
        return dc + 1, tc + 1

    def generate(tokenizer, dynamics, action, shape, dc, tc, rng):
        calls.append(("generate", dc, tc, action))
        return np.zeros((1, 1, 8, 8, 3), dtype=np.uint8), dc + 1, tc + 1, rng

    model._model._observe_frame_jit = observe
    model._model._next_frame_jit = generate
    monkeypatch.setattr(opendreamer_model, "mesh_context", lambda *args: nullcontext())
    model.state._applied_world_id = None
    return model, calls


def step(model):
    """Drive the three public step hooks with no live state visible to generation."""
    input = asyncio.run(model.process_input())
    state = model.state
    model.state = None
    result = model.generate(input)
    model.state = state
    return asyncio.run(model.process_output(StepOutcome(result=result)))


def test_conditioning_then_ten_actions_preserve_cache(monkeypatch):
    model, calls = ready_model(monkeypatch)
    model.state._delta_x = 5
    assert step(model) is None
    assert step(model) is None
    assert model.state._delta_x == 5
    for index in range(10):
        model.state._pressed_keys = frozenset({"w"})
        model.state._delta_x = index + 1
        model.state._wheel_delta = 1
        output = step(model)
        assert output.main_video.shape == (8, 8, 3)
        assert model.state._delta_x == model.state._wheel_delta == 0
        assert model.state._pressed_keys == frozenset({"w"})
        np.testing.assert_array_equal(calls[-1][3].binary, [[1, 0, 0, 1]])
    assert [call[0] for call in calls] == ["observe"] * 2 + ["generate"] * 10
    assert model._model._dynamics_cache == model._model._tokenizer_cache == 12
    assert "fps" not in vars(OpenDreamer)


def test_reset_reobserves_conditioning_and_session_end_releases_cache(monkeypatch):
    model, calls = ready_model(monkeypatch)
    for _ in range(4):
        step(model)
    model._queue_rollout_reset()
    assert step(model) is None
    assert calls[-1] == ("observe", 0, 0)
    model.on_session_ended()
    assert model._model._dynamics_cache is None
    assert model._model._tokenizer_cache is None
    assert model._model._conditioning is None


def test_missing_conditioning_refuses_and_error_does_not_consume_input(monkeypatch):
    model, _ = ready_model(monkeypatch)
    model._demos = {}
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.state._delta_x = 9
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=RuntimeError("failed"))))
    assert model.state._delta_x == 9
    assert model.state._applied_world_id != model._world_id


def test_model_boundary_imports_without_runtime():
    """Import the entire model/config/assets graph with runtime imports blocked."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.startswith("reactor_runtime"):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import opendreamer_model, opendreamer_utils, opendreamer_assets
opendreamer_model.OpenDreamerModel()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )


def test_anchor_acknowledged_while_observing_and_sent_once(monkeypatch):
    """Conditioning-only results acknowledge initialization before video exists."""
    app, calls = ready_model(monkeypatch)
    initial = asyncio.run(app.process_input())
    assert initial.anchor is not None
    result = app.generate(initial)
    assert result.frame is None
    assert result.observation_index == 1
    assert result.generated_frames == 0
    assert asyncio.run(app.process_output(StepOutcome(result=result))) is None
    assert asyncio.run(app.process_input()).anchor is None
    # Even a retried first snapshot must not wipe existing conditioning caches.
    second = app.generate(initial)
    assert second.observation_index == 2
    assert calls[-1] == ("observe", 1, 1)


def test_generate_reads_only_snapshot(monkeypatch):
    """The generation hook delegates once even when all app-side state is absent."""
    app, _ = ready_model(monkeypatch)
    snapshot = asyncio.run(app.process_input())
    expected = object()
    calls = []

    class FakeModel:
        def generate(self, value):
            calls.append(value)
            return expected

    app._model = FakeModel()
    app.state = None
    app._config = None
    app._demos = None
    assert app.generate(snapshot) is expected
    assert calls == [snapshot]


def test_refusal_never_calls_model(monkeypatch):
    """Missing conditioning is refused before any inference invocation."""
    app, calls = ready_model(monkeypatch)
    app._demos = {}
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    assert not calls


def test_failed_generation_does_not_count_or_consume_controls(monkeypatch):
    """Errors preserve successful-frame count and pending client controls."""
    app, _ = ready_model(monkeypatch)
    step(app)
    step(app)
    app.state._delta_x = 7

    def fail(*args):
        raise RuntimeError("inference failed")

    app._model._next_frame_jit = fail
    snapshot = asyncio.run(app.process_input())
    with pytest.raises(RuntimeError, match="inference failed"):
        app.generate(snapshot)
    with pytest.raises(RuntimeError, match="inference failed"):
        asyncio.run(
            app.process_output(StepOutcome(error=RuntimeError("inference failed")))
        )
    assert app._model._generated_frames == 0
    assert app._last_result.generated_frames == 0
    assert app.state._delta_x == 7


def test_upload_reset_and_stale_ack(monkeypatch):
    """Uploaded images remain CPU-only; old-world output cannot acknowledge reset."""
    import io

    from PIL import Image

    app, _ = ready_model(monkeypatch)
    initial = app.generate(asyncio.run(app.process_input()))
    data = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(data, format="PNG")
    reply = asyncio.run(
        app.set_conditioning_image(
            SimpleNamespace(
                data=data.getvalue(), mime_type="image/png", name="test.png"
            )
        )
    )
    assert reply.selection == "test.png"
    assert app._uploaded_conditioning.actions is None
    assert app._uploaded_conditioning.frames.shape == (2, 8, 8, 3)
    assert asyncio.run(app.process_output(StepOutcome(result=initial))) is None
    assert app.state._applied_world_id != app._world_id
    snapshot = asyncio.run(app.process_input())
    assert snapshot.world_id != initial.world_id
    assert snapshot.anchor.conditioning is app._uploaded_conditioning
    app._model._deps["camera_classes"] = 121
    assert step(app) is None
    assert app._last_result.observation_index == 1
    asyncio.run(app.reset(seed=17))
    assert asyncio.run(app.process_input()).anchor.seed == 17
    app.on_session_ended()
    assert app._uploaded_conditioning is None
    assert app._last_result is None
    assert app._model._world_id is None


@pytest.mark.parametrize("shape", [(8, 8, 4), (8, 8)])
def test_invalid_model_frame_does_not_count(monkeypatch, shape):
    """Only valid RGB model results increment generated-frame progress."""
    app, _ = ready_model(monkeypatch)
    step(app)
    step(app)
    app._model._next_frame_jit = lambda *args: (
        np.zeros((1, 1, *shape), dtype=np.uint8),
        3,
        3,
        99,
    )
    with pytest.raises(ValueError, match="Expected an RGB frame"):
        app.generate(asyncio.run(app.process_input()))
    assert app._model._generated_frames == 0
