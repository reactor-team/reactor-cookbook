"""Test the application/model boundary without loading native policy weights."""

import asyncio
import subprocess
import sys
from collections import deque
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome
from rldx1 import RLDXPipeline, _Publication
from rldx1_model import PolicyNotLoaded, RLDXModel, RLDXModelInput, RLDXModelResult
from rldx1_types import ActionPrediction, RLDXState


def test_application_refuses_without_calling_model():
    app = RLDXPipeline()
    app.state = RLDXState()
    app._video_deltas = [0]
    app._control_hz, app._exec_horizon, app._pace = 20, 16, True
    app._rtc_timing = SimpleNamespace(enabled=True)
    app._views = ("left_view",)
    app._recent = {"left_view": deque()}
    app._schema_pending = False
    app._applied_episode_id = app._episode_id
    app.input = SimpleNamespace(left_view=SimpleNamespace(try_read=lambda *a, **k: []))
    app._engine = SimpleNamespace(
        generate=lambda _: pytest.fail("refusal reached model")
    )
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())


def test_fake_model_forwarding_and_publication():
    app = RLDXPipeline()
    app.state = None
    actions = {
        f"action.{key}": np.zeros((1, 16, 1))
        for key in (
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
        )
    }
    request = RLDXModelInput(7, {"state.example": np.ones((1, 1, 1))}, None)
    calls, sent = [], []

    def generate(value):
        calls.append(value)
        return RLDXModelResult(7, actions, 1)

    async def send(message):
        sent.append(message)

    app._engine = SimpleNamespace(generate=generate)
    app.send = send
    app._publication = _Publication(None, 123, 42, 2, 4.0, ())
    result = app.generate(request)
    assert calls == [request]
    asyncio.run(app.process_output(StepOutcome(result=result, elapsed=0.25)))
    assert len(sent) == 1 and isinstance(sent[0], ActionPrediction)
    assert sent[0].step == 0 and sent[0].source_seq == 42
    assert app._applied_episode_id == 7 and app._last_completed_predictions == 1
    assert not hasattr(result, "input")
    with pytest.raises(FrozenInstanceError):
        request.episode_id = 8


def test_generate_failure_does_not_acknowledge_or_publish():
    app = RLDXPipeline()
    sent = []

    def fail(_):
        raise ValueError("native policy failure")

    async def send(message):
        sent.append(message)

    app._engine = SimpleNamespace(generate=fail)
    app.send = send
    app._publication = _Publication(None, None, None, None, None, ())
    with pytest.raises(ValueError) as captured:
        app.generate(RLDXModelInput(1, {}, None))
    with pytest.raises(ValueError, match="native policy failure"):
        asyncio.run(app.process_output(StepOutcome(error=captured.value)))
    assert app._applied_episode_id is None
    assert app._last_completed_predictions == 0 and not sent
    assert app._publication is None


def test_native_model_ten_requests_episode_and_failure():
    model = RLDXModel()
    with pytest.raises(PolicyNotLoaded):
        model.generate(RLDXModelInput(1, {}, None))
    resets, calls = [], []

    def predict(observation, options):
        calls.append((observation, options))
        if observation.get("fail"):
            raise RuntimeError("native failure")
        return {"action.test": np.asarray([[[observation["value"]]]])}, {}

    model._policy = SimpleNamespace(
        reset=lambda: resets.append(True), get_action=predict
    )
    setup = model.generate(RLDXModelInput(1, None, None))
    assert setup.completed_predictions == 0 and setup.actions is None
    for i in range(10):
        observation, options = {"value": i}, {"rtc_prefix_len": i % 3}
        result = model.generate(RLDXModelInput(1, observation, options))
        assert calls[-1] == (observation, options)
        assert result.completed_predictions == i + 1
        assert result.actions["action.test"].item() == i
    assert len(resets) == 1
    with pytest.raises(RuntimeError, match="native failure"):
        model.generate(RLDXModelInput(1, {"fail": True}, None))
    assert model._completed_predictions == 10
    fresh = model.generate(RLDXModelInput(2, {"value": 99}, None))
    assert fresh.completed_predictions == 1 and len(resets) == 2
    model.reset()
    assert model._episode_id is None and model._completed_predictions == 0


def test_native_reset_retains_legacy_exception_tolerance():
    model = RLDXModel()

    def failed_reset():
        raise RuntimeError("legacy reset failure")

    model._policy = SimpleNamespace(reset=failed_reset)
    result = model.generate(RLDXModelInput(1, None, None))
    assert result.episode_id == 1 and result.actions is None


def test_model_import_graph_is_runtime_free():
    source = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import rldx1_model
assert rldx1_model.RLDXModel()._policy is None
"""
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
