"""Focused contract and lifecycle tests for the DIAMOND adapter."""

from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from reactor_runtime import ApplicationError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import load_config

EXAMPLE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

pipeline_module = importlib.import_module("diamond")
assets_module = importlib.import_module("diamond_assets")
types_module = importlib.import_module("diamond_types")
model_module = importlib.import_module("diamond_model")
Diamond = pipeline_module.Diamond
DiamondOutput = types_module.DiamondOutput
DiamondState = types_module.DiamondState
PreparedScene = model_module.PreparedScene
StateUpdate = types_module.StateUpdate


@dataclass
class _Action:
    keys: list[int]
    mouse_x: float
    mouse_y: float
    left_click: bool
    right_click: bool


class _Scalar:
    def __init__(self, value: bool) -> None:
        self._value = value

    def item(self) -> bool:
        return self._value


class _Client:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def send(self, message: Any) -> None:
        self.messages.append(message)


class _World:
    def __init__(self) -> None:
        self.act_buffer: Any = object()
        self.next_act: Any = SimpleNamespace(size=lambda _dimension: 0)
        self.obs_buffer: Any = None
        self.obs_full_res_buffer: Any = None
        self.reset_count = 0
        self.actions: list[Any] = []

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self.reset_count += 1
        self.act_buffer = object()
        self.next_act = SimpleNamespace(size=lambda _dimension: 0)
        self.obs_buffer = np.full((1, 4, 3, 1, 1), 1.0)
        self.obs_full_res_buffer = np.full((1, 4, 3, 2, 2), 2.0)
        return self.obs_full_res_buffer[:, -1], {}

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, _Scalar, _Scalar, _Scalar, dict[str, object]]:
        self.actions.append(action)
        observation = np.full((1, 3, 2, 2), 3.0 + len(self.actions))
        return observation, _Scalar(False), _Scalar(False), _Scalar(False), {}


def _ready_model() -> Any:
    model = Diamond()
    model.state = DiamondState()
    model._engine._agent = SimpleNamespace(device="test-device")
    model._engine._world = _World()
    model._engine._action_type = _Action
    model._engine._encode_action = lambda action, *, device: action
    model._engine._key_codes = {
        key: index for index, key in enumerate(types_module.KEYS, start=1)
    }
    model._start_session()
    return model


def _stub_video(monkeypatch: pytest.MonkeyPatch, observed: list[np.ndarray]) -> None:
    def convert(observation: np.ndarray) -> np.ndarray:
        observed.append(observation)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(model_module, "to_video_frame", convert)


def _step(model: Any) -> Any:
    input = asyncio.run(model.process_input())
    result = model.generate(input)
    return asyncio.run(model.process_output(StepOutcome(result=result)))


def test_contract_uses_session_hooks_and_documents_side_effects() -> None:
    """Expose session-scoped lifecycle hooks and a complete public schema."""
    contract = ModelContract.of(Diamond)

    assert contract.lifecycle.session_started is not None
    assert contract.lifecycle.session_ended is not None
    assert contract.lifecycle.connected is not None
    assert contract.lifecycle.disconnected is not None
    assert all("Emits" in command.description for command in contract.commands.values())
    assert all(
        field.info.description
        for command in contract.commands.values()
        for field in command.command.__command_fields__.values()
    )

    controller = contract.commands["set_controller"].command.__command_fields__[
        "controller"
    ]
    image = contract.commands["set_spawn_image"].command.__command_fields__["image"]
    assert "next model step" in controller.info.description
    assert "next model-step boundary" in image.info.description

    document = contract.render_schema().to_openapi()
    assert set(document["webhooks"]) == {
        "action_changed",
        "scene_changed",
        "state_update",
    }
    assert all(
        webhook["post"]["summary"].startswith("Emitted ")
        for webhook in document["webhooks"].values()
    )


def test_connect_sends_one_complete_state_snapshot() -> None:
    """Give a joining viewer the durable controls without replaying events."""
    model = _ready_model()
    model.state.controller = "human"
    model.state._pressed_keys = frozenset({"w", "space"})
    model.state._pressed_mouse_buttons = frozenset({"left"})
    client = _Client()

    asyncio.run(model._connected(client))

    assert client.messages == [
        StateUpdate(
            controller="human",
            pressed_keys=["w", "space"],
            pressed_mouse_buttons=["left"],
        )
    ]


def test_durable_control_change_broadcasts_a_state_snapshot() -> None:
    """Broadcast the complete durable controls after accepting a key change."""
    model = _ready_model()
    messages: list[Any] = []

    async def record(message: Any) -> None:
        messages.append(message)

    model.send = record
    reply = asyncio.run(model.set_key_state("w", True))

    assert reply.pressed_keys == ["w"]
    assert messages == [
        StateUpdate(
            controller="human",
            pressed_keys=["w"],
            pressed_mouse_buttons=[],
        )
    ]


def test_playout_uses_fixed_rate_with_short_buffer() -> None:
    """Play at DIAMOND's native rate with enough frames to absorb brief stalls."""
    assert Diamond.fps == 15
    assert Diamond.buffer_size == 4


def test_reconnect_preserves_the_session_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the shared world alive between connected viewers' steps."""
    model = _ready_model()
    world = model._engine._world
    _stub_video(monkeypatch, [])
    model._start_session()

    assert isinstance(_step(model), DiamondOutput)
    assert isinstance(_step(model), DiamondOutput)

    model.state = DiamondState()
    assert isinstance(_step(model), DiamondOutput)
    assert world.reset_count == 1
    assert len(world.actions) == 2


def test_session_end_discards_a_queued_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a queued scene from leaking into the next Reactor session."""
    model = _ready_model()
    world = model._engine._world
    _stub_video(monkeypatch, [])
    _step(model)
    model._pending_scene = model_module.DiamondAnchor(scene=Path("queued"))

    model._end_session()
    assert model._pending_scene is None
    assert model._engine._world_id is None
    assert world.obs_buffer is None

    model.state = DiamondState()
    model._start_session()
    _step(model)
    assert world.reset_count == 2
    assert model._engine._world_id == model._world_id


def test_ten_steps_keep_the_world_and_consume_mouse_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read snapshot controls without resetting the autoregressive world."""
    model = _ready_model()
    _stub_video(monkeypatch, [])
    _step(model)
    for index in range(10):
        model.state._pressed_keys = frozenset({"w"})
        model.state._delta_x = float(index + 1)
        input = asyncio.run(model.process_input())
        state = model.state
        model.state = None
        result = model.generate(input)
        model.state = state
        asyncio.run(model.process_output(StepOutcome(result=result)))
        assert model._engine._world.actions[-1].mouse_x == index + 1
        assert model.state._delta_x == 0
        assert model.state._pressed_keys == frozenset({"w"})
    assert model._engine._world.reset_count == 1
    assert len(model._engine._world.actions) == 10


def test_failed_step_does_not_consume_controls() -> None:
    """Propagate upstream failures without acknowledging a completed action."""
    model = _ready_model()
    model.state._delta_x = 10
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=RuntimeError("failed"))))
    assert model.state._delta_x == 10


def test_queued_scene_is_emitted_before_the_first_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit a queued initial frame before consuming a client action."""
    model = _ready_model()
    world = model._engine._world
    uploaded = np.full((1, 4, 3, 2, 2), 6.0)
    scene = PreparedScene(
        obs=np.full((1, 4, 3, 1, 1), 5.0),
        obs_full_res=uploaded,
        act=world.act_buffer,
        next_act=None,
    )
    model._pending_scene = model_module.DiamondAnchor(
        full_res=np.zeros((3, 2, 2), dtype=np.uint8),
        low_res=np.zeros((3, 1, 1), dtype=np.uint8),
    )
    monkeypatch.setattr(model._engine, "_prepare_uploaded_scene", lambda *_: scene)
    observed: list[np.ndarray] = []
    _stub_video(monkeypatch, observed)

    output = _step(model)

    assert isinstance(output, DiamondOutput)
    np.testing.assert_array_equal(observed, [uploaded[:, -1]])
    assert world.actions == []


def test_disconnect_releases_controls_and_broadcasts_state() -> None:
    """Return controls to neutral and inform remaining viewers on disconnect."""
    model = _ready_model()
    model.state._pressed_keys = frozenset({"w"})
    model.state._pressed_mouse_buttons = frozenset({"left"})
    model.state._delta_x = 4.0
    model.state._delta_y = -2.0
    messages: list[Any] = []

    async def record(message: Any) -> None:
        messages.append(message)

    model.send = record
    asyncio.run(model._disconnected())

    assert model.state._pressed_keys == frozenset()
    assert model.state._pressed_mouse_buttons == frozenset()
    assert model.state._delta_x == 0.0
    assert model.state._delta_y == 0.0
    assert messages == [
        StateUpdate(
            controller="human",
            pressed_keys=[],
            pressed_mouse_buttons=[],
        )
    ]


def test_scene_reset_flushes_pending_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discard frames queued by the world that a scene reset replaces."""
    model = _ready_model()
    flushes: list[None] = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    model._queue_scene_reset()

    assert flushes == [None]


def test_manifest_defines_the_runtime_entrypoint_and_generated_image() -> None:
    """Keep the entrypoint and generated image inputs reproducible."""
    manifest_path = EXAMPLE_DIR / "reactor.yaml"
    config = load_config(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    build = manifest["build"]

    assert config.model_ref == "diamond:Diamond"
    assert build["runtime_version"] == "3.5.0"
    assert build["python_requirements"] == "requirements.txt"
    assert build["cuda_version"] == "12.8.1"
    assert build["python_version"] == "3.12"
    assert build["system_packages"] == ["git"]
    assert build["runtime_env"]["DIAMOND_PATH"] == "/opt/diamond"
    assert "851cefb497733d27f1b85c804104638765860fca" in build["run"][0]
    assert not (EXAMPLE_DIR / "Dockerfile").exists()
    assert "reactor-runtime" not in (EXAMPLE_DIR / "requirements.txt").read_text()


def test_model_download_uses_the_runtime_weights_mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist Hugging Face assets under Reactor's mounted weights root."""
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", str(tmp_path))

    calls = []
    model = Diamond()
    setup = model_module.DiamondSetup((), 42, (150, 280), (30, 56))
    monkeypatch.setattr(pipeline_module, "upstream_root", lambda: tmp_path / "source")
    monkeypatch.setattr(
        model._engine, "load", lambda *args: calls.append(args) or setup
    )
    model.load(tmp_path / "diamond.yaml")
    assert calls == [(tmp_path / "diamond.yaml", tmp_path, tmp_path / "source")]
    assert model._full_resolution == setup.full_resolution


def test_inference_import_scope_stubs_training_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid serving-only conflicts from DIAMOND's eager training imports."""
    names = assets_module._INFERENCE_IMPORT_STUBS
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with assets_module._inference_import_scope():
        assert all(sys.modules[name].__name__ == name for name in names)

    assert all(name not in sys.modules for name in names)


class _RecordingModel:
    def __init__(self):
        self.inputs = []
        self.error = None
        self.world_id = None
        self.index = 0

    def generate(self, input):
        self.inputs.append(input)
        if self.error is not None:
            raise self.error
        fresh = input.world_id != self.world_id
        self.world_id = input.world_id
        self.index = 0 if fresh else self.index + 1
        return model_module.DiamondResult(
            np.zeros((2, 2, 3), dtype=np.uint8),
            input.world_id,
            self.index,
            False,
            fresh,
            not fresh,
        )

    def reset(self):
        self.world_id = None


def test_refused_session_never_reaches_model():
    app = Diamond()
    app.state = DiamondState()
    app._engine = engine = _RecordingModel()
    with pytest.raises(ApplicationError, match="active"):
        _step(app)
    assert not engine.inputs
    app._start_session()
    app._end_session()
    with pytest.raises(ApplicationError, match="active"):
        _step(app)
    assert not engine.inputs


def test_hook_input_only_forwarding_and_single_anchor():
    app = _ready_model()
    app._engine = engine = _RecordingModel()
    for index in range(10):
        app.state._pressed_keys = frozenset({"w" if index % 2 else "a"})
        app.state._delta_x = float(index)
        input = asyncio.run(app.process_input())
        with pytest.raises(FrozenInstanceError):
            input.world_id = 999
        state, app.state = app.state, None
        result = app.generate(input)
        app.state = state
        asyncio.run(app.process_output(StepOutcome(result=result)))
        assert result.index == index
    assert engine.inputs[0].anchor is not None
    assert all(input.anchor is None for input in engine.inputs[1:])
    assert [input.delta_x for input in engine.inputs] == list(range(10))


def test_generate_failure_is_not_acknowledged():
    app = _ready_model()
    app._engine = engine = _RecordingModel()
    engine.error = RuntimeError("failed inference")
    app.state._delta_x = 12
    input = asyncio.run(app.process_input())
    with pytest.raises(RuntimeError, match="failed inference") as raised:
        app.generate(input)
    with pytest.raises(RuntimeError, match="failed inference"):
        asyncio.run(app.process_output(StepOutcome(error=raised.value)))
    assert app._applied_world_id is None
    assert app.state._delta_x == 12
    assert asyncio.run(app.process_input()).anchor is not None
    assert engine.index == 0


def test_native_terminal_requests_next_world_in_application(monkeypatch):
    app = _ready_model()
    _stub_video(monkeypatch, [])
    _step(app)
    engine = app._engine
    engine._world.step = lambda action: (
        np.zeros((1, 3, 2, 2)),
        _Scalar(False),
        _Scalar(True),
        _Scalar(False),
        {},
    )
    input = asyncio.run(app.process_input())
    result = app.generate(input)
    assert result.terminal and result.index == 1
    with pytest.raises(model_module.WorldComplete):
        engine.generate(input)
    asyncio.run(app.process_output(StepOutcome(result=result)))
    fresh = asyncio.run(app.process_input())
    assert fresh.world_id == input.world_id + 1 and fresh.anchor is not None
    assert engine._world.reset_count == 1
    app.generate(fresh)
    assert engine._world.reset_count == 2


def test_model_guards_and_failure_do_not_advance_index(monkeypatch):
    app = _ready_model()
    _stub_video(monkeypatch, [])
    input = asyncio.run(app.process_input())
    with pytest.raises(model_module.NoAnchor):
        app._engine.generate(replace(input, anchor=None))
    _step(app)

    def fail(action):
        raise RuntimeError("GPU failed")

    app._engine._world.step = fail
    with pytest.raises(RuntimeError, match="GPU failed"):
        _step(app)
    assert app._engine._index == 0
    assert app._engine._replay_step == 0


def test_reset_random_and_controller_use_new_ids(monkeypatch):
    app = _ready_model()
    app._engine = _RecordingModel()

    async def send(message):
        pass

    app.send = send
    _step(app)
    previous = app._world_id
    asyncio.run(app.reset())
    assert app._world_id == previous + 1
    assert asyncio.run(app.process_input()).anchor == model_module.DiamondAnchor()
    app._spawn_dirs = (Path("spawn0"), Path("spawn1"))
    asyncio.run(app.random_scene())
    assert app._pending_scene.scene in app._spawn_dirs
    app._pending_scene = model_module.DiamondAnchor(
        full_res=np.zeros((3, 2, 2), dtype=np.uint8)
    )
    asyncio.run(app.set_controller("replay"))
    assert app._pending_scene is None
    assert app.state.controller == "replay"


def test_model_and_assets_import_without_runtime():
    import subprocess

    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('reactor_runtime'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import diamond_model, diamond_assets
assert diamond_model.DiamondModel()._world is None
"""
    subprocess.run([sys.executable, "-c", script], cwd=EXAMPLE_DIR, check=True)


def test_replay_consumes_last_condition_action_then_recorded_actions(monkeypatch):
    app = _ready_model()
    _stub_video(monkeypatch, [])
    _step(app)

    class Actions:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            value = self.values[index[-1] if isinstance(index, tuple) else index]
            return SimpleNamespace(clone=lambda: value)

        def size(self, dimension):
            return len(self.values)

    engine = app._engine
    engine._world.act_buffer = Actions(["conditioning"])
    engine._world.next_act = Actions(["recorded0", "recorded1"])
    input = replace(asyncio.run(app.process_input()), controller="replay")
    results = [engine.generate(input) for _ in range(3)]
    assert engine._world.actions == ["conditioning", "recorded0", "recorded1"]
    assert [result.terminal for result in results] == [False, False, True]
    assert [result.index for result in results] == [1, 2, 3]
    assert all(result.clear_controls for result in results)
    with pytest.raises(model_module.WorldComplete):
        engine.generate(input)


def test_initial_session_frame_retains_controls_but_requested_reset_clears(monkeypatch):
    app = _ready_model()
    _stub_video(monkeypatch, [])
    app.state._pressed_keys = frozenset({"w"})
    _step(app)
    assert app.state._pressed_keys == frozenset({"w"})
    app._queue_scene_reset()
    app.state._pressed_keys = frozenset({"a"})
    _step(app)
    assert not app.state._pressed_keys
