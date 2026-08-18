"""Focused contract and lifecycle tests for the DIAMOND adapter."""

from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import load_config

EXAMPLE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

pipeline_module = importlib.import_module("diamond")
support_module = importlib.import_module("diamond_support")
types_module = importlib.import_module("diamond_types")
Diamond = pipeline_module.Diamond
DiamondOutput = types_module.DiamondOutput
DiamondState = types_module.DiamondState
PreparedScene = types_module.PreparedScene
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
    model._agent = SimpleNamespace(device="test-device")
    model._world = _World()
    model._action_type = _Action
    model._encode_action = lambda action, *, device: action
    model._key_codes = {
        key: index for index, key in enumerate(types_module.KEYS, start=1)
    }
    return model


def _stub_video(monkeypatch: pytest.MonkeyPatch, observed: list[np.ndarray]) -> None:
    def convert(observation: np.ndarray) -> np.ndarray:
        observed.append(observation)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline_module, "to_video_frame", convert)


def test_contract_uses_session_hooks_and_documents_side_effects() -> None:
    """Expose session-scoped lifecycle hooks and a complete public schema."""
    contract = ModelContract.of(Diamond)

    assert contract.lifecycle.session_started is not None
    assert contract.lifecycle.session_ended is not None
    assert contract.lifecycle.connected is not None
    assert contract.lifecycle.disconnected is None
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
    model.state.paused = True
    model.state._pressed_keys = frozenset({"w", "space"})
    model.state._pressed_mouse_buttons = frozenset({"left"})
    client = _Client()

    asyncio.run(model._connected(client))

    assert client.messages == [
        StateUpdate(
            controller="human",
            paused=True,
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
            paused=False,
            pressed_keys=["w"],
            pressed_mouse_buttons=[],
        )
    ]


def test_playout_matches_upstream_game_timing() -> None:
    """Advance the world at its native rate without queuing stale controls."""
    assert Diamond.fps == 15
    assert Diamond.buffer_size == 1


def test_reconnect_preserves_the_session_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the shared world alive when the pipeline generator is recreated."""
    model = _ready_model()
    world = model._world
    _stub_video(monkeypatch, [])
    model._start_session()

    first_connection = model.inference()
    assert isinstance(next(first_connection), DiamondOutput)
    assert isinstance(next(first_connection), DiamondOutput)
    first_connection.close()

    model.state = DiamondState()
    second_connection = model.inference()
    assert isinstance(next(second_connection), DiamondOutput)
    assert world.reset_count == 1
    assert len(world.actions) == 2


def test_session_end_discards_a_queued_scene() -> None:
    """Prevent a queued scene from leaking into the next Reactor session."""
    model = _ready_model()
    world = model._world
    model._start_session()
    model._pending_scene = PreparedScene("low", "full", "actions", None)

    model._end_session()
    assert model._pending_scene is None
    assert model._initial_observation is None

    model.state = DiamondState()
    model._start_session()
    assert world.reset_count == 2
    assert model._initial_observation is not None


def test_queued_scene_is_emitted_before_the_first_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit a queued initial frame before consuming a client action."""
    model = _ready_model()
    world = model._world
    uploaded = np.full((1, 4, 3, 2, 2), 6.0)
    model._pending_scene = PreparedScene(
        obs=np.full((1, 4, 3, 1, 1), 5.0),
        obs_full_res=uploaded,
        act=world.act_buffer,
        next_act=None,
    )
    observed: list[np.ndarray] = []
    _stub_video(monkeypatch, observed)

    output = next(model.inference())

    assert isinstance(output, DiamondOutput)
    np.testing.assert_array_equal(observed, [uploaded[:, -1]])
    assert world.actions == []


def test_paused_scene_upload_emits_without_a_model_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Display a queued frame while paused without spending inference compute."""
    model = _ready_model()
    world = model._world
    model._paused = True
    model.state.paused = True
    uploaded = np.full((1, 4, 3, 2, 2), 8.0)
    model._pending_scene = PreparedScene(
        obs=np.full((1, 4, 3, 1, 1), 7.0),
        obs_full_res=uploaded,
        act=world.act_buffer,
        next_act=None,
    )
    observed: list[np.ndarray] = []
    _stub_video(monkeypatch, observed)

    inference = model.inference()

    assert isinstance(next(inference), DiamondOutput)
    np.testing.assert_array_equal(observed, [uploaded[:, -1]])
    assert next(inference) is None
    assert world.actions == []


def test_manifest_and_runtime_pin_match_the_package_entrypoint() -> None:
    """Keep the manifest entrypoint and public Runtime pin reproducible."""
    config = load_config(EXAMPLE_DIR / "reactor.yaml")

    assert config.model_ref == "diamond:Diamond"
    assert "reactor-runtime==3.1.2" in (EXAMPLE_DIR / "requirements.txt").read_text()


def test_model_download_uses_the_runtime_weights_mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist Hugging Face assets under Reactor's mounted weights root."""
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", str(tmp_path))

    assert pipeline_module._weights_cache_path() == (
        tmp_path / "diamond-csgo-world-model" / "huggingface"
    )


def test_inference_import_scope_stubs_training_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid serving-only conflicts from DIAMOND's eager training imports."""
    names = support_module._INFERENCE_IMPORT_STUBS
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with support_module._inference_import_scope():
        assert all(sys.modules[name].__name__ == name for name in names)

    assert all(name not in sys.modules for name in names)
