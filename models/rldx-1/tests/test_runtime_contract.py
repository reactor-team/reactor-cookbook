# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests that the port still matches the standalone reactor-runtime's contract.

These are the checks the 2.x -> 3.x move could silently break: the authoring
surface the model imports, the declared inbound tracks (an ``Input`` subclass is
no longer a dataclass, so ``__tracks__`` is the source of truth), the ``load()``
signature the runner calls with a config *path*, and the client-facing schema —
same commands, messages, and tracks as the 2.x image served.

Skipped when reactor-runtime isn't installed, so the pure-seam tests in
test_model_schema.py still run on a bare checkout.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("reactor_runtime", reason="reactor-runtime is not installed")

import pipeline  # noqa: E402
from model_types import (  # noqa: E402
    ActionPrediction,
    CommandError,
    ModelSchema,
    RLDXInput,
    RLDXState,
)
from robot_state import FrameStateTags  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_declared_input_tracks():
    # build_schema reads these to reject a checkpoint whose camera views don't
    # match the tracks this port declares.
    assert tuple(RLDXInput.__tracks__) == ("left_view", "right_view", "wrist_view")


def test_load_takes_a_config_path():
    params = list(inspect.signature(pipeline.RLDXPipeline.load).parameters)
    assert params == ["self", "config_path"]


def test_read_config_parses_the_repo_config():
    config = pipeline.read_config(REPO_ROOT / "config.yml")
    assert config["control_hz"] == 20
    assert config["state_fallback"] == "hold_last"
    # No config declared at all is not an error: load() falls back to defaults.
    assert pipeline.read_config(None) == {}


def test_pipeline_instantiates_without_weights():
    # Constructing the model resolves its state class, auto-setters, and command
    # surface — everything but load(), which needs a GPU and a checkpoint.
    assert isinstance(pipeline.RLDXPipeline(), pipeline.ReactorPipeline)


def test_policy_memory_resets_once_per_session_not_per_connection():
    model = pipeline.RLDXPipeline()
    resets = 0

    def count_reset():
        nonlocal resets
        resets += 1

    model._reset_memory = count_reset
    model._schema = {}
    model.send = AsyncMock()

    assert getattr(model.on_session_started, "__reactor_session_started__", False)
    assert getattr(model.on_connect, "__reactor_connected__", False)

    model.on_session_started()
    asyncio.run(model.on_connect())

    assert resets == 1
    assert model._schema_pending is True


def test_messages_carry_their_chunk_fields():
    chunk = ActionPrediction(
        end_effector_position=[[0.0, 0.0, 0.0]],
        gripper_close=[[0.0]],
        step=7,
    )
    assert chunk.end_effector_position == [[0.0, 0.0, 0.0]]
    assert chunk.step == 7
    # Unset fields default to None rather than failing construction.
    assert chunk.base_motion is None
    assert CommandError(command="state", reason="missing").command == "state"
    assert ModelSchema(views=["left_view"]).views == ["left_view"]


def test_action_prediction_echoes_the_source_stamp_and_view_skew():
    chunk = ActionPrediction(source_capture_us=1_700_000_000_000_000, source_seq=41, view_skew_us=640)
    assert chunk.source_capture_us == 1_700_000_000_000_000
    assert chunk.source_seq == 41
    assert chunk.view_skew_us == 640
    # A client that embeds no stamp gets nulls, not zeros it might trust.
    assert ActionPrediction(step=0).source_capture_us is None


def _pipeline_resolving_state(fallback="error"):
    """A pipeline wired for _resolve_state() only — no load(), no checkpoint."""
    p = pipeline.RLDXPipeline()
    # The runtime binds `state` at connect; stand in for it with the same class.
    p.state = RLDXState()
    p._state_dims = {"gripper_qpos": 2}
    p._state_fallback = fallback
    p._last_state = None
    p._state_degraded = False
    p._frame_tags = FrameStateTags()
    return p


def _grip(state):
    return float(state["state.gripper_qpos"][0, 0, 0])


def test_state_json_field_still_works_for_a_client_that_cannot_tag_frames():
    p = _pipeline_resolving_state()
    p.state.state_json = json.dumps({"gripper_qpos": [1.0, 1.0]})
    state, degraded = p._resolve_state()
    assert degraded is None and _grip(state) == 1.0


def test_frame_metadata_is_preferred_over_the_field():
    p = _pipeline_resolving_state()
    p.state.state_json = json.dumps({"gripper_qpos": [1.0, 1.0]})
    p._frame_tags.offer(json.dumps({"gripper_qpos": [2.0, 2.0]}).encode())
    state, degraded = p._resolve_state()
    assert degraded is None and _grip(state) == 2.0


def test_unusable_tag_falls_back_to_the_declared_policy():
    p = _pipeline_resolving_state(fallback="hold_last")
    p._frame_tags.offer(json.dumps({"gripper_qpos": [3.0, 3.0]}).encode())
    assert _grip(p._resolve_state()[0]) == 3.0
    # A garbage tag doesn't silently reuse the good one: the fallback does, and
    # it says so, so the client gets a command_error.
    p._frame_tags.offer(b"{garbage")
    state, degraded = p._resolve_state()
    assert _grip(state) == 3.0
    assert degraded is not None and "holding last-known state" in degraded


def test_no_state_at_all_skips_inference():
    p = _pipeline_resolving_state()
    state, degraded = p._resolve_state()
    assert state is None
    assert "no usable frame tag, no state_json" in degraded


def test_schema_announces_the_state_carrier():
    assert ModelSchema(state_source="frame_metadata").state_source == "frame_metadata"
    assert ModelSchema(state_tag_keys=["capture_us", "seq"]).state_tag_keys == [
        "capture_us",
        "seq",
    ]


def test_rendered_schema_is_the_2x_client_contract():
    from reactor_runtime.schema import render

    doc = render(REPO_ROOT)
    assert sorted(doc["paths"]) == [
        "/events/get_schema",
        "/events/reset",
        "/events/set_state_json",
        "/events/set_task_description",
    ]
    assert sorted(doc["webhooks"]) == [
        "action_prediction",
        "command_error",
        "model_schema",
    ]
    tracks = doc["x-reactor"]["tracks"]
    assert [t["name"] for t in tracks] == ["left_view", "right_view", "wrist_view"]
    assert {t["direction"] for t in tracks} == {"in"}

    def field_schema(command: str, field: str):
        return doc["paths"][f"/events/{command}"]["post"]["requestBody"][
            "content"
        ]["application/json"]["schema"]["properties"][field]

    # Runtime 3.x moderation is opt-in. Preserve the high-rate robot state path,
    # while explicitly opting the client-authored free-text instruction in.
    state_json = field_schema("set_state_json", "state_json")
    assert state_json["x-reactor-moderate"] is False
    assert (
        field_schema("set_task_description", "task_description")[
            "x-reactor-moderate"
        ]
        is True
    )
