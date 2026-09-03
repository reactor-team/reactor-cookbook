# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests for the checkpoint-derived schema handshake (REA-4318).

``build_schema`` is the pure seam between the values ``load()`` resolves from
the checkpoint (RLWRLD's source of truth) and the ``model_schema`` message
sent to the client at session start. No torch or checkpoint needed.
"""

from __future__ import annotations

import json

import pytest

from rldx1_schema import build_schema

VIEWS = ["left_view", "right_view", "wrist_view"]


def build(**overrides):
    kwargs = dict(
        views=VIEWS,
        declared_views=VIEWS,
        video_delta_indices=[-6, -4, -2, 0],
        state_dims={"end_effector_position_relative": 3, "gripper_qpos": 2},
        action_dims={"end_effector_position": 3},
        action_order=["end_effector_position"],
        action_horizon=16,
        exec_horizon=16,
        rtc_delay=0,
        inference_trigger="streaming",
        control_hz=20.0,
        resolution=(256, 256),
        rtc_mode="off",
        embodiment="general_embodiment",
        state_fallback="hold_last",
        state_source="frame_metadata",
        state_tag_keys=["capture_us", "seq"],
    )
    kwargs.update(overrides)
    return build_schema(**kwargs)


def test_schema_reflects_checkpoint_values():
    schema = build()
    assert schema["views"] == VIEWS
    assert schema["video_delta_indices"] == [-6, -4, -2, 0]
    assert schema["resolution"] == [256, 256]
    assert schema["control_hz"] == 20.0
    assert schema["state_dims"] == {
        "end_effector_position_relative": 3,
        "gripper_qpos": 2,
    }
    assert schema["action_dims"] == {"end_effector_position": 3}
    assert schema["action_order"] == ["end_effector_position"]
    assert schema["action_horizon"] == 16
    assert schema["exec_horizon"] == 16
    assert schema["rtc_delay"] == 0
    assert schema["inference_trigger"] == "streaming"
    assert schema["rtc_mode"] == "off"
    assert schema["dtype"] == "float32"
    assert schema["embodiment"] == "general_embodiment"
    assert schema["state_fallback"] == "hold_last"
    # Where the client should put robot state: tagged onto each video frame,
    # with set_state_json still accepted as a fallback (see robot_state.py).
    assert schema["state_source"] == "frame_metadata"
    # The optional extras a tagging client may embed in that same state JSON.
    assert schema["state_tag_keys"] == ["capture_us", "seq"]


def test_single_frame_checkpoint_window():
    schema = build(video_delta_indices=[0])
    assert schema["video_delta_indices"] == [0]


def test_mismatched_video_keys_raise():
    with pytest.raises(ValueError, match="image_0"):
        build(views=["image_0", "image_1"])


def test_view_order_follows_checkpoint_not_declaration():
    reordered = list(reversed(VIEWS))
    schema = build(views=reordered)
    assert schema["views"] == reordered


def test_numpy_scalars_normalized_to_json_types():
    import numpy as np

    schema = build(
        video_delta_indices=[np.int64(-2), np.int64(0)],
        state_dims={"gripper_qpos": np.int64(2)},
        action_horizon=np.int64(16),
        control_hz=np.float64(20.0),
    )
    json.dumps(schema)


def test_schema_is_json_serializable():
    json.dumps(build())
