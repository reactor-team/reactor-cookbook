# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests for the robot-state seam: JSON -> model state vectors, and the rule
that picks one tag out of three tagged video streams.

``robot_state`` is pure (numpy only), so these run on a bare checkout without
torch, a checkpoint, or reactor-runtime.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_state import FrameStateTags, parse_state, zero_state

DIMS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}


def state(x: float = 0.0) -> dict:
    return {
        "end_effector_position_relative": [x, 0.0, 0.0],
        "end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.0, 0.0],
        "base_position": [0.0, 0.0, 0.0],
        "base_rotation": [0.0, 0.0, 0.0, 1.0],
    }


def tag(x: float = 0.0, **stamp: int) -> bytes:
    # Compact separators, as the client sends them; `capture_us` / `seq` ride in
    # the same object as the state vectors.
    payload = {**state(x), **stamp}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def pos_x(parsed) -> float:
    return float(parsed["state.end_effector_position_relative"][0, 0, 0])


# ── parse_state ───────────────────────────────────────────────────────────────

def test_parses_into_batched_float32_vectors():
    parsed = parse_state(json.dumps(state(0.5)), DIMS)
    assert set(parsed) == {f"state.{k}" for k in DIMS}
    arr = parsed["state.end_effector_position_relative"]
    assert arr.shape == (1, 1, 3) and arr.dtype == np.float32
    assert arr[0, 0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "payload",
    [
        "",                                   # nothing sent
        "not json",                           # not JSON
        "[1, 2, 3]",                          # JSON, but not an object
        json.dumps({"gripper_qpos": [0.0]}),  # missing keys
    ],
)
def test_unusable_payloads_are_none_not_zeros(payload):
    assert parse_state(payload, DIMS) is None


def test_wrong_length_vector_is_rejected():
    bad = state()
    bad["gripper_qpos"] = [0.0, 0.0, 0.0]
    assert parse_state(json.dumps(bad), DIMS) is None


def test_non_numeric_vector_is_rejected():
    bad = state()
    bad["base_position"] = ["a", "b", "c"]
    assert parse_state(json.dumps(bad), DIMS) is None


def test_zero_state_matches_the_declared_dims():
    zeros = zero_state(DIMS)
    assert {k: v.shape for k, v in zeros.items()} == {
        f"state.{k}": (1, 1, d) for k, d in DIMS.items()
    }
    assert all(not v.any() for v in zeros.values())


# ── FrameStateTags: one state out of three streams ───────────────────────────

def test_no_tag_seen_yet_is_none():
    assert FrameStateTags().parse(DIMS) is None


def test_last_tag_offered_wins():
    tags = FrameStateTags()
    for x in (0.1, 0.2, 0.3):  # left, right, wrist of one poll
        tags.offer(tag(x))
    parsed = tags.parse(DIMS)
    assert parsed["state.end_effector_position_relative"][0, 0, 0] == pytest.approx(0.3)


def test_untagged_frame_leaves_the_previous_tag_standing():
    tags = FrameStateTags()
    tags.offer(tag(0.7))
    tags.offer(None)  # a view whose frames carry no metadata
    tags.offer(b"")   # an empty trailer is "nothing attached", not "no state"
    parsed = tags.parse(DIMS)
    assert parsed["state.end_effector_position_relative"][0, 0, 0] == pytest.approx(0.7)


def test_repeated_bytes_are_parsed_once():
    tags = FrameStateTags()
    tags.offer(tag(0.4))
    first = tags.parse(DIMS)
    tags.offer(tag(0.4))  # same bytes, next view / next tick
    assert tags.parse(DIMS) is first  # identity: cached, not re-parsed
    tags.offer(tag(0.9))  # different bytes -> fresh parse
    assert tags.parse(DIMS) is not first


def test_unparseable_tag_reports_none_rather_than_the_older_state():
    tags = FrameStateTags()
    tags.offer(tag(0.2))
    assert tags.parse(DIMS) is not None
    tags.offer(b"{garbage")
    # The freshest tag is unusable: say so, and let the pipeline's state_fallback
    # decide whether to hold the last state, zero-fill, or skip the tick.
    assert tags.parse(DIMS) is None


def test_non_utf8_tag_is_rejected_without_raising():
    tags = FrameStateTags()
    tags.offer(b"\xff\xfe\x00")
    assert tags.parse(DIMS) is None


def test_clear_forgets_the_episode():
    tags = FrameStateTags()
    tags.offer(tag(0.3))
    tags.clear()
    assert tags.parse(DIMS) is None


# ── FrameStateTags: stamped tags order themselves ────────────────────────────

def test_reserved_stamp_keys_do_not_disturb_the_state_parse():
    parsed = parse_state(json.dumps({**state(0.5), "capture_us": 123, "seq": 7}), DIMS)
    assert set(parsed) == {f"state.{k}" for k in DIMS}
    assert pos_x(parsed) == pytest.approx(0.5)


def test_embedded_seq_beats_arrival_order():
    tags = FrameStateTags()
    tags.offer(tag(0.2, seq=2))
    tags.offer(tag(0.1, seq=1))  # a view of the previous tick, arriving late
    assert pos_x(tags.parse(DIMS)) == pytest.approx(0.2)


def test_embedded_capture_us_orders_tags_without_a_seq():
    tags = FrameStateTags()
    tags.offer(tag(0.2, capture_us=2_000))
    tags.offer(tag(0.1, capture_us=1_000))
    assert pos_x(tags.parse(DIMS)) == pytest.approx(0.2)


def test_wire_capture_stamp_orders_tags_that_embed_nothing():
    tags = FrameStateTags()
    tags.offer(tag(0.2), capture_time_us=2_000)
    tags.offer(tag(0.1), capture_time_us=1_000)
    assert pos_x(tags.parse(DIMS)) == pytest.approx(0.2)


def test_an_embedded_stamp_outranks_a_wire_only_tag():
    tags = FrameStateTags()
    tags.offer(tag(0.2, seq=1))
    # A later frame with a much larger wire stamp, but nothing embedded: the
    # client's own ordering is not overruled by the sender engine's clock.
    tags.offer(tag(0.1), capture_time_us=9_000_000)
    assert pos_x(tags.parse(DIMS)) == pytest.approx(0.2)


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("key", ["capture_us", "seq"])
def test_non_finite_embedded_stamp_is_ignored_without_raising(key, stamp):
    tags = FrameStateTags()
    tags.offer(tag(0.2, **{key: stamp}), capture_time_us=2_000)
    assert pos_x(tags.parse(DIMS)) == pytest.approx(0.2)
    assert tags.stamp == (None, None)


def test_stamp_reports_the_freshest_tags_client_values():
    tags = FrameStateTags()
    assert FrameStateTags().stamp == (None, None)
    tags.offer(tag(0.1))
    assert tags.stamp == (None, None)  # an unstamped client asks for no echo
    tags.offer(tag(0.2, capture_us=1_700_000_000_000_000, seq=42))
    assert tags.stamp == (1_700_000_000_000_000, 42)


def test_clear_forgets_the_stamps():
    tags = FrameStateTags()
    tags.offer(tag(0.2, seq=5))
    tags.clear()
    assert tags.stamp == (None, None)
    # The cleared ordering state does not let the old tick outrank the new one.
    tags.offer(tag(0.1, seq=1))
    assert pos_x(tags.parse(DIMS)) == pytest.approx(0.1)
