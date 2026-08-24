# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests for the pipeline's pure frame-window helpers.

``_align_by_capture`` decides which frame of each view goes into the temporal
window, and it never touches pixels — sentinel objects stand in for frames here,
so a wrong pick is visible by identity. Importing ``pipeline`` needs
reactor-runtime (but no torch, checkpoint, or GPU).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("reactor_runtime", reason="reactor-runtime is not installed")

from pipeline import _align_by_capture, _FrameCandidate  # noqa: E402
from robot_state import FrameStateTags  # noqa: E402

VIEWS = ("left_view", "right_view", "wrist_view")


class Frame:
    """Stand-in for a resized video frame; compares by identity, prints by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name}>"


def candidate(
    stamp: int | None,
    frame: Frame,
    metadata: bytes | None = None,
) -> _FrameCandidate:
    return _FrameCandidate(capture_time_us=stamp, data=frame, metadata=metadata)


def state_tag(seq: int) -> bytes:
    return json.dumps(
        {
            "gripper_qpos": [float(seq), float(seq)],
            "capture_us": seq * 1_000,
            "seq": seq,
        }
    ).encode()


def test_views_align_on_the_newest_instant_they_all_cover():
    left = Frame("left")
    right = Frame("right")
    wrist = Frame("wrist")
    commit = _align_by_capture(
        {
            "left_view": [candidate(1_000, Frame("left-old")), candidate(2_000, left)],
            "right_view": [candidate(1_100, Frame("right-old")), candidate(2_050, right)],
            "wrist_view": [candidate(1_050, Frame("wrist-old")), candidate(1_980, wrist)],
        }
    )
    assert commit.frames == {"left_view": left, "right_view": right, "wrist_view": wrist}
    assert commit.view_skew_us == 70  # 2_050 - 1_980


def test_a_lagging_view_drags_the_reference_back():
    # left and right are publishing ahead; wrist's newest frame is a tick old, so
    # the window is built from the tick wrist can actually cover.
    left = Frame("left@2000")
    right = Frame("right@2010")
    wrist = Frame("wrist@2005")
    commit = _align_by_capture(
        {
            "left_view": [candidate(2_000, left), candidate(3_000, Frame("left@3000"))],
            "right_view": [candidate(2_010, right), candidate(3_020, Frame("right@3020"))],
            "wrist_view": [candidate(1_000, Frame("wrist@1000")), candidate(2_005, wrist)],
        }
    )
    assert commit.frames == {"left_view": left, "right_view": right, "wrist_view": wrist}
    assert commit.view_skew_us == 10  # 2_010 - 2_000


def test_a_lagging_view_keeps_state_on_the_aligned_tick():
    # Tick 3 has arrived on the faster views, but wrist only covers tick 2. The
    # policy must receive tick-2 state with the tick-2 image commit, and the echo
    # must name tick 2 as well.
    commit = _align_by_capture(
        {
            "left_view": [
                candidate(2_000, Frame("left@2"), state_tag(2)),
                candidate(3_000, Frame("left@3"), state_tag(3)),
            ],
            "right_view": [
                candidate(2_000, Frame("right@2"), state_tag(2)),
                candidate(3_000, Frame("right@3"), state_tag(3)),
            ],
            "wrist_view": [candidate(2_000, Frame("wrist@2"), state_tag(2))],
        }
    )
    tags = FrameStateTags()
    for selected in commit.state_candidates:
        tags.offer(selected.metadata, capture_time_us=selected.capture_time_us)

    parsed = tags.parse({"gripper_qpos": 2})
    assert float(parsed["state.gripper_qpos"][0, 0, 0]) == 2.0
    assert tags.stamp == (2_000, 2)


def test_an_unstamped_newest_frame_falls_back_to_newest_per_view():
    newest = {v: Frame(f"{v}-newest") for v in VIEWS}
    commit = _align_by_capture(
        {
            "left_view": [
                candidate(2_000, Frame("left-old")),
                candidate(3_000, newest["left_view"]),
            ],
            "right_view": [
                candidate(2_010, Frame("right-old")),
                candidate(3_010, newest["right_view"]),
            ],
            "wrist_view": [
                candidate(2_005, Frame("wrist-old")),
                candidate(None, newest["wrist_view"]),
            ],
        }
    )
    assert commit.frames == newest
    assert commit.view_skew_us is None


def test_single_entry_views_align_to_themselves():
    only = {v: Frame(v) for v in VIEWS}
    commit = _align_by_capture(
        {
            "left_view": [candidate(2_000, only["left_view"])],
            "right_view": [candidate(2_400, only["right_view"])],
            "wrist_view": [candidate(2_100, only["wrist_view"])],
        }
    )
    assert commit.frames == only
    assert commit.view_skew_us == 400
