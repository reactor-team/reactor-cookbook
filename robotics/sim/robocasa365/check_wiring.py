# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Offline wiring check: no GPU, no network, no simulator.

Verifies the two things a live run cannot tell you it got wrong, because
both failure modes look exactly like a bad policy:

1. Each vendor camera maps to the model's track of the SAME view, in the
   model's declared order. A swapped mapping still connects, still predicts,
   and silently conditions the policy on cross-wired views.
2. Frames are pushed one-per-camera as a SET per history slot. The model
   pairs its three tracks frame-for-frame in arrival order, so a camera
   that skips a slot shifts its whole history against the other two.

    uv run python check_wiring.py
"""

from __future__ import annotations

import asyncio

import numpy as np

from robocasa365_sim import client as c

# The model's contract: xr1-robocasa365 declares these three video tracks,
# in this order (it is also the prompt-template order).
MODEL_TRACKS = ("left_agentview", "right_agentview", "wrist_view")

failures = []


def check(label: str, ok: bool) -> None:
    print(f"  {label:<66} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(label)


print("camera -> track mapping")
check(
    "TRACK_ORDER matches the model's declared tracks, in order",
    c.TRACK_ORDER == MODEL_TRACKS,
)
check(
    "every vendor camera maps to a distinct track (bijective)",
    sorted(c.TRACK_FOR_CAMERA.values()) == sorted(MODEL_TRACKS)
    and len(set(c.TRACK_FOR_CAMERA.values())) == len(c.CAMERA_ORDER),
)
check(
    "left camera -> left track, right -> right, eye-in-hand -> wrist",
    c.TRACK_FOR_CAMERA["video.robot0_agentview_left"] == "left_agentview"
    and c.TRACK_FOR_CAMERA["video.robot0_agentview_right"] == "right_agentview"
    and c.TRACK_FOR_CAMERA["video.robot0_eye_in_hand"] == "wrist_view",
)


class _FakeTrack:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.q = self  # client calls self._tracks[name].q.put_nowait(frame)

    def put_nowait(self, frame: np.ndarray) -> None:
        self.frames.append(frame)


print("per-slot push ordering")
# Drive _push_frames against fake tracks: no SDK, no connection.
fake = c.ReactorEvalClient.__new__(c.ReactorEvalClient)
fake._tracks = {name: _FakeTrack() for name in c.TRACK_ORDER}
history = {
    cam: [np.full((8, 8, 3), 10 * s + i, dtype=np.uint8) for s in range(4)]
    for i, cam in enumerate(c.CAMERA_ORDER)
}
asyncio.run(fake._push_frames(history))

check(
    "every track received exactly one frame per history slot",
    all(len(t.frames) == 4 for t in fake._tracks.values()),
)
check(
    "slot k of each track is the camera's own slot-k frame, unstacked",
    all(
        np.array_equal(
            fake._tracks[c.TRACK_FOR_CAMERA[cam]].frames[s], history[cam][s]
        )
        for cam in c.CAMERA_ORDER
        for s in range(4)
    ),
)
check(
    "frames keep the camera's native shape (no horizontal stacking)",
    all(t.frames[0].shape == (8, 8, 3) for t in fake._tracks.values()),
)

print("keepalive")
check(
    "ping interval is inside the runtime's 20 s silence watchdog",
    0 < c._PING_INTERVAL_S < 20,
)

print()
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s))")
    raise SystemExit(1)
print("RESULT: PASS")
