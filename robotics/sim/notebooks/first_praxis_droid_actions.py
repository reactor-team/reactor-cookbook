import asyncio
import time

import numpy as np
from reactor_robotics.praxis import (
    CHUNK_ROWS,
    DROID_FRAME_HW,
    DROID_MODEL,
    DROID_TRACKS,
    FIRST_CHUNK_ROWS,
    HISTORY_ROWS,
    POSE_WIDTH,
    check_chunk,
    encode_droid_state,
)
from reactor_robotics.session import ReactorSession

FIELDS = {"poses": POSE_WIDTH, "grips": 0}
# Sample measured state for a transport check; not a command to move a robot.
STATE = encode_droid_state(
    [[0.45, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]] * HISTORY_ROWS, [0.0] * HISTORY_ROWS
)


async def next_chunk(session, timeout_s=30.0):
    """Keep sending state until a chunk arrives; fail fast on prediction_failed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await session.send("set_robot_state_json", {"robot_state_json": STATE})
        for failure in session.drain("prediction_failed"):
            raise RuntimeError(f"prediction_failed: {failure}")
        try:
            return await session.next_message("action_prediction", timeout_s=0.5)
        except asyncio.TimeoutError:
            continue
    raise TimeoutError("no action_prediction; send get_status to see waiting_for")


async def main():
    session = ReactorSession(DROID_MODEL, fps=15, frame_size=DROID_FRAME_HW)
    try:
        await session.connect(
            DROID_TRACKS, subscribe=("action_prediction", "prediction_failed")
        )
        # Synthetic frames: never execute their predicted actions on hardware.
        rng = np.random.default_rng(7)
        session.set_frames(
            {
                name: rng.integers(0, 256, (*DROID_FRAME_HW, 3), dtype=np.uint8)
                for name in DROID_TRACKS
            }
        )
        await session.send("set_task", {"task": "Put the red block in the bowl."})
        await session.send("set_pose_frame", {"pose_frame": "fr3_hand"})

        # The first chunk waits for 25 frames of each camera (about 2 s).
        started = time.perf_counter()
        first = await next_chunk(session)
        check_chunk(first, FIELDS, step=0, rows=FIRST_CHUNK_ROWS)
        print("step 0 poses:", np.asarray(first["poses"]).shape)
        print("first pose:", np.round(first["poses"][0], 3).tolist())
        print("first grip:", round(float(first["grips"][0]), 3))
        print("first chunk seconds:", round(time.perf_counter() - started, 2))

        # Echo the executed step to get the next chunk.
        started = time.perf_counter()
        await session.send("set_executed_step", {"executed_step": int(first["step"])})
        second = await next_chunk(session)
        check_chunk(second, FIELDS, step=1, rows=CHUNK_ROWS)
        print("step 1 poses:", np.asarray(second["poses"]).shape)
        print("echo-to-chunk seconds:", round(time.perf_counter() - started, 2))
    finally:
        await session.close()


asyncio.run(main())
