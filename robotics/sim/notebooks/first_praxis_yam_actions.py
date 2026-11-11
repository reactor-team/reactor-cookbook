import asyncio
import time

import numpy as np
from reactor_robotics.praxis import (
    CHUNK_ROWS,
    FIRST_CHUNK_ROWS,
    HISTORY_ROWS,
    POSE_WIDTH,
    YAM_FRAME_HW,
    YAM_MODEL,
    YAM_TRACKS,
    check_chunk,
    encode_yam_state,
)
from reactor_robotics.session import ReactorSession

FIELDS = {"left_poses": POSE_WIDTH, "left_grips": 0, "right_poses": POSE_WIDTH, "right_grips": 0}
# Sample measured state in the rig world frame; not a command to move a robot.
STATE = encode_yam_state(
    [[0.3, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0]] * HISTORY_ROWS,
    [0.0] * HISTORY_ROWS,
    [[0.3, -0.2, 0.25, 1.0, 0.0, 0.0, 0.0]] * HISTORY_ROWS,
    [0.0] * HISTORY_ROWS,
)
# Training-caption style: clauses joined with "then", no closing period.
TASK = "Pick up the pen from the center of the table, then place the pen in the bag"


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
    session = ReactorSession(YAM_MODEL, fps=15, frame_size=YAM_FRAME_HW)
    try:
        await session.connect(
            YAM_TRACKS, subscribe=("action_prediction", "prediction_failed")
        )
        # Synthetic 16:10 frames: never execute their predicted actions on hardware.
        rng = np.random.default_rng(7)
        session.set_frames(
            {
                name: rng.integers(0, 256, (*YAM_FRAME_HW, 3), dtype=np.uint8)
                for name in YAM_TRACKS
            }
        )
        await session.send("set_task", {"task": TASK})

        # The first chunk waits for 25 frames of each camera (about 2 s).
        started = time.perf_counter()
        first = await next_chunk(session)
        check_chunk(first, FIELDS, step=0, rows=FIRST_CHUNK_ROWS)
        print("step 0 left/right poses:", np.asarray(first["left_poses"]).shape)
        print("first left xyz:", np.round(first["left_poses"][0][:3], 3).tolist())
        print("first right xyz:", np.round(first["right_poses"][0][:3], 3).tolist())
        print("first chunk seconds:", round(time.perf_counter() - started, 2))

        # Echo the executed step to get the next chunk.
        started = time.perf_counter()
        await session.send("set_executed_step", {"executed_step": int(first["step"])})
        second = await next_chunk(session)
        check_chunk(second, FIELDS, step=1, rows=CHUNK_ROWS)
        print("step 1 left/right poses:", np.asarray(second["left_poses"]).shape)
        print("echo-to-chunk seconds:", round(time.perf_counter() - started, 2))
    finally:
        await session.close()


asyncio.run(main())
