import asyncio
import json
import time

import numpy as np
from reactor_robotics.session import ReactorSession

TRACKS = ("top", "wrist_left", "wrist_right")
# 25 rows per arm in the rig world frame: [x, y, z, qw, qx, qy, qz] and gripper (0 open).
STATE = {
    "left": {"ee_pose": [[0.3, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0]] * 25, "grip": [0.0] * 25},
    "right": {"ee_pose": [[0.3, -0.2, 0.25, 1.0, 0.0, 0.0, 0.0]] * 25, "grip": [0.0] * 25},
}
FIELDS = ("left_poses", "left_grips", "right_poses", "right_grips")


async def main():
    session = ReactorSession("reactor/praxis-1-yam", fps=15, frame_size=(260, 416))
    try:
        await session.connect(TRACKS, subscribe=("action_prediction",))
        # Training-caption style: clauses joined with "then", no closing period.
        await session.send(
            "set_task",
            {"task": "Pick up the pen from the center of the table, then place the pen in the bag"},
        )
        # Synthetic 16:10 fixtures: never execute their predicted actions on hardware.
        rng = np.random.default_rng(7)
        session.set_frames(
            {name: rng.integers(0, 256, (260, 416, 3), dtype=np.uint8) for name in TRACKS}
        )
        # The first chunk needs 25 frames of each camera (1.6 s at 15 fps).
        await asyncio.sleep(2.0)
        state = {"robot_state_json": json.dumps(STATE, allow_nan=False)}
        started = time.perf_counter()
        await session.send("set_robot_state_json", state)
        reply = await session.next_message("action_prediction", timeout_s=90)
        actions = {name: np.asarray(reply[name], dtype=np.float64) for name in FIELDS}
        assert int(reply["step"]) == 0, reply["step"]
        assert actions["left_poses"].shape == actions["right_poses"].shape == (21, 12)
        assert actions["left_grips"].shape == actions["right_grips"].shape == (21,)
        assert all(np.isfinite(a).all() for a in actions.values())
        print("left/right poses:", actions["left_poses"].shape)
        print("first left pose:", actions["left_poses"][0])
        print("first right pose:", actions["right_poses"][0])
        print("request-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))

        # Echo the executed step; the next chunk has the next step and 24 rows.
        await session.send("set_robot_state_json", state)
        await session.send("set_executed_step", {"executed_step": int(reply["step"])})
        reply = await session.next_message("action_prediction", timeout_s=90)
        assert int(reply["step"]) == 1, reply["step"]
        assert np.asarray(reply["left_poses"]).shape == (24, 12)
        print("next step:", int(reply["step"]), "rows:", len(reply["left_poses"]))
    finally:
        await session.close()


asyncio.run(main())
