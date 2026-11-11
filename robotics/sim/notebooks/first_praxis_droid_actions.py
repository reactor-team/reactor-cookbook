import asyncio
import json
import time

import numpy as np
from reactor_robotics.session import ReactorSession

TRACKS = ("exterior", "wrist")
# 25 rows of measured state at 15 Hz: [x, y, z, qw, qx, qy, qz] and gripper (0 open).
STATE = {"ee_pose": [[0.45, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]] * 25, "grip": [0.0] * 25}


async def main():
    session = ReactorSession("reactor/praxis-1-droid", fps=15, frame_size=(288, 512))
    try:
        await session.connect(TRACKS, subscribe=("action_prediction",))
        await session.send("set_task", {"task": "Put the red block in the bowl."})
        await session.send("set_pose_frame", {"pose_frame": "fr3_hand"})
        # Synthetic fixtures: never execute their predicted actions on hardware.
        rng = np.random.default_rng(7)
        session.set_frames(
            {name: rng.integers(0, 256, (288, 512, 3), dtype=np.uint8) for name in TRACKS}
        )
        # The first chunk needs 25 frames of each camera (1.6 s at 15 fps).
        await asyncio.sleep(2.0)
        state = {"robot_state_json": json.dumps(STATE, allow_nan=False)}
        started = time.perf_counter()
        await session.send("set_robot_state_json", state)
        reply = await session.next_message("action_prediction", timeout_s=90)
        poses = np.asarray(reply["poses"], dtype=np.float64)
        grips = np.asarray(reply["grips"], dtype=np.float64)
        assert int(reply["step"]) == 0, reply["step"]
        assert poses.shape == (21, 12) and grips.shape == (21,), poses.shape
        assert np.isfinite(poses).all() and np.isfinite(grips).all()
        print("poses:", poses.shape, "grips:", grips.shape)
        print("first pose:", poses[0])
        print("request-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))

        # Echo the executed step; the next chunk has the next step and 24 rows.
        await session.send("set_robot_state_json", state)
        await session.send("set_executed_step", {"executed_step": int(reply["step"])})
        reply = await session.next_message("action_prediction", timeout_s=90)
        assert int(reply["step"]) == 1, reply["step"]
        assert np.asarray(reply["poses"]).shape == (24, 12)
        print("next step:", int(reply["step"]), "rows:", len(reply["poses"]))
    finally:
        await session.close()


asyncio.run(main())
