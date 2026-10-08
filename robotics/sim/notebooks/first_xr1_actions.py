import asyncio
import json
import time

import numpy as np
from reactor_robotics.session import ReactorSession

TRACKS = ("ego_view", "wrist_left_view", "wrist_right_view")


async def main():
    session = ReactorSession("reactor/xr1", fps=15, frame_size=(480, 640))
    try:
        await session.connect(TRACKS, subscribe=("action_prediction",))
        await session.send(
            "set_task_description", {"task_description": "place the cup on the table"}
        )
        frames = {
            name: np.full((480, 640, 3), 40 + 60 * i, dtype=np.uint8)
            for i, name in enumerate(TRACKS)
        }
        session.set_frames(frames)
        await asyncio.sleep(0.5)
        state = {
            "left_arm_joint": [0.0] * 7,
            "left_gripper_pos": [0.0],
            "right_arm_joint": [0.0] * 7,
            "right_gripper_pos": [0.0],
        }
        started = time.perf_counter()
        await session.send(
            "set_proprio_json", {"proprio_json": json.dumps(state, allow_nan=False)}
        )
        reply = await session.next_message("action_prediction", timeout_s=90)
        actions = np.asarray(reply["action"], dtype=np.float64)
        assert actions.shape == (30, 60) and np.isfinite(actions).all()
        assert reply["step"] == 0 and reply["prefix_rows"] == 0
        assert np.all(actions[:, [7, 15, *range(20, 60)]] == 0)
        print("actions:", actions.shape)
        print("first left EE delta:", actions[0, :6])
        print("request-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))
    finally:
        await session.close()


asyncio.run(main())
