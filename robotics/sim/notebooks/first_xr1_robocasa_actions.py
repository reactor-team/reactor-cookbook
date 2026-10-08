import asyncio
import json
import time

import numpy as np
from reactor_robotics.session import ReactorSession

TRACKS = ("left_agentview", "right_agentview", "wrist_view")


async def main():
    session = ReactorSession("reactor/xr1-robocasa365", fps=15, frame_size=(256, 256))
    try:
        await session.connect(TRACKS, subscribe=("action_prediction",))
        await session.send(
            "set_task_description", {"task_description": "close the blender lid"}
        )
        session.set_frames(
            {
                name: np.full((256, 256, 3), 40 + 60 * i, dtype=np.uint8)
                for i, name in enumerate(TRACKS)
            }
        )
        # A static synthetic observation makes every history slot identical.
        await asyncio.sleep(0.8)
        rows = [[0.0] * 14 for _ in range(4)]
        await session.send(
            "set_state_history_json",
            {
                "state_history_json": json.dumps(
                    {"state_history": rows}, allow_nan=False
                )
            },
        )
        started = time.perf_counter()
        await session.send(
            "set_executed_step_json", {"executed_step_json": json.dumps({"step": 0})}
        )
        reply = await session.next_message("action_prediction", timeout_s=90)
        actions = np.asarray(reply["action"], dtype=np.float64)
        assert actions.shape == (16, 60) and np.isfinite(actions).all()
        assert reply["step"] == 0
        print("actions:", actions.shape)
        print("first simulator action:", actions[0, :12])
        print("request-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))
    finally:
        await session.close()


asyncio.run(main())
