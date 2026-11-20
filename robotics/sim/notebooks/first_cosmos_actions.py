import asyncio
import time

import numpy as np
from reactor_robotics.cosmos_droid import TRACKS, encode_proprio
from reactor_robotics.session import ReactorSession


async def main():
    session = ReactorSession(
        "reactor/cosmos-nano-policy-droid", fps=15, frame_size=(180, 320)
    )
    try:
        await session.connect(TRACKS, subscribe=("action_prediction",))
        await session.send(
            "set_task_description", {"task_description": "Put the banana in the bowl."}
        )
        session.set_frames(
            {name: np.zeros((180, 320, 3), dtype=np.uint8) for name in TRACKS}
        )
        await asyncio.sleep(0.2)
        # Sample state for a transport check; not a command to move the robot.
        proprio = encode_proprio([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0], 0.0)
        started = time.perf_counter()
        await session.send("set_proprio_json", {"proprio_json": proprio})
        reply = await session.next_message("action_prediction", timeout_s=90)
        actions = np.asarray(reply["action"], dtype=np.float64)
        assert reply["step"] == 0
        assert actions.shape == (32, 8)
        assert np.isfinite(actions).all()
        print("actions:", actions.shape)
        print("first step:", actions[0])
        print("request-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))
    finally:
        await session.close()


asyncio.run(main())
