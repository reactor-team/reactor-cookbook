import asyncio
import time

import numpy as np
from reactor_robotics.lingbot_va import VIEWS
from reactor_robotics.session import ReactorSession


async def main():
    session = ReactorSession("reactor/lingbot-va", fps=20, frame_size=(128, 128))
    try:
        await session.connect(VIEWS, subscribe=("action_prediction",))
        session.set_frames(
            {view: np.zeros((128, 128, 3), dtype=np.uint8) for view in VIEWS}
        )
        await asyncio.sleep(0.3)
        await session.send("set_executed_action_json", {"executed_action_json": ""})
        await session.send("reset", {"sampling_seed": 42})
        started = time.perf_counter()
        await session.send(
            "set_task_description", {"task_description": "Put the bowl on the plate."}
        )
        reply = await session.next_message("action_prediction", timeout_s=120)
        actions = np.asarray(reply["action"], dtype=np.float64)
        assert reply["step"] == 0
        assert actions.shape == (16, 7)
        assert np.isfinite(actions).all()
        executable = actions[4:]
        print("action shape:", actions.shape)
        print("executable shape:", executable.shape)
        print("first predicted row:", executable[0])
        print("task-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))
    finally:
        await session.close()


asyncio.run(main())
