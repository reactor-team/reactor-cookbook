import asyncio
import json
import time

import numpy as np
from reactor_robotics.session import ReactorSession

VIEWS = ("exterior_view_1", "wrist_view")


async def main():
    session = ReactorSession("reactor/fastwam", fps=20, frame_size=(256, 256))
    try:
        await session.connect(VIEWS, subscribe=("action_prediction", "command_error"))
        await session.send(
            "set_task_description", {"task_description": "Open the drawer."}
        )
        session.set_frames(
            {name: np.zeros((256, 256, 3), dtype=np.uint8) for name in VIEWS}
        )
        request = {
            "proprio": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02, -0.02],
            "chunk_id": 1,
            "seed": 42,
        }
        started = time.perf_counter()
        await session.send("set_state_json", {"state_json": json.dumps(request)})
        reply = await session.next_message("action_prediction", timeout_s=120)
        actions = np.asarray(reply["actions"], dtype=np.float32)
        assert reply["step"] == request["chunk_id"]
        assert actions.shape == (32, 7)
        assert np.isfinite(actions).all()
        assert np.isin(actions[:, 6], [-1, 0, 1]).all()
        print("actions:", actions.shape)
        print("first step:", actions[0])
        print("model prediction seconds:", reply["inference_seconds"])
        print("request-to-reply ms:", round((time.perf_counter() - started) * 1000, 1))
    finally:
        await session.close()


asyncio.run(main())
