import asyncio
import json

import numpy as np
from reactor_robotics.session import ReactorSession

TRACKS = ("wrist_view", "exterior_view_1", "exterior_view_2")


async def main():
    session = ReactorSession(
        "reactor/flux3-action-droid", fps=15, frame_size=(360, 640)
    )
    try:
        await session.connect(
            TRACKS,
            subscribe=("checkpoint_selected", "action_prediction", "command_error"),
        )
        await session.send("get_checkpoint", {})
        status = await session.next_message("checkpoint_selected", timeout_s=20)
        selected = status["checkpoint"]
        if selected not in status["available"]:
            raise RuntimeError(
                "Default checkpoint unavailable; select a listed alternative"
            )
        await session.send("select_checkpoint", {"checkpoint": selected})
        confirmed = await session.next_message("checkpoint_selected", timeout_s=20)
        assert confirmed["locked"] and confirmed["checkpoint"] == selected
        await session.send(
            "set_task_description", {"task_description": "put the marker in the cup"}
        )
        rng = np.random.default_rng(0)
        session.set_frames(
            {
                name: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
                for name in TRACKS
            }
        )
        request = {
            "proprio": [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8, 0.0],
            "chunk_id": 1,
            "seed": 1,
        }
        await session.send("set_state_json", {"state_json": json.dumps(request)})
        reply = await session.next_message("action_prediction", timeout_s=90)
        actions = np.asarray(reply["actions"], dtype=np.float64)
        assert reply["step"] == request["chunk_id"]
        assert actions.shape == (32, 8) and np.isfinite(actions).all()
        assert reply["checkpoint"] == selected
        print("checkpoint:", selected, "actions:", actions.shape)
    finally:
        await session.close()


asyncio.run(main())
