import asyncio
import time

import numpy as np
from reactor_robotics.session import ReactorSession

TRACKS = ("exterior_1", "exterior_2", "wrist")
STATE = {
    "joint_position": [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.885, 0.0],
    "gripper_position": 0.0,
}


async def main():
    session = ReactorSession("reactor/dreamzero", fps=15, frame_size=(180, 320))
    try:
        await session.connect(
            TRACKS,
            subscribe=(
                "action_chunk",
                "command_error",
                "episode_started",
                "prompt_accepted",
            ),
            ready_timeout_s=900,
        )
        # Synthetic fixtures: never execute their predicted actions on hardware.
        rng = np.random.default_rng(7)
        frames = {
            name: rng.integers(0, 256, (180, 320, 3), dtype=np.uint8) for name in TRACKS
        }
        for field, value in STATE.items():
            await session.send("set_" + field, {field: value})
        session.set_frames(frames)
        await asyncio.sleep(0.5)
        started = time.perf_counter()
        await session.send("set_prompt", {"prompt": "put the block in the bin"})
        reply = await session.next_message("action_chunk", timeout_s=300)
        actions = np.asarray(reply["actions"], dtype=np.float64)
        assert actions.shape == (24, 8), actions.shape
        assert np.isfinite(actions).all()
        assert reply["chunk_index"] == 0, reply["chunk_index"]
        print("actions:", actions.shape)
        print("first row:", actions[0])
        print("obs_seq:", reply["obs_seq"])
        print("model inference seconds:", reply["inference_seconds"])
        print("prompt-to-chunk seconds:", round(time.perf_counter() - started, 3))
    finally:
        await session.close()


asyncio.run(main())
