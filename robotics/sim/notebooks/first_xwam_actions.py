import asyncio

import numpy as np
from reactor_robotics.xwam import VIEWS, XwamClient


async def main():
    client = XwamClient(model="reactor/xwam")
    try:
        await client.connect()
        prediction = await client.predict(
            frames={name: np.zeros((240, 320, 3), dtype=np.uint8) for name in VIEWS},
            # Left xyz, quaternion wxyz, gripper; then the same for the right arm.
            proprio=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] * 2,
            task="Pick up the bottle.",
        )
        assert prediction.step == 1
        assert prediction.actions.shape == (32, 14)
        assert np.isfinite(prediction.actions).all()
        print("actions:", prediction.actions.shape)
        print("first step:", prediction.actions[0])
        print("request-to-reply ms:", round(prediction.latency_ms, 1))
    finally:
        await client.close()


asyncio.run(main())
