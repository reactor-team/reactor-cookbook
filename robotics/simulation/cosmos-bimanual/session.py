"""Small hosted Cosmos3 Nano Sim Bimanual session example."""

import asyncio
import os
from pathlib import Path

from reactor_sdk import Reactor


class Simulation:
    def __init__(self):
        self.client = Reactor(
            model_name="cosmos3-nano-sim-bimanual",
            api_key=os.environ["REACTOR_API_KEY"],
            api_url="https://api.reactor.inc",
            max_session_duration_seconds=300,
        )
        self.frames = 0
        self.statuses = []
        self.latest = None
        self.controls = {
            arm: {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
            for arm in ("left", "right")
        }
        self.controls.update(left_grip=0.0, right_grip=0.0)

        @self.client.on_message
        def message(value):
            if value.get("type") == "simulation_status":
                self.statuses.append(value.get("data", {}))

    async def start(self, image, prompt):
        await self.client.connect()

        @self.client.track("main_video").on_raw_frame
        def frame(data, width, height, *extra):
            self.frames += 1
            self.latest = (bytes(data), width, height)

        uploaded = await self.client.upload_file(str(image))
        await self.client.send_command("set_image", {"image": uploaded})
        await self.client.send_command("set_seed", {"seed": 42})
        await self.send_controls()
        await self.client.send_command("set_prompt", {"prompt": prompt})
        await self.wait_frames(40)

    async def send_controls(self):
        await self.client.send_command("set_teleop", {"teleop": self.controls})

    async def wait_frames(self, count):
        target = self.frames + count
        async with asyncio.timeout(90):
            while self.frames < target:
                await asyncio.sleep(0.05)

    async def reset(self):
        offset = len(self.statuses)
        await self.client.send_command("reset", {})
        async with asyncio.timeout(90):
            while not any(s.get("step_index") == 0 for s in self.statuses[offset:]):
                await asyncio.sleep(0.05)
        await self.wait_frames(40)

    def snapshot(self, path):
        from PIL import Image

        if self.latest is None:
            raise RuntimeError("No video frame received")
        data, width, height = self.latest
        # SDK raw frames are BGRA. Reject unexpected layouts explicitly.
        if len(data) != width * height * 4:
            raise ValueError(f"Unexpected raw frame size: {len(data)}")
        Image.frombytes("RGBA", (width, height), data, "raw", "BGRA").convert(
            "RGB"
        ).save(Path(path))
