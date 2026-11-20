"""Initialize an AgiBot scene, change controls, and restart the world."""

import argparse
import asyncio
from pathlib import Path

from session import Simulation


async def run(args):
    sim = Simulation()
    async with sim.client:
        await sim.start(args.image, args.prompt)
        sim.snapshot(args.output / "initial.png")
        sim.controls["left"]["position"][0] = 0.01
        sim.controls["left_grip"] = 1.0
        await sim.send_controls()
        await sim.wait_frames(80)
        sim.snapshot(args.output / "controlled.png")
        await sim.reset()
        sim.snapshot(args.output / "reset.png")
        print(f"Received {sim.frames} preview frames; controls and reset completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))
