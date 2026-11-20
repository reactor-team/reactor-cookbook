"""Control the AgiBot neural simulator from a terminal; save preview snapshots."""

import argparse
import asyncio
from pathlib import Path

from session import Simulation


async def run(args):
    sim = Simulation()
    arm = "left"
    async with sim.client:
        await sim.start(args.image, args.prompt)
        print("Commands: left/right, x+/x-/y+/y-/z+/z-, open/close, reset, snap, quit")
        index = 0
        while True:
            try:
                command = (await asyncio.to_thread(input, f"{arm}> ")).strip().lower()
            except EOFError:
                break
            if command == "quit":
                break
            if command in ("left", "right"):
                arm = command
                continue
            if command in ("x+", "x-", "y+", "y-", "z+", "z-"):
                axis = "xyz".index(command[0])
                position = sim.controls[arm]["position"]
                position[axis] = max(
                    -0.10,
                    min(0.10, position[axis] + (0.01 if command[1] == "+" else -0.01)),
                )
                await sim.send_controls()
                await sim.wait_frames(20)
            elif command in ("open", "close"):
                sim.controls[f"{arm}_grip"] = float(command == "close")
                await sim.send_controls()
                await sim.wait_frames(20)
            elif command == "reset":
                await sim.reset()
            elif command != "snap":
                print("Unknown command; use one of the commands listed above.")
                continue
            index += 1
            path = args.output / f"preview-{index:03}.png"
            sim.snapshot(path)
            print(f"Saved {path}; received {sim.frames} preview frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))
