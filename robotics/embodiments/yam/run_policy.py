# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Send one observation to hosted Rho and save its predicted actions."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from rho_client import RhoClient, TRACKS, observation


def load_observation(folder: Path) -> tuple[dict, np.ndarray]:
    frames = {}
    for name in TRACKS:
        with Image.open(folder / f"{name}.png") as image:
            frames[name] = np.asarray(image.convert("RGB"))
    return observation(frames, json.loads((folder / "state.json").read_text()))


def demo_observation() -> tuple[dict, np.ndarray]:
    # A transport smoke fixture, not a task or hardware pose recommendation.
    frames = {name: np.full((480, 640, 3), 32 + 48 * index, dtype=np.uint8)
              for index, name in enumerate(TRACKS)}
    state = [0.4, 0., 0.2, 1., 0., 0., 0., 1., 0., 1.] * 2
    return observation(frames, state)


async def run(args: argparse.Namespace) -> None:
    frames, state = demo_observation() if args.demo else load_observation(args.observation)
    async with RhoClient(args.task) as client:
        result = await client.predict(frames, state, seed=args.seed)
    result["synthetic_input"] = args.demo
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved 50 x 20 predicted actions to {args.output}; no actions executed.")
    if result["invalid_gripper_rows"]:
        print(f"Out-of-range gripper values in {len(result['invalid_gripper_rows'])} rows; raw values preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--observation", type=Path, help="Directory with state.json and three camera PNGs")
    inputs.add_argument("--demo", action="store_true", help="Synthetic transport smoke test only")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("output") / (time.strftime("prediction-%Y%m%d-%H%M%S") + ".json"))
    asyncio.run(run(parser.parse_args()))
