# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Drive reactor/rho-yam-box at real-time cadence with synthetic cameras and a mock YAM.

Nothing here moves a robot. Replace SyntheticCameras and MockYam with your
hardware to close the loop.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from client import (
    CONTROL_HZ,
    SEED_MAX,
    VIEWS,
    RhoClient,
    arm_vector,
    pack_state,
    position_jump,
)

TASK = "pick up the red cube and place it in the box"
# Both arms at a mid-workspace pose, identity rotation, gripper value 0.3.
# These values are inside the range of the checkpoint's training data.
HOME_ARM = arm_vector([0.4, 0.0, 0.2], np.eye(3), 0.3)
HOME_STATE = pack_state(HOME_ARM, HOME_ARM)


class SyntheticCameras:
    """Camera-like RGB frames. Replace read() with a capture from your three cameras.

    Each view is a fixed scene (a color gradient and a few colored blocks) that
    moves a few pixels between reads, like a live camera. Random noise would be
    a poor stand-in: a video encoder cannot compress it, so it would add
    hundreds of milliseconds of transport that real camera frames do not have.
    """

    def __init__(self, seed: int = 0, height: int = 480, width: int = 640) -> None:
        self.rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:height, 0:width]
        self.scenes = {}
        for view in VIEWS:
            low, high = self.rng.integers(40, 216, size=(2, 3))
            # A smooth gradient from one color to another, top left to bottom right.
            t = ((x / width + y / height) / 2)[..., None]
            scene = low + (high - low) * t
            for _ in range(6):
                top, left = (
                    self.rng.integers(0, height - 80),
                    self.rng.integers(0, width - 80),
                )
                h, w = self.rng.integers(30, 80, size=2)
                scene[top : top + h, left : left + w] = self.rng.integers(
                    0, 256, size=3
                )
            self.scenes[view] = scene.astype(np.uint8)

    def read(self) -> dict[str, np.ndarray]:
        """Return one uint8 RGB frame for each view in VIEWS."""
        # Keep the camera roles fixed: a wrist frame on scene_view gives wrong actions.
        dy, dx = self.rng.integers(-3, 4, size=2)
        return {
            view: np.roll(scene, (int(dy), int(dx)), axis=(0, 1))
            for view, scene in self.scenes.items()
        }


class MockYam:
    """A stand-in robot that reaches each target exactly. Replace it with your YAM driver.

    A real driver must:
    - read the measured end-effector pose of both arms (forward kinematics),
    - convert each target row to joint commands (inverse kinematics),
    - check limits, and refuse a target that is far from the measured pose,
    - not block the event loop: the client sends camera frames on it.
    """

    def __init__(self, control_hz: float = CONTROL_HZ, realtime: bool = True) -> None:
        self.period_s = 1 / control_hz
        self.realtime = realtime
        self.state = HOME_STATE.copy()

    def get_state(self) -> np.ndarray:
        """Return the 20-value state: left arm, then right arm."""
        return self.state.copy()

    async def execute(self, rows: np.ndarray) -> None:
        """Step through the target rows at the control rate."""
        for row in rows:
            if self.realtime:
                await asyncio.sleep(self.period_s)
            self.state = np.array(row, dtype=np.float64)


def report(label: str, prediction, state: np.ndarray) -> str:
    """Format one prediction as one line of text."""
    left, right = position_jump(state, prediction.actions[0])
    return (
        f"{label}={prediction.step} shape={prediction.actions.shape} "
        f"execute={prediction.execution_horizon} "
        f"model={prediction.inference_seconds * 1000:.1f} ms "
        f"RTT={prediction.round_trip_ms:.1f} ms "
        f"view skew={prediction.view_skew_us} us "
        f"first-row jump L={left * 1000:.0f} mm R={right * 1000:.0f} mm"
    )


async def run(args: argparse.Namespace) -> None:
    cameras = SyntheticCameras()
    robot = MockYam(realtime=not args.no_realtime)
    print("Synthetic protocol smoke test (not task quality); no robot moves")
    results = []
    async with RhoClient(model=args.model, settle_s=args.settle_s) as client:
        for _ in range(args.requests):
            state = robot.get_state()
            prediction = await client.predict(
                cameras.read(), state, args.task, seed=args.seed
            )
            results.append(prediction)
            print(report("step", prediction, state))
            # Execute the first rows, then ask again from the new state.
            await robot.execute(prediction.to_execute)
        await client.reset()
        state = robot.get_state()
        prediction = await client.predict(
            cameras.read(), state, args.task, seed=args.seed
        )
        print(report("Reset check: step", prediction, state))
    print(f"PASS: {len(results) + 1} valid predictions; session closed")
    print(
        f"Median model={np.median([p.inference_seconds for p in results]) * 1000:.1f} ms; "
        f"median request RTT={np.median([p.round_trip_ms for p in results]):.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="reactor/rho-yam-box")
    parser.add_argument(
        "--task", default=TASK, help="Language instruction, 1-300 characters"
    )
    parser.add_argument(
        "--requests", type=int, default=5, help="Predictions before one reset check"
    )
    parser.add_argument(
        "--seed", type=int, help="Sampling seed; default is each request's chunk_id"
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help="Extra wait after pushing new frames, before the request (not needed)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not wait 1/15 s per executed row",
    )
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("requests must be positive")
    if args.seed is not None and not 0 <= args.seed <= SEED_MAX:
        parser.error(f"seed must be between 0 and {SEED_MAX}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
