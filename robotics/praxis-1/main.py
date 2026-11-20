# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Run a few Praxis-1 chunks against a synthetic robot; never actuates hardware.

The synthetic robot streams moving test images and reports a static pose. It
checks the protocol (stamping, the 25-row state, the step handshake), not task
quality. ``--mode stepped`` runs the same loop in a stepped session: one call
per chunk that carries the chunk's video, the state and the step echo.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Mapping

import numpy as np

from praxis_client import (
    DROID,
    EMBODIMENTS,
    JOIN_MEANINGS,
    STEPPED_JOIN_MEANINGS,
    ArmState,
    CommandRow,
    Embodiment,
    PraxisClient,
    SteppedPraxisClient,
)

#: A plausible start pose per arm: [x, y, z, qw, qx, qy, qz], gripper open.
START_POSES = {
    "arm": ArmState([0.45, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0], 0.0),
    "left": ArmState([0.30, 0.25, 0.20, 1.0, 0.0, 0.0, 0.0], 0.0),
    "right": ArmState([0.30, -0.25, 0.20, 1.0, 0.0, 0.0, 0.0], 0.0),
}
#: A default task per robot. YAM's has the form of its training captions, with
#: no closing period; the model reads the task verbatim.
TASKS = {
    "droid": "Put the red block in the bowl.",
    "yam": "Pick up the pen from the center of the table, then place the pen in the bag",
}


class SyntheticRobot:
    """Implements :class:`praxis_client.Robot` with no hardware.

    Each camera shows a gradient with a square that moves one step per read, so
    consecutive frames differ. The arms hold their start pose; ``execute``
    records the commands instead of moving, and ``hold`` counts its calls.
    """

    def __init__(self, embodiment: Embodiment) -> None:
        self.embodiment = embodiment
        if embodiment.max_size:
            width, height = embodiment.max_size
        else:
            width, height = embodiment.max_side or 416, 260
        y, x = np.mgrid[0:height, 0:width]
        self.base = np.stack(
            [x * 255 // width, y * 255 // height, np.full_like(x, 96)], axis=-1
        ).astype(np.uint8)
        self.reads = 0
        self.executed: list[CommandRow] = []
        self.holds = 0

    def read_cameras(self) -> Mapping[str, np.ndarray]:
        self.reads += 1
        frames = {}
        height, width = self.base.shape[:2]
        for i, name in enumerate(self.embodiment.cameras):
            frame = self.base.copy()
            left = (self.reads * 8 + i * 60) % (width - 48)
            frame[height // 2 - 24 : height // 2 + 24, left : left + 48] = 255
            frames[name] = frame
        return frames

    def read_arm_state(self) -> Mapping[str, ArmState]:
        return {name: START_POSES[name] for name in self.embodiment.arms}

    def execute(self, row: CommandRow) -> None:
        self.executed.append(row)

    def hold(self) -> None:
        self.holds += 1


async def run(args: argparse.Namespace) -> None:
    embodiment = EMBODIMENTS[args.robot]
    robot = SyntheticRobot(embodiment)
    stepped = args.mode == "stepped"
    client_class = SteppedPraxisClient if stepped else PraxisClient
    task = args.task or TASKS[args.robot]
    print(
        f"Synthetic robot on {args.model or embodiment.model}, {args.mode} "
        "(not task quality)"
    )
    async with client_class(
        robot,
        embodiment,
        model=args.model,
        pose_frame=args.pose_frame if embodiment is DROID else None,
        status_every=1,
    ) as client:
        joins: set[str] = set()
        for episode in range(args.episodes):
            chunks = await client.run_episode(task, args.chunks)
            joins.update(c.join for c in chunks if c.join)
            for chunk in chunks:
                first = chunk.rows[0].targets
                arm = next(iter(first))
                sent = f" bytes={chunk.sent_bytes}" if stepped else ""
                print(
                    f"episode={episode} step={chunk.step} rows={len(chunk.rows)} "
                    f"latency={chunk.latency_ms:.0f} ms{sent} join={chunk.join or '?'} "
                    f"first {arm} xyz={np.round(first[arm].position, 3).tolist()}"
                )
    print(f"Executed {len(robot.executed)} rows; session closed")
    meanings = STEPPED_JOIN_MEANINGS if stepped else JOIN_MEANINGS
    for join in sorted(joins):
        print(f"join={join}: {meanings.get(join, 'unknown value')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=sorted(EMBODIMENTS), default="droid")
    parser.add_argument(
        "--mode",
        choices=("streamed", "stepped"),
        default="streamed",
        help="streamed: stamped video tracks; stepped: one call per chunk",
    )
    parser.add_argument("--model", help="owner/name; default reactor/praxis-1-<robot>")
    parser.add_argument(
        "--task", help="default: a task in the form the robot's model was trained on"
    )
    parser.add_argument("--chunks", type=int, default=4, help="chunks per episode")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--pose-frame",
        default="fr3_hand",
        choices=("fr3_hand", "robolab_droid", "molmo_spaces_droid"),
        help="DROID only: the frame of ee_pose and of the returned poses",
    )
    args = parser.parse_args()
    if args.chunks < 1 or args.episodes < 1:
        parser.error("chunks and episodes must be positive")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
