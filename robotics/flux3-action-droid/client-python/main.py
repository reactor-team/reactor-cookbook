# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Run synthetic observations or replay an NPZ through FLUX 0.3.0; never actuates a robot."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from client import CHECKPOINTS, SEED_MAX, VIEWS, FluxClient, validate_observation


def observations(path: Path | None, count: int) -> list[tuple[dict, np.ndarray, str]]:
    if path is None:
        rng = np.random.default_rng(0)
        return [
            (
                {
                    view: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
                    for view in VIEWS
                },
                np.array([0, -0.6, 0, -2.2, 0, 1.6, 0.8, 0]),
                "put the marker in the cup",
            )
            for _ in range(count)
        ]
    with np.load(path, allow_pickle=False) as data:
        required = {*VIEWS, "proprio", "task"}
        if not required.issubset(data.files):
            raise ValueError(f"NPZ needs arrays: {sorted(required)}")
        if data["task"].ndim != 1 or data["task"].dtype.kind != "U":
            raise ValueError("task must be an (N,) Unicode array, not pickled objects")
        n = len(data["task"])
        if n < count:
            raise ValueError(f"Only {n} observations; choose --requests <= {n}")
        if data["proprio"].shape != (n, 8):
            raise ValueError("proprio must have shape (N, 8)")
        if any(data[view].ndim != 4 or len(data[view]) != n for view in VIEWS):
            raise ValueError("Every view must have shape (N, H, W, 3)")
        rows = [
            (
                {view: data[view][i] for view in VIEWS},
                data["proprio"][i],
                str(data["task"][i]),
            )
            for i in range(count)
        ]
    for frames, proprio, task in rows:
        validate_observation(frames, proprio)
        if not task.strip() or len(task) > 300:
            raise ValueError("Every task must contain 1–300 characters")
    return rows


async def run(args: argparse.Namespace) -> None:
    # Validate the whole replay before reserving a GPU session.
    rows = observations(args.observations, args.requests)
    print(
        "NPZ replay"
        if args.observations
        else "Synthetic protocol smoke test (not task quality)"
    )
    results = []
    async with FluxClient(
        args.checkpoint, model=args.model, settle_s=args.settle_s
    ) as client:
        print("Available:", ", ".join(client.available))
        print("Pinned:", client.checkpoint)
        for frames, proprio, task in rows:
            pred = await client.predict(frames, proprio, task, seed=args.seed)
            results.append(pred)
            print(
                f"step={pred.step} checkpoint={pred.checkpoint} shape={pred.actions.shape} "
                f"model={pred.inference_seconds * 1000:.1f} ms RTT={pred.round_trip_ms:.1f} ms"
            )
        await client.reset()
        frames, proprio, task = rows[0]
        pred = await client.predict(frames, proprio, task, seed=args.seed)
        print(
            f"Reset check: step={pred.step}, checkpoint={pred.checkpoint}, shape={pred.actions.shape}"
        )
    print(f"PASS: {len(results) + 1} valid predictions; session closed")
    print(
        f"Median model={np.median([p.inference_seconds for p in results]) * 1000:.1f} ms; "
        f"median request RTT={np.median([p.round_trip_ms for p in results]):.1f} ms"
    )
    print(
        "RTT excludes camera settling. No action equality or robot success assertion."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="base-bf16")
    parser.add_argument("--model", default="reactor/flux3-action-droid")
    parser.add_argument(
        "--observations", type=Path, help="Replay NPZ; omit for synthetic inputs"
    )
    parser.add_argument(
        "--requests", type=int, default=5, help="Predictions before one reset check"
    )
    parser.add_argument(
        "--seed", type=int, help="Sampling seed; default is each request's chunk_id"
    )
    parser.add_argument(
        "--settle-s", type=float, default=0.3, help="Camera settling heuristic"
    )
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("requests must be positive")
    if args.seed is not None and not 0 <= args.seed <= SEED_MAX:
        parser.error(f"seed must be between 0 and {SEED_MAX}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
