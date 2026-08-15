# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""RoboCasa365 evaluation against a Reactor-served `xr1-robocasa365`.

Reuses the vendor's `select_tasks` / `evaluate_task` / `build_summary`
UNMODIFIED and swaps only the client, so the rollout loop, task registry,
seeding and success criteria are the upstream ones. Every vendor knob
(`--task-set`, `--num-trials`, `--replan-steps`, `--seed`, ...) is accepted
and forwarded.

    python -m robocasa365_sim.entry \
        --vendor-dir ~/Xiaomi-Robotics-1 \
        --api-url http://127.0.0.1:8080 \
        --task-name CloseBlenderLid --num-trials 3 \
        --save-root-dir eval_results/reactor

`--model-path` is not needed: the model side owns the processor. `--crop-ratio`
must match the served model's configured value, because the crop happens
model-side in this topology.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def _load_vendor(vendor_dir: Path):
    """Import the vendor's `entry` module from a RoboCasa365 checkout.

    The checkout is not vendored here and not pip-installable, so its location
    is a parameter rather than a fixed path: pass `--vendor-dir` or set
    `XR1_VENDOR_DIR`.
    """
    eval_dir = vendor_dir / "eval_robocasa365"
    if not (eval_dir / "entry.py").is_file():
        raise SystemExit(
            f"no eval_robocasa365/entry.py under {vendor_dir}. Point --vendor-dir "
            "(or XR1_VENDOR_DIR) at a Xiaomi-Robotics-1 checkout."
        )
    sys.path.insert(0, str(eval_dir))
    sys.path.insert(0, str(vendor_dir / "deploy"))
    import entry  # noqa: PLC0415

    return entry


def parse_args(entry, argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate xr1-robocasa365 over the Reactor transport."
    )
    parser.add_argument("--vendor-dir", default=os.environ.get("XR1_VENDOR_DIR"))
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", default="xr1-robocasa365")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.15,
        help="Wait between the frame pushes and the echo, covering transit.",
    )
    # Vendor knobs; defaults identical to the vendor's own parse_args.
    parser.add_argument("--robot-type", default=entry.ROBOT_TYPE)
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--task-set", default="target50")
    parser.add_argument("--task-name", action="append", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--obs-history", type=int, default=4)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--save-root-dir", default="eval_results/robocasa365-reactor")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-failure-videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args(argv)
    # entry.build_summary reads args.model_path.
    args.model_path = f"reactor:{args.model}@{args.api_url}"
    return args


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--vendor-dir", default=os.environ.get("XR1_VENDOR_DIR"))
    known, _ = pre.parse_known_args(argv)
    if not known.vendor_dir:
        raise SystemExit("--vendor-dir (or XR1_VENDOR_DIR) is required")
    entry = _load_vendor(Path(known.vendor_dir).expanduser().resolve())

    from robocasa365_sim.client import ReactorEvalClient

    args = parse_args(entry, argv)
    entry.validate_args(args)

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    task_to_index, selected_tasks = entry.select_tasks(args, TASK_SET_REGISTRY)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.save_root_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing evaluation outputs to %s", output_dir)
    logging.info("Transport: Reactor SDK -> %s (%s)", args.api_url, args.model)

    client = ReactorEvalClient(
        api_url=args.api_url, model=args.model, settle_s=args.settle_s
    )
    task_stats = {}
    try:
        for env_name in selected_tasks:
            task_stats[env_name] = entry.evaluate_task(
                env_name=env_name,
                task_index=task_to_index[env_name],
                args=args,
                client=client,
                gym=gym,
                get_task_horizon=get_task_horizon,
                convert_action=convert_action,
                output_dir=output_dir,
            )
    finally:
        summary = entry.build_summary(args, task_stats)
        summary["transport"] = {
            "kind": "reactor_sdk",
            "api_url": args.api_url,
            "model": args.model,
            "settle_s": args.settle_s,
            "latency": client.latency_stats(),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if task_stats:
            rate = summary["episode_success_rate"]
            logging.info("Wrote summary to %s", output_dir / "summary.json")
            logging.info(
                "Success rate: %.2f%% (%d/%d)  |  infer %s",
                100 * (rate or 0),
                summary["successes"],
                summary["num_episodes"],
                client.latency_stats(),
            )
        client.close()


if __name__ == "__main__":
    main()
