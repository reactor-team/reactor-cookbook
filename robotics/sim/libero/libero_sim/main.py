# ──────────────────────────────────────────────────────────────────────────
# Entrypoint.
#
#   main thread:   LiberoEnv (LIBERO/robosuite) ── SimDriver @20 Hz ─┐
#                                                                     ├─ RolloutState
#   bridge thread: Bridge (reactor-sdk) ── api.reactor.inc ── lingbot-va ┘
#
# The sim is on the main thread because macOS gives it no choice (see
# loop.SimDriver).
#
# Run:
#   export LIBERO_CONFIG_PATH="$PWD/.libero"
#   export REACTOR_API_KEY=...          # or --api-key
#   python -m libero_sim.main --task-id 0
#
# The task string is taken from the LIBERO task by default, so it matches
# what the checkpoint was trained on; --task only overrides it for
# experiments.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading

from .bridge import DEFAULT_MODEL, BridgeThread
from .contract import ACTION_HORIZON, CAM_SIZE, CONTROL_HZ, SEED_SKIP_STEPS, VIEWS
from .env import EnvConfig, LiberoEnv
from .loop import RolloutState, SimDriver
from .record import FrameRecorder

# A ceiling, not an expectation: connect is normally seconds. This timeout
# only exists so a model that never comes up fails with a clear message
# instead of hanging.
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_MAX_EPISODE_STEPS = 600


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LingBot-VA LIBERO sim-wrapper example")
    p.add_argument("--suite", default="libero_10",
                   help="LIBERO task suite (default libero_10 = LIBERO-Long)")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-state-id", type=int, default=0,
                   help="which of the benchmark's pruned_init states to start from")
    p.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS,
                   help="give up on an episode after this many steps (0 = never)")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="end the run after this many episodes even without a success "
                        "(0 = retry until success); use 1 for one attempt per task")
    p.add_argument("--task", default=None,
                   help="override the language instruction (default: the LIBERO task's own)")
    p.add_argument("--seed", type=int, default=0, help="seeds the LIBERO env")
    p.add_argument("--cam-size", type=int, default=CAM_SIZE,
                   help=f"offscreen render resolution; the recording captures it "
                        f"full-size while the policy always receives it downsampled "
                        f"to {CAM_SIZE} (raise it for a higher-quality --record clip)")
    p.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    p.add_argument("--exec-steps", type=int, default=ACTION_HORIZON,
                   help=f"steps to execute per chunk before echoing (1..{ACTION_HORIZON})")
    p.add_argument("--seed-skip", type=int, default=SEED_SKIP_STEPS,
                   help="leading steps of an episode's FIRST chunk left unexecuted. "
                        "That frame is the server's conditioning slot, not a prediction; "
                        "0 restores the old behaviour of executing it")
    p.add_argument("--settle-steps", type=int, default=5,
                   help="zero-action steps after set_init_state, so the policy is not "
                        "prompted mid-drop")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="stop the rollout after this long (0 = run until Ctrl-C)")
    p.add_argument("--echo-delay", type=float, default=0.1,
                   help="settle window before echoing, so post-execution frames land first")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Reactor model name; try the unqualified name if connect fails")
    # lingbot-va serves from api.reactor.inc. The wrong host answers with a
    # 429 that reads as missing capacity, which has cost debugging time more
    # than once.
    p.add_argument("--api-url",
                   default=os.environ.get("REACTOR_API_URL", "https://api.reactor.inc"))
    p.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT)
    p.add_argument("--api-key", default=os.environ.get("REACTOR_API_KEY", ""))
    p.add_argument("--no-flip", action="store_true",
                   help="publish raw (upside-down) frames; for diagnosing orientation only")
    p.add_argument("--record", default=None, metavar="PATH",
                   help="write the published camera views, side by side, to an mp4")
    p.add_argument("--overlay", action="store_true",
                   help="burn the CONNECTING/WAITING/RUNNING state into the recorded frames")
    p.add_argument("--capture", choices=("realtime", "active"), default="realtime",
                   help="realtime: sample on the wall clock, lock-step holds included; "
                        "active: drop the holds and keep only policy-driven motion")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit(
            "REACTOR_API_KEY is required (env var or --api-key). Create a key "
            "at https://reactor.inc/account/api-keys and export it in this shell."
        )
    if not os.environ.get("LIBERO_CONFIG_PATH"):
        logging.getLogger("libero").warning(
            "LIBERO_CONFIG_PATH is unset; LIBERO will read ~/.libero/config.yaml "
            "and prompt interactively if it is missing"
        )

    env = LiberoEnv(EnvConfig(
        suite=args.suite, task_id=args.task_id, init_state_id=args.init_state_id,
        cam_size=args.cam_size, seed=args.seed, control_hz=args.control_hz,
        settle_steps=args.settle_steps,
    ))
    task = args.task or env.language

    stop_event = threading.Event()
    rollout = RolloutState(
        env, exec_steps=args.exec_steps, seed_skip=args.seed_skip,
        flip_frames=not args.no_flip, max_episode_steps=args.max_episode_steps,
        max_episodes=args.max_episodes, stop_on_success=stop_event,
    )
    driver = SimDriver(rollout, hz=args.control_hz, max_seconds=args.max_seconds, stop_event=stop_event)
    log = logging.getLogger("libero")

    bridge = BridgeThread(
        rollout=rollout, api_key=args.api_key, api_url=args.api_url, task=task,
        model_name=args.model, echo_delay=args.echo_delay,
    )

    recorder = None
    if args.record:
        def _status() -> str:
            if not bridge.is_ready:
                return "CONNECTING"
            return "WAITING" if rollout.is_idle() else "RUNNING"
        recorder = FrameRecorder(
            args.record, {n: rollout.frame_source(n) for n in VIEWS}, hz=args.control_hz,
            status=_status, overlay=args.overlay, active_only=(args.capture == "active"),
        )
        recorder.start()

    bridge.start()

    # Don't start stepping until the model is actually there. The first
    # chunk is what unblocks the rollout anyway.
    log.info("connecting to %s", args.model)
    if not bridge.ready.wait(timeout=args.connect_timeout):
        bridge.stop()
        raise SystemExit(f"bridge did not become ready within {args.connect_timeout:.0f}s")
    if bridge.failed is not None:
        raise SystemExit(f"bridge failed to connect: {bridge.failed}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: driver.stop())

    try:
        log.info("connected; running (Ctrl-C to stop)")
        driver.run()
    finally:
        bridge.stop()
        bridge.join(timeout=10.0)
        if recorder:
            recorder.stop()
            log.info("wrote %s", args.record)
        env.close()
        log.info("diagnostics: %s", rollout.diag)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _run(args)


if __name__ == "__main__":
    main()
