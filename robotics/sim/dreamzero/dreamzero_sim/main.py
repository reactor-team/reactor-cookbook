# ──────────────────────────────────────────────────────────────────────────
# Entrypoint.
#
#   RoboLab (Isaac Sim, its own container, UNMODIFIED)
#        │  openpi msgpack-WebSocket :5000
#        ▼
#   WebsocketPolicyServer -> RoboLabPolicy -> Bridge (reactor-sdk)
#        │
#        ▼  api.reactor.inc  ──▶  dreamzero
#
# Run the gateway anywhere RoboLab can reach:
#
#   export REACTOR_API_KEY=...
#   python -m dreamzero_sim.main --port 5000
#
# then point RoboLab's unmodified runner at it. RoboLab scores the episodes itself.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .bridge import DEFAULT_API_URL, DEFAULT_MODEL, DEFAULT_READY_TIMEOUT_S, Bridge
from .gateway import RoboLabPolicy
from .policy_server import PolicyServerConfig, WebsocketPolicyServer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoboLab gateway example: Isaac Sim's DROID benchmark "
        "driven by a Reactor-served dreamzero"
    )
    p.add_argument(
        "--port", type=int, default=5000, help="openpi WebSocket port RoboLab connects to"
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--api-url", default=os.environ.get("REACTOR_API_URL", DEFAULT_API_URL)
    )
    p.add_argument("--api-key", default=os.environ.get("REACTOR_API_KEY", ""))
    p.add_argument(
        "--chunk-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for a fresh chunk per request (default 300)",
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT_S,
        help="seconds to allow for the session to reach READY. This model is "
        f"served on two GPUs, so a cold start is minutes (default {DEFAULT_READY_TIMEOUT_S:.0f})",
    )
    p.add_argument(
        "--prime-stagger",
        type=float,
        default=2.0,
        help="seconds between starting each camera sender on the first "
        "observation, so the order the streams appear on the wire is "
        "deterministic. 0 disables (default 2.0)",
    )
    p.add_argument(
        "--connect-eagerly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="bring the Reactor session up at startup instead of on RoboLab's "
        "first request. DEFAULT OFF, and normally leave it off: the tracks are "
        "queue-fed, so an early connection sits idle with no video flowing for "
        "as long as RoboLab takes to boot, which has been observed to break "
        "the serving runtime's mapping of inbound video to track names. Turn it "
        "on only to surface a signalling or capacity problem at startup rather "
        "than inside RoboLab's first query",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit(
            "REACTOR_API_KEY is required (env var or --api-key). Create a key "
            "at https://reactor.inc/account/api-keys and export it in this shell."
        )
    log = logging.getLogger("dreamzero_sim")
    bridge = Bridge(
        api_key=args.api_key,
        api_url=args.api_url,
        model_name=args.model,
        chunk_timeout_s=args.chunk_timeout,
        ready_timeout_s=args.ready_timeout,
        prime_stagger_s=args.prime_stagger,
    )
    if args.connect_eagerly:
        await bridge.ensure_connected()

    policy = RoboLabPolicy(bridge)
    server = WebsocketPolicyServer(
        policy=policy,
        server_config=PolicyServerConfig(),
        host=args.host,
        port=args.port,
    )
    log.info(
        "gateway serving openpi on %s:%d -> %s (%s); point RoboLab's run.py here",
        args.host,
        args.port,
        args.model,
        args.api_url,
    )
    try:
        await server.run()
    finally:
        await bridge.close()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Ctrl-C during an evaluation is normal; the session closes on the way
        # out, which is what releases the GPU workers.
        pass


if __name__ == "__main__":
    main()
