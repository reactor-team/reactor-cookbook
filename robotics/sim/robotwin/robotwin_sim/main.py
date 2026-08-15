# ──────────────────────────────────────────────────────────────────────────
# Entrypoint.
#
#   RoboTwin 2.0 env (the authors' client, UNMODIFIED)
#        │  pickled dict over zmq :10086
#        ▼
#   Gateway  ──▶  Bridge (reactor-sdk)  ──▶  api.reactor.inc  ──▶  xwam
#
# Run the gateway in its own virtualenv (the sim's env pins numpy 1.23.5,
# which reactor-sdk cannot use; see README.md "Two virtualenvs"):
#
#   export REACTOR_API_KEY=...
#   python -m robotwin_sim.main --port 10086
#
# then run the authors' client from their checkout, in the sim's env, with
# --server_port pointed here. The client scores the episodes itself.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .bridge import DEFAULT_API_URL, DEFAULT_MODEL, Bridge
from .gateway import DEFAULT_PORT, Gateway


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoboTwin 2.0 gateway example: the authors' evaluation "
        "client, unmodified, against an xwam model served on Reactor"
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port to bind for the authors' client (default {DEFAULT_PORT}, "
        "which is what their --server_port defaults to)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--api-url", default=os.environ.get("REACTOR_API_URL", DEFAULT_API_URL)
    )
    p.add_argument("--api-key", default=os.environ.get("REACTOR_API_KEY", ""))
    p.add_argument(
        "--fps",
        type=int,
        default=15,
        help="video track rate. Also sets the settle wait before each request, "
        "so raising it shortens the round trip (default 15)",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=None,
        help="seconds between pushing frames and sending the request, so the "
        "new observation clears the encoder first (default: 3 track periods, "
        "never below 0.2 s)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for one chunk before retrying (default 30)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help="re-sends per request. A retry keeps the seeds and bumps a "
        "counter, so the answer is identical to the lost one (default 2)",
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="seconds to allow for the session to reach READY. A cold "
        "deployment schedules a GPU and stages weights first (default 300)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit(
            "REACTOR_API_KEY is required (env var or --api-key). Create a key "
            "at https://reactor.inc/account/api-keys and export it in this shell."
        )
    log = logging.getLogger("robotwin_sim")
    log.info("connecting to %s at %s", args.model, args.api_url)
    async with Bridge(
        api_key=args.api_key,
        api_url=args.api_url,
        model_name=args.model,
        fps=args.fps,
        settle_s=args.settle,
        timeout_s=args.timeout,
        retries=args.retries,
        ready_timeout_s=args.ready_timeout,
    ) as bridge:
        await Gateway(bridge.predict).serve(port=args.port)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Ctrl-C during an eval is normal; the session closes on the way out
        # (Bridge.__aexit__), which is what releases the GPU worker.
        pass


if __name__ == "__main__":
    main()
