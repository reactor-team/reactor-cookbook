# ──────────────────────────────────────────────────────────────────────────
# Entrypoint.
#
#   main thread:    openpi WebsocketPolicyServer  ◄── RoboLab (Isaac, its own
#                        │ infer()                     container, UNMODIFIED)
#                        ▼
#                   GatewayState  ── hand-off ──┐
#   bridge thread:  Bridge (reactor-sdk) ── api.reactor.inc ── cosmos-nano-policy-droid
#
# Run the gateway on the sim machine:
#
#   export REACTOR_API_KEY=...
#   python -m cosmos_droid_sim.main --port 8000
#
# then point RoboLab's unmodified cosmos3 runner at it, exactly as its docs say:
#
#   ./isaac-sim/python.sh policies/cosmos3/run.py --task BananaInBowlTask \
#       --headless --num-envs 1 --num-runs 1 \
#       --remote-host localhost --remote-port 8000
#
# RoboLab scores the episode itself (output/<ts>_cosmos3/<task>/log_0_env0.json).
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import os

from .bridge import DEFAULT_API_URL, DEFAULT_MODEL, BridgeThread
from .gateway import GatewayState

# A ceiling, not an expectation: a warm session answers in under a second.
# The timeout exists so a model that never comes READY fails with a message
# instead of hanging RoboLab's first request forever (a cold start pulls a
# multi-GB checkpoint).
DEFAULT_CONNECT_TIMEOUT = 300.0
DEFAULT_CHUNK_TIMEOUT = 90.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cosmos-nano-policy-droid RoboLab gateway example")
    p.add_argument("--port", type=int, default=8000,
                   help="openpi WebSocket port RoboLab connects to")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-url", default=os.environ.get("REACTOR_API_URL", DEFAULT_API_URL))
    p.add_argument("--api-key", default=os.environ.get("REACTOR_API_KEY", ""))
    p.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT)
    p.add_argument("--chunk-timeout", type=float, default=DEFAULT_CHUNK_TIMEOUT,
                   help="max seconds to wait for a chunk before failing the request")
    p.add_argument("--settle", type=float, default=0.1,
                   help="pause between pushing frames and opening the flow gate, "
                        "so the fresh frame lands before the model predicts")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit(
            "REACTOR_API_KEY is required (env var or --api-key). Create a key "
            "at https://reactor.inc/account/api-keys and export it in this shell."
        )

    from openpi_server.websocket_policy_server import WebsocketPolicyServer

    gateway = GatewayState(chunk_timeout_s=args.chunk_timeout)
    bridge = BridgeThread(
        gateway=gateway, api_key=args.api_key, api_url=args.api_url,
        model_name=args.model, settle_s=args.settle,
    )
    bridge.start()

    log = logging.getLogger("cosmos_droid")
    log.info("connecting to %s at %s", args.model, args.api_url)
    if not bridge.ready.wait(timeout=args.connect_timeout):
        bridge.stop()
        raise SystemExit(f"bridge did not become ready within {args.connect_timeout:.0f}s")
    if bridge.failed is not None:
        raise SystemExit(f"bridge failed to connect: {bridge.failed}")

    server = WebsocketPolicyServer(policy=gateway, host=args.host, port=args.port, metadata={})
    log.info("gateway serving openpi on %s:%d; point RoboLab's run.py here", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        bridge.stop()
        bridge.join(timeout=10.0)
        log.info("diagnostics: %s", gateway.diag)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _run(args)


if __name__ == "__main__":
    main()
