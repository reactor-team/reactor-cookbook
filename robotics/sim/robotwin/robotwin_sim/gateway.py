# ──────────────────────────────────────────────────────────────────────────
# The gateway: this example's analog of the other examples' RolloutState,
# and the reason there is no env.py here at all.
#
# RoboTwin 2.0 is not a library this example wraps: the authors ship an
# evaluation client that owns the episode loop, the seeds, the expert
# feasibility checks, the instruction sampling and the success predicates,
# and its seam for a policy is a broker frontend port speaking pickled
# dicts. So this example IS that port. Their client connects to it exactly
# as it would to their own policy broker, and every request is relayed to an
# xwam model served on Reactor.
#
# Keeping their client completely unmodified is the only thing that makes a
# success rate measured here comparable to the one in their paper. The
# substituted pieces are transport and serving, nothing else.
#
# Threading: none. The authors' client is lock-step (one request out, blocks
# for the reply) and the model is single-inference-in-flight, so a single
# asyncio loop running a zmq.asyncio ROUTER socket alongside the aiortc
# session is the whole design: no hand-off, no lock. The simulator is out
# of process, which is what buys that simplicity.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import pickle
from typing import Awaitable, Callable, Protocol

import zmq
import zmq.asyncio

from .contract import decode_request, encode_reply

log = logging.getLogger("robotwin_sim.gateway")

#: The port the authors' client expects the broker frontend on
#: (``--server_port``). Bind it and their client needs no change.
DEFAULT_PORT = 10086


class Predictor(Protocol):
    """What the gateway needs from the Reactor side; see bridge.Bridge."""

    async def predict(self, request): ...  # -> (actions, proprios)


class Gateway:
    """Binds the authors' broker frontend port and answers their protocol."""

    def __init__(self, predict: Callable[..., Awaitable]) -> None:
        self._predict = predict
        #: Requests answered, for the shutdown line.
        self.handled = 0

    async def handle(self, payload: bytes) -> bytes:
        """One pickled request in, one pickled reply out.

        Kept separate from the socket so it can be exercised offline; see
        ``check_wiring.py``.
        """
        request = decode_request(pickle.loads(payload))
        actions, proprios = await self._predict(request)
        self.handled += 1
        return pickle.dumps(encode_reply(actions, proprios))

    async def serve(self, port: int = DEFAULT_PORT) -> None:
        """Serve forever on ``tcp://*:port``."""
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.ROUTER)
        sock.bind(f"tcp://*:{port}")
        log.info(
            "listening on tcp://*:%d; point the authors' client here with "
            "--server_port %d",
            port,
            port,
        )
        try:
            while True:
                frames = await sock.recv_multipart()
                # DEALER->ROUTER delivers [identity, payload]; REQ->ROUTER adds
                # an empty delimiter frame: [identity, b"", payload]. Relay
                # whatever envelope arrived so either client socket type works.
                identity, payload = frames[0], frames[-1]
                envelope = frames[1:-1]
                try:
                    reply = await self.handle(payload)
                except Exception as exc:
                    # A raised exception here would strand the client blocking
                    # on recv() forever. There is no error frame in the
                    # authors' protocol, so log loudly and let their own
                    # timeout end the episode as a failure.
                    log.exception("request failed: %s", exc)
                    continue
                await sock.send_multipart([identity, *envelope, reply])
        finally:
            sock.close(linger=0)
            ctx.term()
            log.info("gateway stopped after %d request(s)", self.handled)
