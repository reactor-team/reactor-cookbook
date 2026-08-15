# ──────────────────────────────────────────────────────────────────────────
# The openpi msgpack-WebSocket policy-server protocol: the seam RoboLab
# connects to.
#
# As spoken by RoboLab's own DreamZero client:
#
# - On connect the SERVER sends one packed config dict. The client stores it
#   as its server metadata and will hang on connect if it never arrives.
# - Each request is a packed dict with an `endpoint` key:
#     endpoint="infer"  -> observation payload; reply with the packed action
#                          (this gateway sends {"actions": (24, 8) float32}).
#     endpoint="reset"  -> {"session_ids": [...] | None}; reply with anything
#                          non-empty. The client ignores the body but blocks
#                          until something arrives.
# - On exception the server sends the traceback as a TEXT frame and closes;
#   the client turns a text reply into an exception. That is far easier to
#   debug than a bare disconnect, so it is worth preserving exactly.
#
# `infer` and `reset` may be coroutines here: this gateway is async all the
# way down (one event loop runs the WebSocket server and the WebRTC session
# together), unlike cosmos_droid_sim, which uses the packaged openpi server
# on its main thread and puts the Reactor side on a second thread.
#
# Included here rather than taken as a dependency so this package installs
# with plain `websockets` + `msgpack`; see README.md "Why the protocol
# server is in this package". Derived from the openpi project (Physical
# Intelligence), Apache-2.0.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import traceback
from typing import Any, Optional

import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy

log = logging.getLogger("dreamzero_sim.policy_server")


@dataclasses.dataclass
class PolicyServerConfig:
    """The handshake that tells RoboLab what to send.

    These defaults are DreamZero-DROID's: 180x320 frames, two exterior
    cameras plus a wrist camera, session ids on (the model is stateful, so the
    server has to know when the episode changed), joint-position action space.
    """

    image_resolution: Optional[tuple[int, int]] = (180, 320)
    needs_wrist_camera: bool = True
    n_external_cameras: int = 2
    needs_stereo_camera: bool = False
    needs_session_id: bool = True
    action_space: str = "joint_position"


async def _maybe_await(value: Any) -> Any:
    """Await *value* when it is awaitable, else return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


class WebsocketPolicyServer:
    """Serve one policy over the openpi msgpack-WebSocket protocol.

    *policy* must expose ``infer(obs) -> dict | ndarray`` and
    ``reset(reset_info) -> None``; either may be async.
    """

    def __init__(
        self,
        policy: Any,
        server_config: PolicyServerConfig | None = None,
        host: str = "0.0.0.0",
        port: int = 5000,
    ) -> None:
        self._policy = policy
        self._server_config = server_config or PolicyServerConfig()
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Run the server on a fresh event loop (blocking)."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Serve until cancelled, on the caller's event loop."""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            # compression=None / max_size=None: the frames are raw uint8
            # images. Deflate wastes CPU on them, and the default 1 MiB cap is
            # far too small for a three-camera observation.
            compression=None,
            max_size=None,
        ) as server:
            log.info("listening on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(
        self, websocket: websockets.asyncio.server.ServerConnection
    ) -> None:
        log.info("connection from %s opened", websocket.remote_address)

        # Handshake first: the client blocks on this before its first query.
        await websocket.send(
            msgpack_numpy.packb(dataclasses.asdict(self._server_config))
        )

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                endpoint = obs.pop("endpoint", "infer")
                if endpoint == "reset":
                    await _maybe_await(self._policy.reset(obs))
                    reply: Any = "reset successful"
                else:
                    action = await _maybe_await(self._policy.infer(obs))
                    reply = msgpack_numpy.packb(action)
                await websocket.send(reply)
            except websockets.ConnectionClosed:
                log.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                # Hand the client the traceback before closing.
                tb = traceback.format_exc()
                log.error("request failed:\n%s", tb)
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="internal server error; traceback in previous frame",
                )
                raise
