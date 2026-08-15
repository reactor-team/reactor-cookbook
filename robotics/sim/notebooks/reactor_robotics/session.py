# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""One live model session: connect, publish named video tracks, stay alive.

This module manages the connection lifecycle (handler registration,
readiness, keepalive) so the scripts don't have to. Three orderings
matter:

1. Handlers are registered before ``connect()``, because ``READY`` can arrive
   before the first ``await`` after ``connect()`` returns.
2. Tracks are published only after ``READY``. The runtime creates no track
   for an earlier ``publish_track``, and the model then waits for frames that
   never arrive. Status goes ``CONNECTING`` -> ``WAITING`` -> ``READY``
   asynchronously after ``connect()``; we await the event.
3. A ``ping`` goes out every 10 s for the life of the session, including
   while the caller sits in ``predict()``. The runtime disconnects a client
   that sends nothing for 20 s, and ``reactor-sdk==0.8.0`` leaves keepalive
   to the client.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable, Sequence

import numpy as np

from .track import RepeatingFrameTrack

#: The production API, where the served models live. ``REACTOR_API_URL`` is
#: kept as an escape hatch for pointing a script at another deployment.
DEFAULT_API_URL = "https://api.reactor.inc"

#: The runtime's watchdog fires at 20 s of client silence; ping at half that.
PING_INTERVAL_S = 10.0

log = logging.getLogger("reactor_robotics.session")


def api_url() -> str:
    """Resolve the API URL: ``REACTOR_API_URL`` if set, else PROD."""
    return os.environ.get("REACTOR_API_URL") or DEFAULT_API_URL


def require_api_key() -> str:
    """Return ``REACTOR_API_KEY``, or raise with an actionable message.

    Never logs, prints, or embeds the value. The caller gets the string and
    hands it straight to the SDK, which mints the session JWT in-process.
    """
    key = os.environ.get("REACTOR_API_KEY")
    if not key:
        raise RuntimeError(
            "REACTOR_API_KEY is not set.\n"
            "\n"
            "Create an API key at https://reactor.inc/account/api-keys, then\n"
            "export it in the shell that runs this script (NOT in the script\n"
            "itself; a literal key in a committed file is a leaked key):\n"
            "\n"
            "    export REACTOR_API_KEY=<your key>\n"
            "    uv run python <model>_quickstart.py\n"
            "\n"
            "To point at a non-default deployment, also set REACTOR_API_URL\n"
            f"(default: {DEFAULT_API_URL})."
        )
    return key


def describe_api_key() -> str:
    """A one-line confirmation that the key is present, revealing nothing.

    Reports only the length: run output gets pasted around, and even a
    short prefix of a credential is a credential fragment.
    """
    key = require_api_key()
    return (
        f"REACTOR_API_KEY is set ({len(key)} characters; value not shown).\n"
        f"REACTOR_API_URL -> {api_url()}"
    )


class ReactorSession:
    """A connected model session with named video tracks and a keepalive.

    Usage::

        session = ReactorSession("xwam")
        await session.connect(["head_view", "left_wrist_view"])
        session.track("head_view").set_frame(rgb_uint8)
        await session.send("set_state_json", {"state_json": "..."})
        chunk = await session.next_message("action_prediction", timeout_s=20)
        await session.close()

    Subscribe to a reply type with :meth:`subscribe` *before* :meth:`connect`
    if you need to be sure no early message is dropped; :meth:`next_message`
    subscribes lazily otherwise.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_url_: str | None = None,
        fps: int = 15,
        frame_size: tuple[int, int] = (240, 320),
        ping_interval_s: float = PING_INTERVAL_S,
    ) -> None:
        from reactor_sdk import Reactor

        self.model = model
        self.api_url = api_url_ or api_url()
        self.fps = fps
        self.frame_size = frame_size
        self._ping_interval_s = ping_interval_s

        self._reactor = Reactor(
            model, api_key=api_key or require_api_key(), api_url=self.api_url
        )
        self.tracks: dict[str, RepeatingFrameTrack] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._ready = asyncio.Event()
        self._keepalive_task: asyncio.Task | None = None
        self._connected = False
        #: Handlers are registered on the SDK object, which outlives a failed
        #: connect(). Registering them twice would deliver every reply twice,
        #: so a retried connect() must not re-register.
        self._handlers_registered = False
        #: Status transitions, in order. The scripts print this.
        self.status_log: list[str] = []
        #: Message types seen that nobody subscribed to. Non-empty at the end
        #: of a session usually means a protocol assumption is wrong.
        self.unclaimed: list[str] = []

    # ---------------------------------------------------------------- wiring

    def subscribe(self, msg_type: str) -> asyncio.Queue:
        """Get (creating if needed) the queue for one reply ``type``."""
        return self._queues.setdefault(msg_type, asyncio.Queue())

    def track(self, name: str) -> RepeatingFrameTrack:
        return self.tracks[name]

    async def connect(
        self,
        track_names: Sequence[str],
        *,
        subscribe: Iterable[str] = (),
        ready_timeout_s: float = 120.0,
    ) -> None:
        """Register handlers, connect, await READY, publish tracks, start ping.

        The order of the first three is the whole point of this method; see
        the module docstring.
        """
        from reactor_sdk import ReactorStatus

        for msg_type in subscribe:
            self.subscribe(msg_type)

        # (1) Handlers FIRST, before connect() can deliver anything. Exactly
        # once, even if an earlier connect() failed and the caller is retrying
        # (a capacity 429 is a normal retry path for a multi-GPU model).
        if self._handlers_registered:
            await self._finish_connect(track_names, ready_timeout_s)
            return
        self._handlers_registered = True

        @self._reactor.on_status
        def _on_status(status) -> None:  # pragma: no cover - network callback
            name = getattr(status, "name", str(status))
            self.status_log.append(name)
            log.info("status: %s", name)
            if status == ReactorStatus.READY:
                self._ready.set()

        @self._reactor.on_message
        def _on_message(msg) -> None:  # pragma: no cover - network callback
            if not isinstance(msg, dict):
                return
            # Wire envelope: {"type": "<name>", "data": {...}}.
            msg_type = msg.get("type")
            queue = self._queues.get(msg_type)
            if queue is None:
                if msg_type not in self.unclaimed:
                    self.unclaimed.append(msg_type)
                log.debug("no subscriber for message type %r", msg_type)
                return
            queue.put_nowait(msg.get("data") or {})

        @self._reactor.on_error
        def _on_error(err) -> None:  # pragma: no cover - network callback
            log.error("session error: %s", err)

        await self._finish_connect(track_names, ready_timeout_s)

    async def _finish_connect(
        self, track_names: Sequence[str], ready_timeout_s: float
    ) -> None:
        """Steps (2) and (3): connect, await READY, publish tracks, ping.

        Split out from :meth:`connect` so a retried attempt re-runs only this
        part and never re-registers handlers.
        """
        self._ready.clear()
        await self._reactor.connect()
        self._connected = True

        # (2) READY before publish_track; an earlier publish creates no track.
        await asyncio.wait_for(self._ready.wait(), timeout=ready_timeout_s)

        for name in track_names:
            # Reuse an existing track object across a reconnect so a caller
            # holding a reference keeps pushing to the live one.
            self.tracks.setdefault(
                name,
                RepeatingFrameTrack(name, fps=self.fps, size=self.frame_size),
            )
            await self._reactor.publish_track(name, self.tracks[name])
        log.info(
            "connected to %s at %s; tracks published: %s",
            self.model,
            self.api_url,
            ", ".join(track_names),
        )

        # (3) Keepalive, or the runtime drops us after 20 s of quiet.
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive())

    async def _keepalive(self) -> None:
        from reactor_sdk.types import MessageScope

        while True:
            try:
                await self._reactor.send_command("ping", {}, scope=MessageScope.RUNTIME)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - transport hiccup
                log.warning("keepalive ping failed", exc_info=True)
            await asyncio.sleep(self._ping_interval_s)

    # -------------------------------------------------------------- messages

    async def send(self, command: str, payload: dict | None = None) -> None:
        await self._reactor.send_command(command, payload or {})

    async def next_message(self, msg_type: str, *, timeout_s: float) -> dict:
        """Await the next ``data`` payload of one message type."""
        return await asyncio.wait_for(
            self.subscribe(msg_type).get(), timeout=timeout_s
        )

    def drain(self, msg_type: str) -> list[dict]:
        """Take every already-queued payload of one type without blocking."""
        queue = self.subscribe(msg_type)
        out: list[dict] = []
        while True:
            try:
                out.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                return out

    def set_frames(self, frames: dict[str, np.ndarray]) -> None:
        """Replace the repeating frame on each named track.

        Raises on an unknown track name: the server accepts a wrist frame
        published on the head track, so a typo has to fail here instead.
        """
        unknown = sorted(set(frames) - set(self.tracks))
        if unknown:
            raise KeyError(
                f"no such track(s): {unknown}; published tracks are "
                f"{sorted(self.tracks)}"
            )
        missing = sorted(set(self.tracks) - set(frames))
        if missing:
            raise KeyError(
                f"missing frames for track(s): {missing}. Every view must get "
                "a frame; the model waits for all of them before answering."
            )
        for name, frame in frames.items():
            self.tracks[name].set_frame(frame)

    # ----------------------------------------------------------------- close

    async def close(self) -> None:
        """Stop the keepalive and disconnect. Safe to call twice.

        Always call this: a live session holds a real GPU worker.
        """
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive_task = None
        if self._connected:
            try:
                await self._reactor.disconnect()
            except Exception:  # pragma: no cover - best-effort teardown
                log.warning("disconnect() failed during close", exc_info=True)
            self._connected = False
        log.info("session closed")
