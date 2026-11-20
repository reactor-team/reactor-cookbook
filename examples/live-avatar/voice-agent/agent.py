"""The voice agent: an OpenAI Realtime conversation, spoken through the avatar in the room.

The page's server starts each call: it creates the room, dispatches this agent into it, starts the avatar model's
session, which joins as ``avatar`` publishing on behalf of this agent, and only then lets the user in. This worker holds
the conversation and calls nothing else: no Reactor key, no runtime URL.

:class:`RoomAvatar` is the agent's side of LiveKit's avatar protocol, in the shape of LiveKit's avatar plugins minus
the provider call. It routes the agent's speech to the avatar with ``DataStreamAudioOutput`` (byte streams on
``lk.audio_stream``, interrupts as the ``lk.clear_buffer`` RPC, playback reports back as ``lk.playback_started`` and
``lk.playback_finished``), so the user hears the agent only through the avatar, lip-synced. It also forwards the user's
speaking state on ``avatar.user_speaking_state``. When the call ends the agent deletes the room, so the avatar leaves
even if the page could not stop the call.

Run it from this folder with ``.env`` filled in (see ``.env.example``)::

    uv run python agent.py dev

``AVATAR_AGENT_NAME`` makes the worker explicit-dispatch only: it joins only the rooms the page's server dispatches it
into. The dispatch's metadata may carry ``voice`` (an OpenAI Realtime voice) and ``instructions`` (the system prompt).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.voice.avatar import AvatarSession, DataStreamAudioOutput
from livekit.agents.voice.events import UserStateChangedEvent
from livekit.plugins import openai

load_dotenv(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("avatar-agent")

#: The identity the page's server mints the avatar's token with.
AVATAR_IDENTITY = "avatar"
#: The rate of the speech the agent streams to the avatar.
SAMPLE_RATE = 24_000

DEFAULT_INSTRUCTIONS = (
    "You are a friendly presenter on a video call. Always speak English, unless the user explicitly asks "
    "you to switch language. Keep answers short and conversational, two or three sentences at most."
)
GREETING = "Greet the user in one short sentence."


class AvatarAudioOutput(DataStreamAudioOutput):
    """``DataStreamAudioOutput`` that copes with the avatar leaving the room.

    An utterance the avatar never reports finished would hold the agent's playout, and ``lk.clear_buffer`` to a
    participant that has left fails. When the avatar leaves, or is gone when the buffer is cleared, the utterance in
    flight is marked finished (interrupted) here instead.
    """

    def __init__(self, room: rtc.Room, **kwargs: Any) -> None:
        super().__init__(room, **kwargs)
        self._avatar_room = room
        room.on("participant_disconnected", self._on_participant_left)

    def _avatar_here(self) -> bool:
        return any(
            p.identity == self._destination_identity
            for p in self._avatar_room.remote_participants.values()
        )

    def _finish_locally(self) -> None:
        if self._clear_buffer_timeout_handler:
            self._clear_buffer_timeout_handler.cancel()
            self._clear_buffer_timeout_handler = None
        if self._pushed_duration > 0:
            self._pushed_duration = 0.0
            self.on_playback_finished(playback_position=0.0, interrupted=True)

    def _on_participant_left(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._destination_identity:
            self._finish_locally()

    def clear_buffer(self) -> None:
        if self._started and not self._avatar_here():
            self._finish_locally()
            return
        super().clear_buffer()


class RoomAvatar(AvatarSession):
    """The agent's side of the avatar the page's server started in this room before the user joined."""

    def __init__(self) -> None:
        super().__init__()
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def avatar_identity(self) -> str:
        return AVATAR_IDENTITY

    @property
    def provider(self) -> str:
        return "reactor"

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:  # type: ignore[override]
        await super().start(agent_session, room)
        agent_session.output.replace_audio_tail(
            AvatarAudioOutput(
                room=room,
                destination_identity=AVATAR_IDENTITY,
                sample_rate=SAMPLE_RATE,
                # Speech waits for the avatar's video track, and playback counts as started when the avatar says so.
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                wait_playback_start=True,
            )
        )
        agent_session.on(
            "user_state_changed", lambda ev: self._forward_user_state(room, ev)
        )

    def _forward_user_state(self, room: rtc.Room, ev: UserStateChangedEvent) -> None:
        payload = json.dumps(
            {"state": "speaking" if ev.new_state == "speaking" else "idle"}
        )
        task = asyncio.get_running_loop().create_task(
            room.local_participant.publish_data(
                payload,
                reliable=True,
                topic="avatar.user_speaking_state",
                destination_identities=[AVATAR_IDENTITY],
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def dispatch_setup(metadata: str) -> tuple[str, str]:
    """The voice and the instructions from the dispatch's metadata, or the defaults."""
    try:
        data = json.loads(metadata or "{}")
    except ValueError:
        data = {}
    voice = data.get("voice") if isinstance(data.get("voice"), str) else None
    instructions = (
        data.get("instructions") if isinstance(data.get("instructions"), str) else None
    )
    return voice or os.environ.get(
        "AVATAR_AGENT_VOICE"
    ) or "marin", instructions or DEFAULT_INSTRUCTIONS


server = AgentServer()


@server.rtc_session(agent_name=os.environ.get("AVATAR_AGENT_NAME", ""))
async def entrypoint(ctx: JobContext) -> None:
    async def close_room() -> None:
        await ctx.delete_room()

    ctx.add_shutdown_callback(close_room)
    await ctx.connect()
    voice, instructions = dispatch_setup(ctx.job.metadata)
    await ctx.wait_for_participant()
    logger.info("conversation set up", extra={"voice": voice})
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice=voice),
        # The avatar cannot pause its speech, only drop it.
        turn_handling={"interruption": {"resume_false_interruption": False}},
    )
    avatar = RoomAvatar()
    try:
        await avatar.start(session, room=ctx.room)
        await avatar.wait_for_join(timeout=60)
        await session.start(agent=Agent(instructions=instructions), room=ctx.room)
    except Exception:
        logger.exception("the call setup failed")
        ctx.shutdown(reason="the call could not start")
        return
    session.on(
        "close",
        lambda ev: ctx.shutdown(
            reason=f"the agent session closed: {getattr(ev, 'reason', '')}"
        ),
    )
    session.generate_reply(instructions=GREETING)


if __name__ == "__main__":
    cli.run_app(server)
