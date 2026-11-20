"""Native publisher lifecycle and temporal-window checks; no model or GPU."""
import asyncio
from unittest.mock import patch

import numpy as np

from dreamzero_sim.bridge import Bridge
from dreamzero_sim.contract import ACTION_SHAPE, TRACKS


class NativeTrack:
    def __init__(self, reactor, name):
        self.reactor, self.name = reactor, name
        self.frames = []
        self.published = True

    def push_frame(self, frame):
        assert self.published
        assert frame.dtype == np.uint8 and frame.shape[-1] == 3
        self.frames.append(frame.copy())
        if self.name == TRACKS[-1] and self.reactor.predicting:
            self.reactor.message({"type": "action_chunk", "data": {
                "actions": np.zeros(ACTION_SHAPE).tolist(), "obs_seq": 1,
                "chunk_index": 0, "inference_seconds": 0.1,
            }})

    def unpublish(self):
        self.published = False
        self.reactor.events.append(("unpublish", self.name))


class NativeReactor:
    def __init__(self, *args, **kwargs):
        self.events, self.tracks = [], {}
        self.predicting = False
        self.fail_publish = None
        self.fail_connect = False

    def on_status(self, handler):
        return handler

    def on_message(self, handler):
        self.message = handler
        return handler

    def on_error(self, handler):
        return handler

    async def connect(self):
        self.events.append(("connect",))
        if self.fail_connect:
            raise ConnectionError("test connection failure")
        # Native connect resolves READY without waiting for status callbacks.

    async def publish_track(self, name):
        if name == self.fail_publish:
            raise RuntimeError("test publish failure")
        self.events.append(("publish", name))
        self.tracks[name] = NativeTrack(self, name)
        return self.tracks[name]

    async def send_command(self, command, data):
        self.events.append((command, data))
        if command == "set_prompt":
            self.predicting = True
            reply = {"type": "episode_started", "data": {}}
            self.message(reply)
            return reply
        return None

    async def disconnect(self):
        assert all(not t.published for t in self.tracks.values())
        self.events.append(("disconnect",))

    def close(self):
        self.events.append(("close",))


async def check():
    with patch("reactor_sdk.Reactor", NativeReactor):
        bridge = Bridge(api_key="unused", prime_stagger_s=0)
        await bridge.ensure_connected()
        reactor = bridge._reactor
        assert list(reactor.tracks) == list(TRACKS)
        for n in range(4):
            await bridge._push({name: np.full((4, 4, 3), n, np.uint8) for name in TRACKS})
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        for track in reactor.tracks.values():
            assert [int(frame[0, 0, 0]) for frame in track.frames] == [0, 1, 2, 3]
        # A prompt reply can arrive synchronously before frames are pushed.
        frames = {name: np.ones((4, 4, 3), np.uint8) for name in TRACKS}
        result = await asyncio.wait_for(bridge.predict(frames, [0.0] * 7, 0.0, "pick"), 1)
        assert result.shape == ACTION_SHAPE
        assert bridge.diag.chunks_returned == 1 and bridge._chunks.empty()
        publishers = [track._task for track in bridge._tracks.values()]
        await bridge.close()
        assert all(task.done() for task in publishers)
        assert reactor.events[-2:] == [("disconnect",), ("close",)]
        assert not any(event[0] == "ping" for event in reactor.events)
        await bridge.close()

        # Failure halfway through publication releases the tracks already started.
        bridge = Bridge(api_key="unused", prime_stagger_s=0)
        bridge._reactor.fail_publish = TRACKS[1]
        try:
            await bridge.ensure_connected()
        except RuntimeError:
            pass
        else:
            raise AssertionError("publish failure swallowed")
        assert not bridge.is_connected
        assert bridge._reactor.events[-2:] == [("disconnect",), ("close",)]
        assert all(track._task is None for track in bridge._tracks.values())

        bridge = Bridge(api_key="unused")
        bridge._reactor.fail_connect = True
        try:
            await bridge.ensure_connected()
        except ConnectionError:
            pass
        else:
            raise AssertionError("connect failure swallowed")
        assert bridge._reactor.events[-1] == ("close",)


asyncio.run(check())
print("TRANSPORT OK: native RGB publishing, real temporal windows, early replies, cleanup")
