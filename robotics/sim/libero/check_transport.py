"""Native SDK lifecycle/episode ordering checks without LIBERO or a session."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from reactor_sdk import ReactorStatus

from libero_sim.bridge import Bridge
from libero_sim.contract import VIEWS


class NativeTrack:
    def __init__(self, reactor, name):
        self.reactor, self.name = reactor, name
        self.frames = []
        self.published = True

    def push_frame(self, frame):
        assert self.published and frame.dtype == np.uint8 and frame.shape[-1] == 3
        self.frames.append(frame.copy())

    def unpublish(self):
        self.published = False
        self.reactor.events.append(("unpublish", self.name))


class NativeReactor:
    def __init__(self, **kwargs):
        self.events, self.tracks = [], {}
        self.fail_publish = None
        self.status = ReactorStatus.DISCONNECTED
        self.executed = "old episode actions"
        self.reply_on = "reset"
        self.block_connect = False

    def on_status(self, status):
        return lambda handler: handler

    def on_message(self, handler):
        self.message = handler
        return handler

    def on_error(self, handler):
        return handler

    def get_status(self):
        return self.status

    async def connect(self):
        self.events.append(("connect",))
        if self.block_connect:
            await asyncio.Future()
        self.status = ReactorStatus.READY

    async def publish_track(self, name):
        if name == self.fail_publish:
            raise RuntimeError("test publish failure")
        self.events.append(("publish", name))
        self.tracks[name] = NativeTrack(self, name)
        return self.tracks[name]

    async def send_command(self, command, data):
        self.events.append((command, data))
        if command == "set_executed_action_json":
            self.executed = data["executed_action_json"]
        if command == "reset":
            assert self.executed == "", "reset consumed the previous episode's echo"
        if command == self.reply_on:
            reply = {"type": "action_prediction", "data": {"action": [[0.0] * 7] * 16, "step": 0}}
            # SDK 1.6 delivers the callback AND returns the correlated envelope.
            self.message(reply)
            return reply
        return None

    async def disconnect(self):
        assert all(not track.published for track in self.tracks.values())
        self.events.append(("disconnect",))
        self.status = ReactorStatus.DISCONNECTED

    def close(self):
        self.events.append(("close",))


class Rollout:
    def __init__(self):
        self.predictions = []
        self.episode_start = True
        self.diag = SimpleNamespace(echoes_sent=0)
        self.frame = np.zeros((4, 4, 3), np.uint8)
        self.seq = 0

    def frame_reader(self, name):
        return lambda: (self.frame, self.seq)

    def submit_chunk(self, prediction):
        self.predictions.append(prediction)
        self.episode_start = False

    def is_episode_start(self):
        return self.episode_start

    def take_pending_echo(self):
        return None

    def request_reset(self):
        self.episode_start = True


async def check():
    with patch("libero_sim.bridge.Reactor", NativeReactor):
        for reply_on in ("reset", "set_task_description"):
            rollout = Rollout()
            bridge = Bridge(rollout, api_key="unused", api_url="unused", task="pick")
            reactor = bridge.reactor
            reactor.reply_on = reply_on
            async with bridge:
                commands = reactor.events[1 + len(VIEWS):]
                assert commands == [
                    ("set_executed_action_json", {"executed_action_json": ""}),
                    ("reset", {}),
                    ("set_task_description", {"task_description": "pick"}),
                ], commands
                assert len(rollout.predictions) == 1, "early seed dropped or ingested twice"
                await asyncio.sleep(0.01)
                assert list(reactor.tracks) == list(VIEWS)
                assert all(len(track.frames) == 1 for track in reactor.tracks.values())
                rollout.seq += 1
                rollout.frame = np.full((4, 4, 3), 37, np.uint8)
                await asyncio.sleep(0.01)
                assert all(int(track.frames[-1][0, 0, 0]) == 37 for track in reactor.tracks.values())
                publishers = [track._task for track in bridge._tracks]
                pump = bridge._pump_task
                # Next episode must clear the old echo before reset as well.
                reactor.executed = "previous episode"
                await bridge.set_task("place")
                await asyncio.sleep(0.03)
                assert len(rollout.predictions) == 2
                assert reactor.events[-1] == ("set_task_description", {"task_description": "place"})
            assert all(task.done() for task in publishers) and pump.done()
            assert reactor.events[-2:] == [("disconnect",), ("close",)]

        bridge = Bridge(Rollout(), api_key="unused", api_url="unused", task="pick")
        bridge.reactor.fail_publish = list(VIEWS)[1]
        try:
            async with bridge:
                raise AssertionError("partial publication should fail")
        except RuntimeError:
            pass
        assert bridge.reactor.events[-2:] == [("disconnect",), ("close",)]
        assert not bridge._tracks

        # Cancellation while native connect is pending must release the handle.
        bridge = Bridge(Rollout(), api_key="unused", api_url="unused", task="pick")
        bridge.reactor.block_connect = True
        startup = asyncio.create_task(bridge.__aenter__())
        await asyncio.sleep(0)
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)
        assert bridge.reactor.events[-2:] == [("disconnect",), ("close",)]


asyncio.run(check())
print("TRANSPORT OK: native RGB lifecycle, clear/reset/task order, seeds handled once")
