"""Offline SDK integration and protocol regressions; no session is created."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from reactor_sdk import ReactorStatus, Track, TrackDirection, TrackKind

from cosmos_droid_sim.bridge import Bridge
from cosmos_droid_sim.contract import CMD_SET_EXECUTED_STEP, CMD_SET_PROPRIO, TRACKS
from cosmos_droid_sim.gateway import GatewayRequest, GatewayState
from cosmos_droid_sim.tracks import CameraTrack


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bridge = Bridge(GatewayState(), api_key="offline-test", settle_s=0)
        self.bridge.reactor.get_status = Mock(return_value=ReactorStatus.READY)
        self.action = np.ones((32, 8))

    def reply(self, step):
        return {"type": "action_prediction", "data": {"step": step, "action": self.action.tolist()}}

    def request(self):
        return GatewayRequest(frames={}, proprio_json="{}", task="pick up")

    async def test_immediate_first_unsolicited_reply_is_not_drained(self):
        async def command(name, data):
            if name == CMD_SET_PROPRIO:
                # The server replies while send_command is still awaiting ACK.
                self.bridge._accept_chunk(self.reply(1))
                await asyncio.sleep(0)
            return None

        self.bridge.reactor.send_command = AsyncMock(side_effect=command)
        self.bridge._accept_chunk(self.reply(0))
        req = self.request()
        await asyncio.wait_for(self.bridge._serve_one(req), 0.5)
        self.assertTrue(req.done.is_set())
        self.assertEqual(req.step, 1)
        np.testing.assert_array_equal(req.action, self.action)
        self.assertNotIn(CMD_SET_EXECUTED_STEP, [c.args[0] for c in self.bridge.reactor.send_command.call_args_list])

    async def test_correlated_first_reply_and_next_execution_echo(self):
        async def command(name, data):
            if name == CMD_SET_PROPRIO and self.bridge._last_step is None:
                message = self.reply(1)
            elif name == CMD_SET_EXECUTED_STEP:
                message = self.reply(2)
            else:
                return None
            # Native SDK 1.6 dispatches this message as well as returning it.
            self.bridge._accept_chunk(message)
            return message

        self.bridge.reactor.send_command = AsyncMock(side_effect=command)
        first, second = self.request(), self.request()
        await asyncio.wait_for(self.bridge._serve_one(first), 0.5)
        self.assertTrue(self.bridge._chunks.empty(), "correlated reply was enqueued twice")
        await asyncio.wait_for(self.bridge._serve_one(second), 0.5)
        self.assertTrue(self.bridge._chunks.empty(), "correlated reply was enqueued twice")
        self.assertEqual((first.step, second.step), (1, 2))
        echo = self.bridge.reactor.send_command.call_args
        self.assertEqual(echo.args[0], CMD_SET_EXECUTED_STEP)
        import json
        payload = json.loads(echo.args[1]["executed_step_json"])
        self.assertEqual(payload["step"], 1)
        np.testing.assert_array_equal(payload["action"], self.action)

    async def test_native_sdk_publish_and_rgb_pump_then_cleanup(self):
        reactor = self.bridge.reactor
        # Keep real Reactor.publish_track and Track.push_frame. Only the
        # network operation and final native encoder boundary are substituted.
        reactor.connect = AsyncMock()
        reactor.disconnect = AsyncMock()
        reactor._require_handle = Mock()
        reactor._async_op = AsyncMock(return_value=None)
        reactor._push_video_frame = Mock()
        reactor.close = Mock(wraps=reactor.close)
        reactor._tracks = {name: Track(reactor, name, TrackKind.VIDEO, TrackDirection.SENDONLY) for name in TRACKS}
        async with self.bridge:
            await asyncio.sleep(0.02)
            self.assertEqual(reactor._async_op.await_count, 3)
            self.assertEqual({c.args[0] for c in reactor._push_video_frame.call_args_list}, set(TRACKS))
            tasks = [*self.bridge._pump_tasks, self.bridge._relay_task]
        self.assertTrue(all(t.done() for t in tasks))
        reactor.disconnect.assert_awaited_once_with()
        reactor.close.assert_called_once_with()

    async def test_rgb_pixels_request_pacing_and_heartbeat(self):
        frame = np.zeros((2, 4, 3), np.uint8)
        frame[:] = [7, 31, 191]
        reactor = self.bridge.reactor
        track = Track(reactor, "wrist_view", TrackKind.VIDEO, TrackDirection.SENDONLY)
        track._published = True
        reactor._push_video_frame = Mock()
        state = [frame, 1]
        source = CameraTrack("wrist_view", lambda: tuple(state))
        with patch("cosmos_droid_sim.tracks.HEARTBEAT_S", 0.02):
            task = asyncio.create_task(source.pump(track))
            try:
                await asyncio.sleep(0.01)
                self.assertEqual(reactor._push_video_frame.call_count, 1)
                state[1] = 2
                await asyncio.sleep(0.01)
                self.assertEqual(reactor._push_video_frame.call_count, 2)
                await asyncio.sleep(0.025)
                self.assertGreaterEqual(reactor._push_video_frame.call_count, 3)
                call = reactor._push_video_frame.call_args
                self.assertEqual(call.args[2:4], (4, 2))
                self.assertEqual(bytes(call.args[1])[:4], bytes([191, 31, 7, 255]))
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_closing_relay_unblocks_active_gateway_request(self):
        req = self.request()
        self.bridge._gateway.take_pending = Mock(return_value=req)
        started = asyncio.Event()

        async def waiting(request):
            started.set()
            await asyncio.Event().wait()

        self.bridge._serve_one = waiting
        self.bridge.reactor.disconnect = AsyncMock()
        self.bridge.reactor.close = Mock()
        self.bridge._relay_task = asyncio.create_task(self.bridge._relay())
        await asyncio.wait_for(started.wait(), 0.5)
        await self.bridge.close()
        self.assertTrue(req.done.is_set())
        self.assertIn("bridge closed", req.error)

    async def test_partial_publish_failure_releases_resources(self):
        reactor = self.bridge.reactor
        reactor.connect = AsyncMock()
        reactor.publish_track = AsyncMock(side_effect=[Mock(), RuntimeError("publish failed")])
        reactor.disconnect = AsyncMock()
        reactor.close = Mock()
        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            await self.bridge.__aenter__()
        reactor.disconnect.assert_awaited_once_with()
        reactor.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
