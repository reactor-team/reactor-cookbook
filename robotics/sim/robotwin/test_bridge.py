"""Offline SDK integration and request/retry lifecycle regressions."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np
from reactor_sdk import Track, TrackDirection, TrackKind

from robotwin_sim.bridge import Bridge
from robotwin_sim.contract import (
    ACTION_SHAPE, CMD_SET_STATE, FIELD_STATE, PROPRIO_DIM, PROPRIO_PRED_SHAPE,
    VIEWS, decode_request,
)


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bridge = Bridge(api_key="offline-test", settle_s=0, timeout_s=0.03, retries=1)
        self.req = decode_request({
            "video": np.zeros((3, 240, 320, 3), np.float32),
            "proprios": np.zeros(PROPRIO_DIM), "prompt": ["pick up"],
            "env_rank": 2, "rollout_id": 3, "step_id": 4,
        })

    def reply(self, step):
        return {"type": "action_prediction", "data": {
            "step": step, "actions": np.ones(ACTION_SHAPE).tolist(),
            "proprios": np.ones(PROPRIO_PRED_SHAPE).tolist(),
        }}

    async def test_correlated_reply_callback_is_not_enqueued_twice(self):
        async def command(name, data):
            if name == CMD_SET_STATE:
                message = self.reply(1)
                self.bridge._accept_reply(message)
                return message
            return None

        self.bridge._reactor.send_command = AsyncMock(side_effect=command)
        actions, proprios = await self.bridge.predict(self.req)
        self.assertEqual(actions.shape, ACTION_SHAPE)
        self.assertEqual(proprios.shape, PROPRIO_PRED_SHAPE)
        self.assertEqual(self.bridge.diag.retries, 0)
        self.assertTrue(self.bridge._replies.empty(), "correlated reply was enqueued twice")

    async def test_unsolicited_reply_and_stale_reply_filter(self):
        async def command(name, data):
            if name == CMD_SET_STATE:
                self.bridge._accept_reply(self.reply(0))
                self.bridge._accept_reply(self.reply(1))
            return None

        self.bridge._reactor.send_command = AsyncMock(side_effect=command)
        await self.bridge.predict(self.req)
        self.assertEqual(self.bridge.diag.stale_replies, [0])
        self.assertEqual(self.bridge.diag.replies, 1)

    async def test_retry_changes_request_but_preserves_seeds(self):
        sent = []

        async def command(name, data):
            if name == CMD_SET_STATE:
                sent.append(data[FIELD_STATE])
                if len(sent) == 2:
                    message = self.reply(1)
                    self.bridge._accept_reply(message)
                    return message
            return None

        self.bridge._reactor.send_command = AsyncMock(side_effect=command)
        await self.bridge.predict(self.req)
        self.assertEqual(self.bridge.diag.retries, 1)
        first, second = map(json.loads, sent)
        self.assertEqual(second.pop("retry"), 1)
        self.assertEqual(first, second)
        self.assertEqual((first["env_rank"], first["rollout_id"], first["step_id"]), (2, 3, 4))

    async def test_command_wait_is_bounded_by_retry_timeout(self):
        attempts = 0

        async def command(name, data):
            nonlocal attempts
            if name == CMD_SET_STATE:
                attempts += 1
                await asyncio.Event().wait()

        self.bridge._reactor.send_command = AsyncMock(side_effect=command)
        with self.assertRaisesRegex(TimeoutError, "after 2 attempts"):
            await asyncio.wait_for(self.bridge.predict(self.req), 0.5)
        self.assertEqual(attempts, 2)

    async def test_native_sdk_publish_rgb_repetition_and_cleanup(self):
        reactor = self.bridge._reactor
        # Exercise the installed SDK's public publishing and RGB conversion;
        # replace only network completion and the native encoder boundary.
        reactor.connect = AsyncMock()
        reactor.disconnect = AsyncMock()
        reactor.send_command = AsyncMock()
        reactor._require_handle = Mock()
        reactor._async_op = AsyncMock(return_value=None)
        reactor._push_video_frame = Mock()
        reactor.close = Mock(wraps=reactor.close)
        reactor._tracks = {view: Track(reactor, view, TrackKind.VIDEO, TrackDirection.SENDONLY) for view in VIEWS}
        async with self.bridge:
            rgb = np.full((2, 4, 3), [7, 31, 191], dtype=np.uint8)
            for source in self.bridge._tracks.values():
                source.set_frame(rgb)
            await asyncio.sleep(0.09)
            self.assertEqual(reactor._async_op.await_count, 3)
            calls = reactor._push_video_frame.call_args_list
            for view in VIEWS:
                view_calls = [c for c in calls if c.args[0] == view]
                self.assertGreaterEqual(len(view_calls), 2)
                self.assertEqual(view_calls[-1].args[2:4], (4, 2))
                self.assertEqual(bytes(view_calls[-1].args[1])[:4], bytes([191, 31, 7, 255]))
            tasks = list(self.bridge._pump_tasks)
            reactor.send_command.assert_not_called()  # SDK owns keepalive.
        self.assertTrue(all(t.done() for t in tasks))
        reactor.disconnect.assert_awaited_once_with()
        reactor.close.assert_called_once_with()

    async def test_partial_publish_failure_cleans_up(self):
        reactor = self.bridge._reactor
        reactor.connect = AsyncMock()
        reactor.publish_track = AsyncMock(side_effect=[Mock(), RuntimeError("publish failed")])
        reactor.disconnect = AsyncMock()
        reactor.close = Mock()
        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            await self.bridge.connect()
        reactor.disconnect.assert_awaited_once_with()
        reactor.close.assert_called_once_with()

    async def test_connect_timeout_cleans_up(self):
        reactor = self.bridge._reactor
        self.bridge.ready_timeout_s = 0.01
        async def stalled():
            await asyncio.Event().wait()
        reactor.connect = AsyncMock(side_effect=stalled)
        reactor.disconnect = AsyncMock()
        reactor.close = Mock()
        with self.assertRaises(TimeoutError):
            await self.bridge.connect()
        reactor.disconnect.assert_awaited_once_with()
        reactor.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
