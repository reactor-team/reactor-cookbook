"""SDK lifecycle and reply-delivery regressions; no hosted model required."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np
from reactor_sdk import Track, TrackDirection, TrackKind
from reactor_robotics.session import ReactorSession
from reactor_robotics.track import RepeatingFrameTrack


class SessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = ReactorSession('offline-test', api_key='offline-test', frame_size=(2, 4))
        reactor = self.session._reactor
        reactor.connect = AsyncMock()
        reactor.disconnect = AsyncMock()
        reactor.close = Mock(wraps=reactor.close)
        reactor._require_handle = Mock()
        reactor._async_op = AsyncMock(return_value=None)
        reactor._push_video_frame = Mock()
        reactor._tracks = {'camera': Track(reactor, 'camera', TrackKind.VIDEO, TrackDirection.SENDONLY)}
        self.messages = []
        reactor.on_message = lambda handler: self.messages.append(handler) or handler

    async def asyncTearDown(self):
        await self.session.close()

    async def test_native_publish_accepts_rgb_and_stops_on_close(self):
        await self.session.connect(['camera'])
        frame = np.full((2, 4, 3), [7, 31, 191], dtype=np.uint8)
        self.session.set_frames({'camera': frame})
        await asyncio.sleep(0.01)
        call = self.session._reactor._push_video_frame.call_args
        self.assertEqual(call.args[0], 'camera')
        self.assertEqual(bytes(call.args[1])[:4], bytes([191, 31, 7, 255]))
        tasks = [self.session._publisher_task] if self.session._publisher_task else []
        await self.session.close()
        await self.session.close()
        self.assertTrue(all(t.done() for t in tasks))
        self.session._reactor.disconnect.assert_awaited_once_with()
        self.session._reactor.close.assert_called_once_with()

    async def test_correlated_return_and_callback_queue_exactly_once(self):
        await self.session.connect([], subscribe=['checkpoint_selected'])
        reply = {'type': 'checkpoint_selected', 'data': {'checkpoint': 'base'}}
        async def command(*args):
            self.messages[0](reply)
            return reply
        self.session._reactor.send_command = AsyncMock(side_effect=command)
        self.assertEqual(await self.session.send('get_checkpoint'), reply)
        self.assertEqual(await self.session.next_message('checkpoint_selected', timeout_s=.1), reply['data'])
        self.assertEqual(self.session.drain('checkpoint_selected'), [])

    async def test_bodyless_ack_and_command_error(self):
        await self.session.connect([])
        self.session._reactor.send_command = AsyncMock(return_value=None)
        self.assertIsNone(await self.session.send('set_task', {}))
        self.session._reactor.send_command.side_effect = RuntimeError('command failed')
        with self.assertRaisesRegex(RuntimeError, 'command failed'):
            await self.session.send('set_task', {})

    async def test_connect_timeout_can_be_closed(self):
        async def stall():
            await asyncio.Event().wait()
        self.session._reactor.connect.side_effect = stall
        with self.assertRaises(TimeoutError):
            await self.session.connect([], ready_timeout_s=.01)
        await self.session.close()
        self.session._reactor.disconnect.assert_awaited_once_with()
        self.session._reactor.close.assert_called_once_with()

    async def test_partial_publication_and_disconnect_failure_release_handle(self):
        self.session._reactor.publish_track = AsyncMock(side_effect=[Mock(), RuntimeError('publish failed')])
        with self.assertRaisesRegex(RuntimeError, 'publish failed'):
            await self.session.connect(['first', 'second'])
        tasks = [self.session._publisher_task] if self.session._publisher_task else []
        self.session._reactor.disconnect.side_effect = RuntimeError('disconnect failed')
        with self.assertRaisesRegex(RuntimeError, 'disconnect failed'):
            await self.session.close()
        self.assertTrue(all(t.done() for t in tasks))
        self.session._reactor.close.assert_called_once_with()

    async def test_publisher_failure_surfaces_before_next_command(self):
        self.session._reactor._push_video_frame.side_effect = RuntimeError('encoder failed')
        await self.session.connect(['camera'])
        await asyncio.sleep(.01)
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            await self.session.send('set_task', {})

    async def test_camera_frame_validation(self):
        source = RepeatingFrameTrack('camera')
        with self.assertRaises(TypeError):
            source.set_frame(np.zeros((2, 4, 3), dtype=float))
        with self.assertRaises(ValueError):
            source.set_frame(np.zeros((2, 4), dtype=np.uint8))
        source.set_frame(np.zeros((2, 4, 3), dtype=np.uint8)[:, ::-1])
        self.assertTrue(source._frame.flags.c_contiguous)
        self.assertEqual(source.pushes, 1)

    async def test_cosmos_retains_reply_during_first_state_ack(self):
        from reactor_robotics.cosmos_droid import CosmosDroidClient, TRACKS
        client = CosmosDroidClient(session=self.session, settle_s=0, timeout_s=.1)
        await self.session.connect([], subscribe=['action_prediction'])
        self.session.set_frames = Mock()
        async def command(name, payload):
            if name == 'set_proprio_json':
                message = {'type': 'action_prediction',
                           'data': {'step': 0, 'action': np.ones((32, 8)).tolist()}}
                self.messages[0](message)
                return message
        self.session._reactor.send_command = AsyncMock(side_effect=command)
        prediction = await client.predict(
            {name: np.zeros((2, 4, 3), np.uint8) for name in TRACKS},
            [0.] * 7, 0., 'pick up')
        self.assertEqual(prediction.step, 0)
        self.assertEqual(self.session.drain('action_prediction'), [])

    async def test_lingbot_reset_keeps_early_seed_and_reapplies_task(self):
        from reactor_robotics.lingbot_va import LingbotVaClient
        client = LingbotVaClient.__new__(LingbotVaClient)
        client.session = self.session
        client._task = 'same task'
        await self.session.connect([], subscribe=['action_prediction'])
        async def command(name, payload):
            if name == 'set_task_description':
                self.messages[0]({'type': 'action_prediction', 'data': {'step': 0}})
        self.session._reactor.send_command = AsyncMock(side_effect=command)
        await client.start_episode('same task')
        self.assertEqual([c.args[0] for c in self.session._reactor.send_command.call_args_list],
                         ['set_executed_action_json', 'reset', 'set_task_description'])
        self.assertEqual(await self.session.next_message('action_prediction', timeout_s=.1), {'step': 0})


if __name__ == '__main__':
    unittest.main()
