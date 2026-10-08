"""Native SDK adapter regressions without a simulator or hosted inference."""

import asyncio
import queue
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from reactor_sdk import Reactor
from robocasa365_sim.client import ReactorEvalClient, TRACK_ORDER


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = ReactorEvalClient.__new__(ReactorEvalClient)
        self.client._api_url = "http://127.0.0.1:8123"
        self.client._model = "xr1-robocasa365"
        self.client._timeout = 0.1
        self.client._msgs = queue.Queue()
        self.client._last_chunk_step = -1
        self.sdk = Reactor(self.client._model, api_url=self.client._api_url, local=True)
        self.sdk.connect = AsyncMock()
        self.sdk.disconnect = AsyncMock()
        self.sdk.close = Mock(wraps=self.sdk.close)
        self.sdk.publish_track = AsyncMock(side_effect=lambda name: Mock(name=name))

    async def asyncTearDown(self):
        self.sdk.close()

    async def test_public_local_url_and_native_track_api(self):
        with patch("reactor_sdk.Reactor", return_value=self.sdk) as factory:
            await self.client._connect()
        factory.assert_called_once_with(
            "xr1-robocasa365", api_url=self.client._api_url, local=True
        )
        self.assertEqual(list(self.client._tracks), list(TRACK_ORDER))
        self.assertEqual(
            [c.args for c in self.sdk.publish_track.await_args_list],
            [(n,) for n in TRACK_ORDER],
        )

    async def test_early_callback_retained_once(self):
        async def connect():
            for handler in self.sdk._handlers["message"]:
                handler(
                    {
                        "type": "action_prediction",
                        "data": {"step": 0, "action": np.zeros((16, 60)).tolist()},
                    }
                )

        self.sdk.connect.side_effect = connect
        with patch("reactor_sdk.Reactor", return_value=self.sdk):
            await self.client._connect()
        self.assertEqual(self.client._await_chunk().shape, (16, 60))
        self.assertTrue(self.client._msgs.empty())

    async def test_partial_publish_failure_can_release_session(self):
        self.sdk.publish_track.side_effect = [
            Mock(),
            RuntimeError("publication failed"),
        ]
        with patch("reactor_sdk.Reactor", return_value=self.sdk):
            with self.assertRaisesRegex(RuntimeError, "publication failed"):
                await self.client._connect()
        await self.client._disconnect()
        self.sdk.disconnect.assert_awaited_once()
        self.sdk.close.assert_called_once()

    async def test_disconnect_failure_still_closes_native_handle(self):
        self.client._reactor = self.sdk
        self.sdk.disconnect.side_effect = RuntimeError("disconnect failed")
        with self.assertRaisesRegex(RuntimeError, "disconnect failed"):
            await self.client._disconnect()
        self.sdk.close.assert_called_once()

    async def test_invalid_action_chunks_rejected(self):
        for actions in (np.zeros((1, 12)), np.full((16, 60), np.nan)):
            self.client._last_chunk_step = -1
            self.client._msgs.put(
                {
                    "type": "action_prediction",
                    "data": {"step": 0, "action": actions.tolist()},
                }
            )
            with self.assertRaisesRegex(RuntimeError, "bad action shape"):
                self.client._await_chunk()

    async def test_duplicate_chunk_is_not_executed_twice(self):
        for step in (0, 0, 1):
            self.client._msgs.put(
                {
                    "type": "action_prediction",
                    "data": {"step": step, "action": np.full((16, 60), step).tolist()},
                }
            )
        self.assertTrue((self.client._await_chunk() == 0).all())
        self.assertTrue((self.client._await_chunk() == 1).all())


class ThreadTimeoutTests(unittest.TestCase):
    def test_timed_out_operation_is_cancelled_on_python310_too(self):
        client = ReactorEvalClient.__new__(ReactorEvalClient)
        client._loop = asyncio.new_event_loop()
        client._thread = threading.Thread(target=client._loop.run_forever)
        client._thread.start()
        client._timeout = 1
        cancelled = threading.Event()

        async def never_finishes():
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        try:
            with self.assertRaises(FutureTimeoutError):
                client._run(never_finishes(), timeout_s=0.03)
            self.assertTrue(cancelled.wait(1))
        finally:
            client._loop.call_soon_threadsafe(client._loop.stop)
            client._thread.join(1)
            client._loop.close()


if __name__ == "__main__":
    unittest.main()
