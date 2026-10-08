# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Protocol tests without credentials, a GPU, or a running model."""

import argparse
import asyncio
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from reactor_sdk import ReactorStatus

from client import (
    ACTION_SHAPE,
    VIEWS,
    RhoClient,
    arm_vector,
    matrix_from_rot6d,
    pack_state,
    rot6d_from_matrix,
    split_arms,
)
from main import HOME_STATE, TASK, MockYam, SyntheticCameras, run

ROOT = Path(__file__).resolve().parents[6]


class FakeReactor:
    """Answers like the deployed model: one reply per request, step echoes chunk_id."""

    def __init__(self, *args, **kwargs):
        self.commands = []
        self.closed = False
        self.fail_ready = False
        self.error = False
        self.silent = False
        self.bad_actions = False
        self.bad_horizon = False

    def on_status(self, handler):
        self.status_handler = handler

    def on_message(self, handler):
        self.message_handler = handler

    async def connect(self):
        if self.fail_ready:
            raise RuntimeError("connection failed")
        self.status_handler(ReactorStatus.READY)

    async def disconnect(self):
        self.closed = True

    async def publish_track(self, name):
        return self

    def push_frame(self, frame):
        pass

    async def send_command(self, command, payload):
        self.commands.append((command, payload))
        if command != "set_state_json" or self.silent:
            return
        request = json.loads(payload["state_json"])
        assert len(request["proprio"]) == 20
        if self.error:
            self.message_handler(
                {
                    "type": "command_error",
                    "data": {"command": "state_json", "reason": "frame is not RGB"},
                }
            )
            return
        # The target rows repeat the request state, so a mock robot stays put.
        row = [float("nan")] * 20 if self.bad_actions else request["proprio"]
        # A delayed reply for an older request arrives first; the client must skip it.
        for step in (request["chunk_id"] - 1, request["chunk_id"]):
            self.message_handler(
                {
                    "type": "action_prediction",
                    "data": {
                        "actions": [row] * 50,
                        # The deployed model's integers arrive as floats.
                        "execution_horizon": 0.0 if self.bad_horizon else 25.0,
                        "step": float(step),
                        "inference_seconds": 0.07,
                    },
                }
            )


class Patched(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for patcher in (
            patch.dict(os.environ, {"REACTOR_API_KEY": "offline-placeholder"}),
            patch("client.Reactor", FakeReactor),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.frames = SyntheticCameras().read()
        self.state = HOME_STATE.copy()


class ClientTest(Patched):
    async def test_predict_sends_contract_and_parses_reply(self):
        async with RhoClient(settle_s=0) as client:
            first = await client.predict(self.frames, self.state, TASK, seed=7)
            second = await client.predict(self.frames, self.state, TASK)
            await client.reset()
            third = await client.predict(self.frames, self.state, TASK)
        self.assertEqual([first.step, second.step, third.step], [0, 1, 2])
        self.assertEqual(first.actions.shape, ACTION_SHAPE)
        self.assertEqual(first.to_execute.shape, (25, 20))
        self.assertIs(type(first.execution_horizon), int)
        names = [c for c, _ in client.reactor.commands]
        self.assertEqual(names.count("set_task_description"), 1)
        self.assertLess(
            names.index("set_task_description"), names.index("set_state_json")
        )
        sent = [
            json.loads(p["state_json"])
            for c, p in client.reactor.commands
            if c == "set_state_json"
        ]
        self.assertEqual(sent[0]["seed"], 7)
        self.assertNotIn("seed", sent[1])
        self.assertEqual(sent[0]["proprio"], self.state.tolist())
        self.assertTrue(client.reactor.closed)
        self.assertIsNone(client._publisher)

    async def test_connection_failure_is_cleaned_up(self):
        client = RhoClient()
        client.reactor.fail_ready = True
        with self.assertRaisesRegex(RuntimeError, "connection failed"):
            async with client:
                pass
        self.assertTrue(client.reactor.closed)

    async def test_malformed_replies_fail(self):
        for attribute, message in (
            ("bad_actions", "finite"),
            ("bad_horizon", "execution_horizon"),
        ):
            client = RhoClient(settle_s=0)
            setattr(client.reactor, attribute, True)
            with self.assertRaisesRegex(ValueError, message):
                async with client:
                    await client.predict(self.frames, self.state, TASK)
            self.assertTrue(client.reactor.closed)

    async def test_command_error_surfaces_without_waiting_for_timeout(self):
        async with RhoClient(settle_s=0) as client:
            client.reactor.error = True
            with self.assertRaisesRegex(RuntimeError, "frame is not RGB"):
                await asyncio.wait_for(client.predict(self.frames, self.state, TASK), 1)

    async def test_invalid_input_does_not_send_a_request(self):
        parallel = self.state.copy()
        parallel[6:9] = parallel[3:6]
        async with RhoClient(settle_s=0) as client:
            for frames, state, task, seed in (
                ({}, self.state, TASK, None),
                (
                    {v: f.astype(float) for v, f in self.frames.items()},
                    self.state,
                    TASK,
                    None,
                ),
                (self.frames, np.zeros(14), TASK, None),
                (self.frames, np.full(20, np.nan), TASK, None),
                (self.frames, parallel, TASK, None),
                (self.frames, self.state, "", None),
                (self.frames, self.state, "x" * 301, None),
                (self.frames, self.state, TASK, -1),
                (self.frames, self.state, TASK, True),
            ):
                with self.assertRaises(ValueError):
                    await client.predict(frames, state, task, seed=seed)
        self.assertFalse(any(c == "set_state_json" for c, _ in client.reactor.commands))

    async def test_timeout_closes_session_without_retry(self):
        client = RhoClient(settle_s=0, timeout_s=0.01)
        client.reactor.silent = True
        with self.assertRaises(asyncio.TimeoutError):
            async with client:
                await client.predict(self.frames, self.state, TASK)
        self.assertTrue(client.reactor.closed)
        self.assertEqual(
            sum(c == "set_state_json" for c, _ in client.reactor.commands), 1
        )

    async def test_main_loop_executes_horizon_and_resets(self):
        args = argparse.Namespace(
            model="reactor/rho-yam-box",
            task=TASK,
            requests=3,
            seed=None,
            settle_s=0,
            no_realtime=True,
        )
        with patch("builtins.print") as printed:
            await run(args)
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(lines[-2].startswith("PASS: 4 valid predictions"))
        self.assertIn("Reset check: step=3", lines[-3])

    async def test_mock_robot_reaches_last_executed_row(self):
        robot = MockYam(realtime=False)
        rows = np.tile(self.state, (25, 1))
        rows[-1, 0] = 0.5
        await robot.execute(rows)
        self.assertEqual(robot.get_state()[0], 0.5)


class LayoutTest(unittest.TestCase):
    def test_rot6d_round_trip_and_layout(self):
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        r6 = rot6d_from_matrix(rotation)
        # Column by column: [R00, R10, R20, R01, R11, R21].
        np.testing.assert_allclose(
            r6, rotation[:, 0].tolist() + rotation[:, 1].tolist()
        )
        np.testing.assert_allclose(matrix_from_rot6d(r6), rotation, atol=1e-12)

    def test_pack_and_split_put_left_arm_first(self):
        left = arm_vector([0.1, 0.2, 0.3], np.eye(3), -0.5)
        right = arm_vector([0.4, 0.5, 0.6], np.eye(3), 1.0)
        arms = split_arms(pack_state(left, right))
        np.testing.assert_allclose(arms["left"]["xyz"], [0.1, 0.2, 0.3])
        self.assertEqual(arms["left"]["gripper"], -0.5)
        np.testing.assert_allclose(arms["right"]["xyz"], [0.4, 0.5, 0.6])
        np.testing.assert_allclose(arms["right"]["rotation"], np.eye(3))
        self.assertEqual(
            tuple(VIEWS), ("scene_view", "left_wrist_view", "right_wrist_view")
        )


class ReadmeTest(unittest.TestCase):
    def test_complete_python_example_runs_as_written(self):
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
        examples = re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
        complete = [code for code in examples if "asyncio.run(main())" in code]
        self.assertEqual(len(complete), 1)
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "offline-placeholder"}),
            patch("client.Reactor", FakeReactor),
            patch("builtins.print"),
        ):
            exec(compile(complete[0], "README example.py", "exec"), {})  # noqa: S102 - run the documented example against a fake SDK

    def test_relative_links_resolve(self):
        pages = (
            "robotics/README.md",
            "robotics/embodiments/README.md",
            "robotics/embodiments/yam/README.md",
            "robotics/embodiments/yam/rho-yam-box/README.md",
        )
        for page in pages:
            path = ROOT / page
            for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", path.read_text()):
                target = target.split("#", 1)[0]
                if target and "://" not in target:
                    with self.subTest(page=page, target=target):
                        self.assertTrue((path.parent / target).exists())


if __name__ == "__main__":
    unittest.main()
