# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Protocol tests without credentials, a GPU, or a running model."""

import asyncio
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from reactor_sdk import ReactorStatus

from client import CHECKPOINTS, VIEWS, FluxClient, select_checkpoint
from main import observations


class FakeReactor:
    def __init__(self, *args, **kwargs):
        self.checkpoint = "base-bf16"
        self.locked = False
        self.available = list(CHECKPOINTS)
        self.commands = []
        self.closed = False
        self.reply_checkpoint = None
        self.bad_actions = False
        self.fail_selection = False
        self.fail_ready = False
        self.error = False
        self.silent = False

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
        if command == "select_checkpoint":
            if self.fail_selection:
                return None
            self.checkpoint = payload["checkpoint"]
            self.locked = True
        if command in ("get_checkpoint", "select_checkpoint"):
            return {
                "type": "checkpoint_selected",
                "data": {
                    "checkpoint": self.checkpoint,
                    "locked": self.locked,
                    "available": self.available,
                },
            }
        if command == "set_state_json":
            request = json.loads(payload["state_json"])
            if self.silent:
                return
            if self.error:
                self.message_handler(
                    {
                        "type": "command_error",
                        "data": {"command": "state_json", "reason": "invalid camera"},
                    }
                )
                return
            # An old reply must not be accepted, even if it arrives first.
            for step in (request["chunk_id"] - 1, request["chunk_id"]):
                self.message_handler(
                    {
                        "type": "action_prediction",
                        "data": {
                            "step": step,
                            "checkpoint": self.reply_checkpoint or self.checkpoint,
                            "actions": [[float("nan") if self.bad_actions else 0.0] * 8]
                            * 32,
                            "inference_seconds": 0.1,
                        },
                    }
                )


class ClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"REACTOR_API_KEY": "offline-placeholder"})
        self.env.start()
        self.factory = patch("client.Reactor", FakeReactor)
        self.factory.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.factory.stop)
        self.frames, self.proprio, self.task = observations(None, 1)[0]

    async def test_all_checkpoint_choices_and_reset(self):
        for checkpoint in CHECKPOINTS:
            with self.subTest(checkpoint=checkpoint):
                async with FluxClient(checkpoint, settle_s=0) as client:
                    first = await client.predict(
                        self.frames, self.proprio, self.task, seed=7
                    )
                    self.assertEqual(first.step, 0)
                    self.assertEqual(first.checkpoint, checkpoint)
                    await client.reset()
                    second = await client.predict(self.frames, self.proprio, self.task)
                    self.assertEqual(second.step, 1)
                    self.assertEqual(second.actions.shape, (32, 8))
                    commands = client.reactor.commands
                    self.assertLess(
                        [c[0] for c in commands].index("select_checkpoint"),
                        [c[0] for c in commands].index("set_state_json"),
                    )
                    sent = [
                        json.loads(p["state_json"])
                        for c, p in commands
                        if c == "set_state_json"
                    ]
                    self.assertEqual(sent[0]["seed"], 7)
                    self.assertNotIn("seed", sent[1])
                self.assertTrue(client.reactor.closed)
                self.assertIsNone(client._publisher)

    async def test_unavailable_and_pinned_checkpoint_do_not_fall_back(self):
        reactor = FakeReactor()
        reactor.available = ["base-bf16"]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await select_checkpoint(reactor, "gd-fp8")
        reactor.available = list(CHECKPOINTS)
        reactor.locked = True
        with self.assertRaisesRegex(RuntimeError, "already pinned"):
            await select_checkpoint(reactor, "gd-fp8")
        self.assertFalse(any(c == "select_checkpoint" for c, _ in reactor.commands))

    async def test_unconfirmed_selection_closes_session(self):
        client = FluxClient("gd-fp8")
        client.reactor.fail_selection = True
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            async with client:
                self.fail("Unconfirmed checkpoint must not enter prediction loop")
        self.assertTrue(client.reactor.closed)

    async def test_connection_failure_is_cleaned_up(self):
        client = FluxClient()
        client.reactor.fail_ready = True
        with self.assertRaisesRegex(RuntimeError, "connection failed"):
            async with client:
                pass
        self.assertTrue(client.reactor.closed)

    async def test_malformed_and_wrong_checkpoint_responses_fail(self):
        for attribute, value, message in (
            ("bad_actions", True, "finite"),
            ("reply_checkpoint", "sd-fp8", "wrong checkpoint"),
        ):
            client = FluxClient(settle_s=0)
            setattr(client.reactor, attribute, value)
            with self.assertRaisesRegex(ValueError, message):
                async with client:
                    await client.predict(self.frames, self.proprio, self.task)
            self.assertTrue(client.reactor.closed)

    async def test_command_error_surfaces_without_waiting_for_timeout(self):
        async with FluxClient(settle_s=0) as client:
            client.reactor.error = True
            with self.assertRaisesRegex(RuntimeError, "invalid camera"):
                await asyncio.wait_for(
                    client.predict(self.frames, self.proprio, self.task), 1
                )

    async def test_invalid_observation_does_not_send_state(self):
        async with FluxClient(settle_s=0) as client:
            for frames, state in (
                ({}, self.proprio),
                ({v: f.astype(float) for v, f in self.frames.items()}, self.proprio),
                (self.frames, np.zeros(7)),
                (self.frames, np.array([0.0] * 7 + [2.0])),
                (self.frames, np.array([1e100] + [0.0] * 7)),
            ):
                with self.assertRaises(ValueError):
                    await client.predict(frames, state, self.task)
            self.assertFalse(
                any(c == "set_state_json" for c, _ in client.reactor.commands)
            )

    async def test_invalid_seed_is_rejected_before_submission(self):
        async with FluxClient(settle_s=0) as client:
            for seed in (-1, 2**63, True, 1.5):
                with self.assertRaisesRegex(ValueError, "seed must be"):
                    await client.predict(
                        self.frames, self.proprio, self.task, seed=seed
                    )
            self.assertFalse(
                any(c == "set_state_json" for c, _ in client.reactor.commands)
            )

    async def test_timeout_closes_session_without_retry(self):
        client = FluxClient(settle_s=0, timeout_s=0.01)
        client.reactor.silent = True
        with self.assertRaises(asyncio.TimeoutError):
            async with client:
                await client.predict(self.frames, self.proprio, self.task)
        self.assertTrue(client.reactor.closed)
        self.assertEqual(
            sum(c == "set_state_json" for c, _ in client.reactor.commands), 1
        )


class ReplayTest(unittest.TestCase):
    def test_npz_loads_named_cameras_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.npz"
            np.savez(
                path,
                **{v: np.zeros((2, 4, 6, 3), np.uint8) for v in VIEWS},
                proprio=np.zeros((2, 8)),
                task=np.array(["task one", "task two"]),
            )
            rows = observations(path, 2)
            self.assertEqual(rows[1][2], "task two")
            self.assertEqual(rows[0][0]["wrist_view"].shape, (4, 6, 3))
            with self.assertRaisesRegex(ValueError, "Only 2"):
                observations(path, 3)


class ReadmeTest(unittest.TestCase):
    def test_complete_python_example_runs_as_written(self):
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
        examples = re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
        complete = [code for code in examples if "asyncio.run(main())" in code]
        self.assertEqual(len(complete), 1)
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "offline-placeholder"}),
            patch("client.Reactor", FakeReactor),
        ):
            exec(compile(complete[0], "README example.py", "exec"), {})  # noqa: S102 - execute the repository's documented example against a fake SDK

    def test_relative_links_resolve(self):
        root = Path(__file__).resolve().parents[4]
        pages = (
            "README.md",
            "robotics/README.md",
            "robotics/sim/README.md",
            "robotics/sim/notebooks/README.md",
            "robotics/flux3-action-droid/README.md",
        )
        for page in pages:
            path = root / page
            for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", path.read_text()):
                target = target.split("#", 1)[0]
                if target and "://" not in target:
                    with self.subTest(page=page, target=target):
                        self.assertTrue((path.parent / target).exists())


if __name__ == "__main__":
    unittest.main()
