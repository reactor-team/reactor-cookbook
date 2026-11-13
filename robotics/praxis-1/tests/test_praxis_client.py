# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Protocol tests without credentials, a GPU, or a running model."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import itertools
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

import main
import praxis_client
from main import SyntheticRobot
from praxis_client import DROID, STACK, YAM, PraxisClient, parse_chunk

TICK_S = 0.002  # fast ticks; the protocol does not depend on the rate


class FakeTrack:
    def __init__(self, reactor: FakeReactor, name: str) -> None:
        self.reactor, self.name = reactor, name

    def push_frame(self, frame: np.ndarray, *, capture_time_us: int) -> None:
        assert frame.dtype == np.uint8 and frame.ndim == 3
        self.reactor.stamps[self.name].append(capture_time_us)


class FakeReactor:
    """Answers like the model's session: waits for the echo, pairs by stamp."""

    rows = 24

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self.stamps: dict[str, list[int]] = {}
        self.state: dict | None = None
        self.task = ""
        self.awaiting: int | None = None
        self.next_step = 0
        self.joins: list[str] = []
        self.commands: list[str] = []
        self.sent_states: list[dict] = []
        self.closed = False

    def on_status(self, handler):
        self.status_handler = handler
        return handler

    def on_message(self, handler):
        self.message_handler = handler
        return handler

    async def connect(self) -> None:
        self.status_handler("ready")

    async def disconnect(self) -> None:
        self.closed = True

    async def publish_track(self, name: str) -> FakeTrack:
        self.stamps[name] = []
        return FakeTrack(self, name)

    async def send_command(self, command: str, data: dict) -> dict | None:
        self.commands.append(command)
        if command == "set_task":
            self.task = data["task"]
        elif command == "set_robot_state_json":
            self.state = json.loads(data["robot_state_json"])
            self.sent_states.append(self.state)
        elif command == "set_executed_step":
            assert self.commands[-2] == "set_robot_state_json", "state before echo"
            if data["executed_step"] == self.awaiting:
                self.awaiting = None
        elif command == "reset_context":
            self.awaiting = None
            reply = {"type": "context_reset", "data": {"next_step": self.next_step}}
            self._maybe_predict()
            return reply
        elif command == "get_status":
            return {
                "type": "session_status",
                "data": {
                    "waiting_for": "",
                    "join": self.joins[-1] if self.joins else "",
                    "next_step": self.next_step,
                    "awaiting_step": -1 if self.awaiting is None else self.awaiting,
                },
            }
        self._maybe_predict()
        return None

    def _maybe_predict(self) -> None:
        if self.awaiting is not None or not self.task or self.state is None:
            return
        times = self.state["time_us"][-STACK:]
        joined = all(set(times) <= set(s) for s in self.stamps.values())
        self.joins.append("capture_time" if joined else "nearest")
        step, self.next_step = self.next_step, self.next_step + 1
        self.awaiting = step
        n = 21 if not self.joins[:-1] or self.commands[-1] == "reset_context" else 24
        pose = [0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 0, 0, 1]
        rows = {"timestamps": [i / 15 for i in range(n)]}
        if "ee_pose" in self.state:
            rows |= {"poses": [pose] * n, "grips": [0.0] * n, "control_json": ""}
        else:
            for arm in ("left", "right"):
                rows |= {f"{arm}_poses": [pose] * n, f"{arm}_grips": [0.0] * n}
        message = {"type": "action_prediction", "data": {"step": step, **rows}}
        asyncio.get_running_loop().call_soon(self.message_handler, message)


class ClientTest(unittest.TestCase):
    def run_client(self, embodiment, chunks=3, episodes=1):
        clock = itertools.count(1_000_000, 1000)
        fake: dict[str, FakeReactor] = {}

        def make(model, **kwargs):
            fake["r"] = FakeReactor(model, **kwargs)
            return fake["r"]

        async def go():
            robot = SyntheticRobot(embodiment)
            async with PraxisClient(
                robot, embodiment, tick_s=TICK_S, status_every=1
            ) as client:
                results = [
                    await client.run_episode("pick", chunks) for _ in range(episodes)
                ]
            return robot, results

        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            patch.object(praxis_client, "Reactor", make),
            patch.object(praxis_client, "time_micros", lambda: next(clock)),
        ):
            robot, results = asyncio.run(go())
        return fake["r"], robot, results

    def test_droid_stamps_pair_every_row(self):
        reactor, robot, [chunks] = self.run_client(DROID)
        self.assertEqual([c.step for c in chunks], [0, 1, 2])
        self.assertEqual([len(c.rows) for c in chunks], [21, 24, 24])
        self.assertEqual(reactor.joins, ["capture_time"] * 3)
        self.assertEqual([c.join for c in chunks], ["capture_time"] * 3)
        self.assertEqual(len(robot.executed), 21 + 24 + 24)
        self.assertTrue(reactor.closed)
        # Both cameras carry the same stamps; the extra frames streamed while
        # waiting are skipped, not paired.
        self.assertEqual(reactor.stamps["exterior"], reactor.stamps["wrist"])
        for state in reactor.sent_states:
            self.assertEqual(len(state["time_us"]), STACK)
            self.assertEqual(len(state["ee_pose"]), STACK)
            self.assertTrue(set(state["time_us"]) <= set(reactor.stamps["wrist"]))

    def test_row_zero_is_the_previous_current_row(self):
        reactor, _, _ = self.run_client(DROID)
        first, second, third = reactor.sent_states
        self.assertEqual(second["time_us"][-22], first["time_us"][-1])  # 21 rows
        self.assertEqual(third["time_us"][0], second["time_us"][-1])  # 24 rows

    def test_echo_follows_state(self):
        reactor, _, _ = self.run_client(DROID)
        for i, command in enumerate(reactor.commands):
            if command == "set_executed_step":
                self.assertEqual(reactor.commands[i - 1], "set_robot_state_json")
        self.assertEqual(reactor.commands.count("set_executed_step"), 2)

    def test_yam_sends_both_arms_with_one_time_us(self):
        reactor, _, [chunks] = self.run_client(YAM, chunks=2)
        state = reactor.sent_states[-1]
        self.assertEqual(set(state), {"left", "right", "time_us"})
        self.assertEqual(len(state["left"]["ee_pose"]), STACK)
        self.assertEqual(set(chunks[0].rows[0].targets), {"left", "right"})
        self.assertEqual(len(reactor.stamps), 3)

    def test_second_episode_resets_context(self):
        reactor, _, results = self.run_client(DROID, chunks=2, episodes=2)
        self.assertEqual(reactor.commands.count("reset_context"), 1)
        self.assertEqual([c.step for c in results[1]], [2, 3])
        self.assertEqual(reactor.joins, ["capture_time"] * 4)

    def test_bad_join_warns(self):
        client = PraxisClient.__new__(PraxisClient)

        class Status:
            async def send_command(self, command, data):
                return {"type": "session_status", "data": {"join": "newest"}}

        client.reactor = Status()
        chunk = parse_chunk(
            DROID, {"step": 0, "timestamps": [0.0], "poses": [[0] * 12], "grips": [0]}
        )
        with self.assertLogs("praxis", "WARNING") as logs:
            asyncio.run(client._poll_status(chunk))
        self.assertEqual(chunk.join, "newest")
        self.assertIn("time_micros", logs.output[0])

    def test_oversized_frames_are_refused(self):
        with self.assertRaises(ValueError):
            DROID.check_frame("wrist", np.zeros((360, 640, 3), np.uint8))
        with self.assertRaises(ValueError):
            YAM.check_frame("top", np.zeros((300, 480, 3), np.uint8))
        YAM.check_frame("top", np.zeros((260, 416, 3), np.uint8))

    def test_demo_prints_each_chunk(self):
        clock = itertools.count(1_000_000, 1000)
        args = argparse.Namespace(
            robot="yam", model=None, task="pick", chunks=2, episodes=1, pose_frame=None
        )
        real = praxis_client.PraxisClient

        def fast(*a, **kw):
            return real(*a, **kw, tick_s=TICK_S)

        out = io.StringIO()
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            patch.object(praxis_client, "Reactor", FakeReactor),
            patch.object(praxis_client, "time_micros", lambda: next(clock)),
            patch.object(main, "PraxisClient", fast),
            contextlib.redirect_stdout(out),
        ):
            asyncio.run(main.run(args))
        text = out.getvalue()
        self.assertIn("step=0 rows=21", text)
        self.assertIn("step=1 rows=24", text)
        self.assertIn("join=capture_time", text)
        self.assertIn("Executed 45 rows", text)

    def test_model_needs_an_owner(self):
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            self.assertRaises(ValueError),
        ):
            PraxisClient(SyntheticRobot(DROID), DROID, model="praxis-1-droid")


if __name__ == "__main__":
    unittest.main()
