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
import types
import unittest
from unittest.mock import patch

import av
import numpy as np

import main
import praxis_client
from main import SyntheticRobot
from praxis_client import (
    DROID,
    STACK,
    YAM,
    PraxisClient,
    SteppedPraxisClient,
    encode_h264,
    parse_chunk,
)

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


def decode_h264(frames) -> list[np.ndarray]:
    """Decode one camera's EncodedFrames of one call as the runtime does: one
    stream, from its first frame."""
    codec = av.CodecContext.create("h264", "r")
    out = []
    for frame in frames:
        out += codec.decode(av.Packet(frame.data))
    out += codec.decode(None)
    return out


class FakeSteppedReactor(FakeReactor):
    """Answers like the model's stepped session: each step applies its state,
    decodes its frames into each camera's window, and pairs the last 25 state
    rows with the 25 newest frames."""

    def __init__(self, model: str, **kwargs: object) -> None:
        super().__init__(model, **kwargs)
        assert kwargs.get("generation") == "stepped", kwargs
        self.generation = "stepped"
        self.window: dict[str, list[np.ndarray]] = {}
        self.media_counts: list[dict[str, int]] = []
        self.step_states: list[dict] = []
        self.keyframe_first: list[bool] = []
        self.reset = False

    async def publish_track(self, name: str) -> FakeTrack:
        self.window[name] = []
        return await super().publish_track(name)

    async def send_command(self, command: str, data: dict) -> dict | None:
        assert command not in ("set_robot_state_json", "set_executed_step"), command
        self.commands.append(command)
        if command == "set_task":
            self.task = data["task"]
        elif command == "reset_context":
            self.awaiting, self.reset = None, True
            return {"type": "context_reset", "data": {"next_step": self.next_step}}
        elif command == "get_status":
            return {"type": "session_status", "data": {"join": "newest"}}
        return None

    async def step(self, state=None, commands=None, media=None):
        self.commands.append("step")
        self.step_states.append(dict(state))
        doc = json.loads(state["robot_state_json"])
        self.state = doc
        self.sent_states.append(doc)
        if "executed_step" in state and state["executed_step"] == self.awaiting:
            self.awaiting = None
        self.media_counts.append({name: len(f) for name, f in media.items()})
        for name, frames in media.items():
            decoded = decode_h264(frames)
            self.keyframe_first.append(decoded[0].key_frame)
            self.window[name] += [f.to_ndarray(format="rgb24") for f in decoded]
        error = None
        if self.awaiting is not None:
            error = f"the robot to execute step {self.awaiting}"
        elif any(len(w) < STACK for w in self.window.values()):
            error = "25 frames of each camera"
        else:
            first = not self.joins or self.reset
            self.reset = False
            self.joins.append("newest")
            step, self.next_step = self.next_step, self.next_step + 1
            self.awaiting = step
            n = 21 if first else 24
            pose = [0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 0, 0, 1]
            rows = {"timestamps": [i / 15 for i in range(n)]}
            if "ee_pose" in doc:
                rows |= {"poses": [pose] * n, "grips": [0.0] * n, "control_json": ""}
            else:
                for arm in ("left", "right"):
                    rows |= {f"{arm}_poses": [pose] * n, f"{arm}_grips": [0.0] * n}
            message = {"type": "action_prediction", "data": {"step": step, **rows}}
            asyncio.get_running_loop().call_soon(self.message_handler, message)
        return types.SimpleNamespace(step=len(self.step_states), error=error)


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
            robot="yam",
            mode="streamed",
            model=None,
            task="pick",
            chunks=2,
            episodes=1,
            pose_frame=None,
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


def no_stamps():
    raise AssertionError("a stepped session must not read time_micros")


class RecordingRobot(SyntheticRobot):
    """Keeps every frame it hands out, to compare with what the model decoded."""

    def __init__(self, embodiment) -> None:
        super().__init__(embodiment)
        self.shown: list[dict[str, np.ndarray]] = []

    def read_cameras(self):
        frames = super().read_cameras()
        self.shown.append({k: v.copy() for k, v in frames.items()})
        return frames


class SteppedClientTest(unittest.TestCase):
    def run_client(self, embodiment, chunks=3, episodes=1):
        fake: dict[str, FakeSteppedReactor] = {}

        def make(model, **kwargs):
            fake["r"] = FakeSteppedReactor(model, **kwargs)
            return fake["r"]

        async def go():
            robot = RecordingRobot(embodiment)
            async with SteppedPraxisClient(
                robot, embodiment, tick_s=TICK_S, status_every=1
            ) as client:
                results = [
                    await client.run_episode("pick", chunks) for _ in range(episodes)
                ]
            return robot, results

        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            patch.object(praxis_client, "Reactor", make),
            patch.object(praxis_client, "time_micros", no_stamps),
            self.assertNoLogs("praxis", "WARNING"),  # join=newest is not warned
        ):
            robot, results = asyncio.run(go())
        return fake["r"], robot, results

    def test_droid_step_carries_frames_state_and_echo(self):
        reactor, robot, [chunks] = self.run_client(DROID)
        self.assertEqual([c.step for c in chunks], [0, 1, 2])
        self.assertEqual([len(c.rows) for c in chunks], [21, 24, 24])
        self.assertEqual(len(robot.executed), 21 + 24 + 24)
        self.assertEqual(reactor.commands.count("step"), 3)
        self.assertTrue(reactor.closed)
        # 25 frames per camera at episode start, then one per executed row.
        self.assertEqual(
            reactor.media_counts,
            [{"exterior": n, "wrist": n} for n in (STACK, 21, 24)],
        )
        self.assertTrue(all(reactor.keyframe_first))
        for state in reactor.sent_states:
            self.assertNotIn("time_us", state)
            self.assertEqual(len(state["ee_pose"]), STACK)
        self.assertNotIn("executed_step", reactor.step_states[0])
        self.assertEqual([s["executed_step"] for s in reactor.step_states[1:]], [0, 1])
        self.assertEqual([c.join for c in chunks], ["newest"] * 3)
        for chunk in chunks:
            self.assertTrue(0 < chunk.sent_bytes < 1_000_000, chunk.sent_bytes)

    def test_decoded_frames_are_the_frames_observed(self):
        reactor, robot, _ = self.run_client(DROID, chunks=2)
        for name in DROID.cameras:
            decoded = reactor.window[name]
            # Every observed tick, in order; the last chunk's frames are kept for
            # a step that never comes.
            shown = [frames[name] for frames in robot.shown][: len(decoded)]
            self.assertEqual(len(decoded), STACK + 21)
            for got, want in zip(decoded, shown):
                diff = np.abs(got.astype(int) - want.astype(int)).mean()
                self.assertLess(diff, 4, f"{name}: mean error {diff:.1f}")

    def test_yam_steps_with_three_cameras_and_both_arms(self):
        reactor, _, [chunks] = self.run_client(YAM, chunks=2)
        self.assertEqual(
            reactor.media_counts[0], {"top": 25, "wrist_left": 25, "wrist_right": 25}
        )
        self.assertEqual(set(reactor.sent_states[-1]), {"left", "right"})
        self.assertEqual(set(chunks[1].rows[0].targets), {"left", "right"})

    def test_second_episode_resets_context(self):
        reactor, _, results = self.run_client(DROID, chunks=2, episodes=2)
        self.assertEqual(reactor.commands.count("reset_context"), 1)
        self.assertEqual([c.step for c in results[1]], [2, 3])
        self.assertEqual([len(c.rows) for c in results[1]], [21, 24])
        reset = reactor.commands.index("reset_context")
        self.assertEqual(reactor.commands[reset + 1], "step")
        third = reactor.step_states[2]  # the second episode's first step
        self.assertNotIn("executed_step", third)
        self.assertEqual(reactor.media_counts[2], {"exterior": 25, "wrist": 25})

    def test_encode_starts_at_a_keyframe_one_frame_each(self):
        robot = SyntheticRobot(YAM)
        frames = [robot.read_cameras()["top"] for _ in range(24)]
        encoded = encode_h264(frames)
        self.assertEqual(len(encoded), 24)
        self.assertEqual((encoded[0].width, encoded[0].height), (416, 260))
        decoded = decode_h264(encoded)
        self.assertEqual(len(decoded), 24)
        self.assertTrue(decoded[0].key_frame)
        self.assertFalse(any(f.key_frame for f in decoded[1:]))
        with self.assertRaises(ValueError):
            encode_h264([np.zeros((259, 416, 3), np.uint8)])

    def test_stepped_state_has_no_time_us(self):
        rows = [
            praxis_client._Row(None, {"arm": main.START_POSES["arm"]})
            for _ in range(STACK)
        ]
        doc = json.loads(praxis_client.state_json(DROID, rows))
        self.assertEqual(set(doc), {"ee_pose", "grip"})
        with self.assertRaises(ValueError):
            praxis_client.state_json(
                DROID, [praxis_client._Row(1, rows[0].state)] + rows
            )

    def test_demo_prints_bytes_and_join(self):
        args = argparse.Namespace(
            robot="droid",
            mode="stepped",
            model=None,
            task="pick",
            chunks=2,
            episodes=1,
            pose_frame="fr3_hand",
        )
        real = praxis_client.SteppedPraxisClient

        def fast(*a, **kw):
            return real(*a, **kw, tick_s=TICK_S)

        out = io.StringIO()
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            patch.object(praxis_client, "Reactor", FakeSteppedReactor),
            patch.object(praxis_client, "time_micros", no_stamps),
            patch.object(main, "SteppedPraxisClient", fast),
            contextlib.redirect_stdout(out),
        ):
            asyncio.run(main.run(args))
        text = out.getvalue()
        self.assertIn("step=0 rows=21", text)
        self.assertIn("step=1 rows=24", text)
        self.assertRegex(text, r"bytes=\d+ join=newest")
        self.assertIn("join=newest: expected in a stepped session", text)
        self.assertIn("Executed 45 rows", text)


if __name__ == "__main__":
    unittest.main()
