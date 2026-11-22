# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Protocol tests without credentials, a GPU, or a running model."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import itertools
import json
import logging
import os
import types
import unittest
from collections import deque
from unittest.mock import patch

import av
import numpy as np

import main
import praxis_client
from main import SyntheticRobot
from praxis_client import (
    DROID,
    MAX_WAIT_FRAMES,
    STACK,
    WINDOW,
    YAM,
    ArmState,
    PraxisClient,
    PredictionFailed,
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
        self.reactor.kept[self.name].append(capture_time_us)


class FakeReactor:
    """Answers like the model's session: waits for the echo, keeps the newest
    50 frames per camera, and pairs each state row with the frame stamped with
    it in that window (``nearest`` when one is missing)."""

    #: Seconds before each chunk arrives (the model's latency).
    delay_s = 0.0
    #: ``prediction_failed`` codes to send, in order, instead of the next chunks;
    #: the session retries by itself after each, as the model does.
    fail_codes: tuple[str, ...] = ()
    #: Whether ``get_status`` takes a while (it then answers after 50 ms).
    slow_status = False

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self.stamps: dict[str, list[int]] = {}
        self.kept: dict[str, deque[int]] = {}  # the model's window, by stamp
        self.state: dict | None = None
        self.task = ""
        self.awaiting: int | None = None
        self.next_step = 0
        self.fresh = True  # the next chunk opens a context (21 rows)
        self.joins: list[str] = []
        self.commands: list[str] = []
        self.sent_states: list[dict] = []
        self.used_states: list[dict] = []  # the state each chunk was computed from
        self.failures = list(self.fail_codes)
        self.stream_at_reset: list[int] = []
        self.waiting_for = "a task"
        self.connected = False
        self.closed = False

    def on_status(self, handler):
        self.status_handler = handler
        return handler

    def on_message(self, handler):
        self.message_handler = handler
        return handler

    async def connect(self) -> None:
        self.connected = True
        self.status_handler("ready")

    async def disconnect(self) -> None:
        self.closed = True

    async def publish_track(self, name: str) -> FakeTrack:
        self.stamps[name] = []
        self.kept[name] = deque(maxlen=WINDOW)
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
            self.awaiting, self.fresh = None, True
            self.stream_at_reset.append(len(next(iter(self.stamps.values()))))
            reply = {"type": "context_reset", "data": {"next_step": self.next_step}}
            self._maybe_predict()
            return reply
        elif command == "get_status":
            if self.slow_status:
                await asyncio.sleep(0.05)
            self.commands.append("status_reply")
            return {
                "type": "session_status",
                "data": {
                    "waiting_for": self.waiting_for,
                    "join": self.joins[-1] if self.joins else "",
                    "next_step": self.next_step,
                    "awaiting_step": -1 if self.awaiting is None else self.awaiting,
                },
            }
        self._maybe_predict()
        return None

    def _send(self, message: dict, delay_s: float = 0.0) -> None:
        loop = asyncio.get_running_loop()
        loop.call_later(delay_s, self.message_handler, message)

    def _maybe_predict(self) -> None:
        if self.awaiting is not None:
            self.waiting_for = f"the robot to execute step {self.awaiting}"
            return
        if not self.task or self.state is None:
            return
        if self.failures:
            code = self.failures.pop(0)
            data = {"step": self.next_step, "code": code, "reason": "test"}
            self._send({"type": "prediction_failed", "data": data})
            self.fresh = True  # the model starts the context over after a failure
            loop = asyncio.get_running_loop()
            loop.call_later(0.01, self._maybe_predict)  # the session's own retry
            return
        times = self.state["time_us"][-STACK:]
        joined = all(set(times) <= set(w) for w in self.kept.values())
        self.joins.append("capture_time" if joined else "nearest")
        self.used_states.append(self.state)
        step, self.next_step = self.next_step, self.next_step + 1
        self.awaiting = step
        n = 21 if self.fresh else 24
        self.fresh = False
        self._send(
            {"type": "action_prediction", "data": prediction(self.state, step, n)},
            self.delay_s,
        )


def prediction(state: dict, step: int, n: int) -> dict:
    """An ``action_prediction`` payload of ``n`` rows for a DROID or YAM state."""
    pose = [0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    rows: dict = {"step": step, "timestamps": [i / 15 for i in range(n)]}
    if "ee_pose" in state:
        rows |= {"poses": [pose] * n, "grips": [0.0] * n, "control_json": ""}
    else:
        for arm in ("left", "right"):
            rows |= {f"{arm}_poses": [pose] * n, f"{arm}_grips": [0.0] * n}
    return rows


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
            data = {"join": "newest", "waiting_for": self.waiting_for}
            return {"type": "session_status", "data": data}
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
            self.used_states.append(doc)
            step, self.next_step = self.next_step, self.next_step + 1
            self.awaiting = step
            message = prediction(doc, step, 21 if first else 24)
            self._send({"type": "action_prediction", "data": message})
        return types.SimpleNamespace(step=len(self.step_states), error=error)


class EchoRobot(SyntheticRobot):
    """Reports the index of the last row it executed as its x position, so each
    state row shows which command it is the result of (-1 before any)."""

    def __init__(self, embodiment) -> None:
        super().__init__(embodiment)
        self.last = -1

    def execute(self, row) -> None:
        super().execute(row)
        self.last = row.index

    def read_arm_state(self):
        pose = [float(self.last), 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
        return {name: ArmState(pose, 0.0) for name in self.embodiment.arms}


class RecordingRobot(SyntheticRobot):
    """Keeps every frame it hands out, to compare with what the model decoded."""

    def __init__(self, embodiment) -> None:
        super().__init__(embodiment)
        self.shown: list[dict[str, np.ndarray]] = []

    def read_cameras(self):
        frames = super().read_cameras()
        self.shown.append({k: v.copy() for k, v in frames.items()})
        return frames


class JoinWarnings(logging.Handler):
    """Collects the client's ``join`` warnings; other warnings (a late tick on a
    loaded machine) are not what a test checks."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.joins: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("join="):
            self.joins.append(record.getMessage())


def run_session(
    client_class,
    fake_class,
    embodiment,
    *,
    robot=None,
    chunks=3,
    episodes=1,
    between=None,
    **options,
):
    """Run ``episodes`` episodes against a fake session. Returns the fake, the
    robot, each episode's chunks (or the exception that ended it), and the
    tasks still pending when the session closed."""
    clock = itertools.count(1_000_000, 1000)
    fake: dict = {}

    def make(model, **kwargs):
        fake["r"] = fake_class(model, **kwargs)
        return fake["r"]

    robot = robot or SyntheticRobot(embodiment)
    options = {"tick_s": TICK_S, "status_every": 1, **options}

    async def go():
        results = []
        async with client_class(robot, embodiment, **options) as client:
            for _ in range(episodes):
                try:
                    results.append(await client.run_episode("pick", chunks))
                except Exception as exc:  # noqa: BLE001 - returned to the test
                    results.append(exc)
                if between:
                    await between()
        await asyncio.sleep(0)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return results, pending

    stamps = no_stamps if client_class is SteppedPraxisClient else lambda: next(clock)
    with (
        patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
        patch.object(praxis_client, "Reactor", make),
        patch.object(praxis_client, "time_micros", stamps),
    ):
        results, pending = asyncio.run(go())
    return fake["r"], robot, results, pending


def gaps(stream: list[int], rows: list[int]) -> list[int]:
    """The frames streamed between each two consecutive rows."""
    index = {t: i for i, t in enumerate(stream)}
    return [index[b] - index[a] - 1 for a, b in itertools.pairwise(rows)]


class ClientTest(unittest.TestCase):
    def run_client(self, embodiment, chunks=3, episodes=1, **kw):
        reactor, robot, results, pending = run_session(
            PraxisClient,
            FakeReactor,
            embodiment,
            chunks=chunks,
            episodes=episodes,
            **kw,
        )
        self.assertEqual(pending, [], "a task outlived the session")
        return reactor, robot, results

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

    def test_row_zero_after_each_chunk(self):
        reactor, _, _ = self.run_client(DROID)
        first, second, third = reactor.sent_states
        # After the episode's first chunk (21 rows), the last 4 start rows lead:
        # the previous call's current row is row 3.
        self.assertEqual(second["time_us"][3], first["time_us"][-1])
        self.assertEqual(second["time_us"][:4], first["time_us"][-4:])
        # After a 24-row chunk, it is row 0.
        self.assertEqual(third["time_us"][0], second["time_us"][-1])

    def test_each_row_is_observed_after_its_command(self):
        reactor, _, _ = self.run_client(DROID, robot=EchoRobot(DROID))
        first, second, third = (
            [p[0] for p in s["ee_pose"]] for s in reactor.sent_states
        )
        self.assertEqual(first, [-1] * STACK)
        self.assertEqual(second, [-1] * 4 + list(range(21)))
        self.assertEqual(third, [20, *range(24)])

    def test_echo_follows_state(self):
        reactor, _, _ = self.run_client(DROID)
        for i, command in enumerate(reactor.commands):
            if command == "set_executed_step":
                self.assertEqual(reactor.commands[i - 1], "set_robot_state_json")
        self.assertEqual(reactor.commands.count("set_executed_step"), 2)

    def test_status_poll_finishes_before_the_next_state(self):
        class SlowStatus(FakeReactor):
            slow_status = True

        reactor, _, results, pending = run_session(
            PraxisClient, SlowStatus, DROID, chunks=3
        )
        self.assertEqual(pending, [])
        self.assertEqual([c.join for c in results[0]], ["capture_time"] * 3)
        states = [
            i for i, c in enumerate(reactor.commands) if c == "set_robot_state_json"
        ]
        replies = [i for i, c in enumerate(reactor.commands) if c == "status_reply"]
        # The poll of chunk k answers before the state that triggers chunk k + 1.
        for reply, state in zip(replies, states[1:]):
            self.assertLess(reply, state)

    def test_yam_sends_both_arms_with_one_time_us(self):
        reactor, _, [chunks] = self.run_client(YAM, chunks=2)
        state = reactor.sent_states[-1]
        self.assertEqual(set(state), {"left", "right", "time_us"})
        self.assertEqual(len(state["left"]["ee_pose"]), STACK)
        self.assertEqual(set(chunks[0].rows[0].targets), {"left", "right"})
        self.assertEqual(len(reactor.stamps), 3)

    def test_second_episode_sends_state_then_resets(self):
        reactor, robot, results = self.run_client(
            DROID, chunks=2, episodes=2, robot=EchoRobot(DROID)
        )
        self.assertEqual(reactor.commands.count("reset_context"), 1)
        self.assertEqual([c.step for c in results[1]], [2, 3])
        self.assertEqual([len(c.rows) for c in results[1]], [21, 24])
        self.assertEqual(reactor.joins, ["capture_time"] * 4)
        reset = reactor.commands.index("reset_context")
        self.assertEqual(reactor.commands[reset - 1], "set_robot_state_json")
        # Episode 2's first chunk is computed from its 25 still rows: the newest
        # 25 frames when the reset arrived, all after the arm's last command.
        used = reactor.used_states[2]
        [seen] = reactor.stream_at_reset
        self.assertEqual(used["time_us"], reactor.stamps["wrist"][seen - STACK : seen])
        self.assertEqual([p[0] for p in used["ee_pose"]], [23] * STACK)
        self.assertEqual(robot.holds, 3)  # after each episode, and on exit

    def test_streaming_while_waiting_stops_before_row_zero_is_evicted(self):
        class Slow(FakeReactor):
            delay_s = 0.3  # 150 ticks of TICK_S: far longer than the window

        reactor, _, [chunks] = self.run_client_with(Slow, DROID, chunks=3)
        self.assertEqual(reactor.joins, ["capture_time"] * 3)
        stream = reactor.stamps["wrist"]
        for state in reactor.sent_states[1:]:
            self.assertEqual(max(gaps(stream, state["time_us"])), MAX_WAIT_FRAMES)
        self.assertLessEqual(STACK + MAX_WAIT_FRAMES + 8, WINDOW)
        for chunk in chunks:
            self.assertGreaterEqual(chunk.latency_ms, 300)

    def test_unbounded_streaming_would_evict_row_zero(self):
        class Slow(FakeReactor):
            delay_s = 0.3

        with (
            patch.object(praxis_client, "MAX_WAIT_FRAMES", 10_000),
            self.assertLogs("praxis", "WARNING") as logs,
        ):
            reactor, _, _ = self.run_client_with(Slow, DROID, chunks=3)
        self.assertIn("join='nearest'", logs.output[0])
        self.assertEqual(reactor.joins[0], "capture_time")
        self.assertEqual(reactor.joins[1:], ["nearest"] * 2)

    def run_client_with(self, fake_class, embodiment, **kw):
        reactor, robot, results, pending = run_session(
            PraxisClient, fake_class, embodiment, **kw
        )
        self.assertEqual(pending, [], "a task outlived the session")
        return reactor, robot, results

    def test_transient_failures_are_retried(self):
        class Busy(FakeReactor):
            fail_codes = ("busy", "timeout")

        with self.assertLogs("praxis", "WARNING") as logs:
            _, _, [chunks] = self.run_client_with(Busy, DROID, chunks=2)
        self.assertEqual([c.step for c in chunks], [0, 1])
        failures = [line for line in logs.output if "prediction_failed" in line]
        self.assertEqual(len(failures), 2)
        self.assertIn("code=busy", failures[0])

    def test_permanent_failure_raises_and_holds(self):
        for code in ("task_too_long", "too_large", "invalid_input", "rejected"):
            with self.subTest(code=code):

                class Refuses(FakeReactor):
                    fail_codes = (code,)

                _, robot, [error] = self.run_client_with(Refuses, DROID, chunks=2)
                self.assertIsInstance(error, PredictionFailed)
                self.assertEqual((error.step, error.code), (0, code))
                self.assertEqual(robot.executed, [])
                self.assertEqual(robot.holds, 2)  # the episode's error, then exit

    def test_timeout_says_what_the_session_waits_for(self):
        class Silent(FakeReactor):
            def _maybe_predict(self) -> None:
                self.waiting_for = "the frames captured with the state rows"

        _, robot, [error] = self.run_client_with(
            Silent, DROID, chunks=1, chunk_timeout_s=0.05
        )
        self.assertIsInstance(error, TimeoutError)
        self.assertIn("step 0", str(error))
        self.assertIn(
            "waiting for: the frames captured with the state rows", str(error)
        )
        self.assertGreaterEqual(robot.holds, 1)

    def test_stale_failure_does_not_end_the_next_episode(self):
        class LateFailure(FakeReactor):
            """The second call never answers in time; its failure lands after
            the client gave up."""

            calls = 0

            def _maybe_predict(self) -> None:
                if self.awaiting is None and self.task and self.state is not None:
                    LateFailure.calls += 1
                    if LateFailure.calls == 2:
                        data = {
                            "step": self.next_step,
                            "code": "rejected",
                            "reason": "",
                        }
                        self._send({"type": "prediction_failed", "data": data}, 0.15)
                        self.fresh = True
                        return
                super()._maybe_predict()

        async def wait():
            await asyncio.sleep(0.3)  # the late failure lands between episodes

        _, _, results = self.run_client_with(
            LateFailure, DROID, chunks=2, episodes=2, chunk_timeout_s=0.1, between=wait
        )
        self.assertIsInstance(results[0], TimeoutError)
        self.assertEqual([len(c.rows) for c in results[1]], [21, 24])

    def test_hold_on_error_and_hold_failure_does_not_hide_it(self):
        class Breaks(SyntheticRobot):
            def execute(self, row):
                if row.index == 5:
                    raise RuntimeError("arm fault")
                super().execute(row)

            def hold(self):
                super().hold()
                if self.holds == 1:
                    raise OSError("hold failed too")

        with self.assertLogs("praxis", "ERROR"):
            _, robot, [error] = self.run_client(DROID, robot=Breaks(DROID))
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "arm fault")
        self.assertEqual(robot.holds, 2)  # on the error, then on exit

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

    def test_frames_are_checked(self):
        def yam(h, w):
            return {name: np.zeros((h, w, 3), np.uint8) for name in YAM.cameras}

        with self.assertRaises(ValueError):
            DROID.check_frame("wrist", np.zeros((360, 640, 3), np.uint8))
        with self.assertRaises(ValueError):
            YAM.check_frame("top", np.zeros((300, 480, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "RGB"):
            DROID.check_frame("wrist", np.zeros((288, 512, 4), np.uint8))  # BGRA
        YAM.check_frames(yam(260, 416), even=True)
        YAM.check_frames(yam(256, 416))
        for h, w in ((234, 416), (250, 400), (416, 416), (300, 416)):
            with self.subTest(size=(w, h)), self.assertRaisesRegex(ValueError, "16:10"):
                YAM.check_frames(yam(h, w))
        with self.assertRaisesRegex(ValueError, "missing"):
            YAM.check_frames({"top": np.zeros((260, 416, 3), np.uint8)})
        odd = {name: np.zeros((287, 511, 3), np.uint8) for name in DROID.cameras}
        DROID.check_frames(odd)  # streamed: any size within the limit
        with self.assertRaisesRegex(ValueError, "even"):
            DROID.check_frames(odd, even=True)

    def test_bad_frames_stop_before_connecting(self):
        class Wide(SyntheticRobot):
            def read_cameras(self):
                return {n: np.zeros((234, 416, 3), np.uint8) for n in YAM.cameras}

        made = []

        def make(model, **kwargs):
            made.append(FakeReactor(model, **kwargs))
            return made[-1]

        async def go():
            async with PraxisClient(Wide(YAM), YAM, tick_s=TICK_S):
                pass

        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            patch.object(praxis_client, "Reactor", make),
            self.assertRaisesRegex(ValueError, "16:10"),
        ):
            asyncio.run(go())
        self.assertFalse(made[0].connected)

    def test_state_is_checked(self):
        good = ArmState([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], 0.0)
        bad = {
            "grip": ArmState(good.ee_pose, float("nan")),
            "quaternion": ArmState([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], 0.0),
            "ee_pose": ArmState([0.4, 0.0, float("inf"), 1.0, 0.0, 0.0, 0.0], 0.0),
        }
        rows = [praxis_client._Row(None, {"arm": good}) for _ in range(STACK)]
        praxis_client.state_json(DROID, rows)
        for what, state in bad.items():
            with self.subTest(what), self.assertRaisesRegex(ValueError, what):
                praxis_client.state_json(
                    DROID, rows[1:] + [praxis_client._Row(None, {"arm": state})]
                )

    def test_demo_prints_each_chunk(self):
        clock = itertools.count(1_000_000, 1000)
        args = argparse.Namespace(
            robot="yam",
            mode="streamed",
            model=None,
            task=None,
            chunks=2,
            episodes=1,
            pose_frame=None,
        )
        real = praxis_client.PraxisClient
        made = []

        def fast(*a, **kw):
            return real(*a, **kw, tick_s=TICK_S)

        def make(model, **kwargs):
            made.append(FakeReactor(model, **kwargs))
            return made[-1]

        out = io.StringIO()
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            patch.object(praxis_client, "Reactor", make),
            patch.object(praxis_client, "time_micros", lambda: next(clock)),
            patch.object(main, "PraxisClient", fast),
            contextlib.redirect_stdout(out),
        ):
            asyncio.run(main.run(args))
        text = out.getvalue()
        self.assertIn("Synthetic robot on reactor/praxis-1-yam, streamed", text)
        self.assertRegex(
            text, r"episode=0 step=0 rows=21 latency=\d+ ms join=capture_time"
        )
        self.assertIn("step=1 rows=24", text)
        self.assertIn("Executed 45 rows; session closed", text)
        self.assertIn("join=capture_time: every state row was paired", text)
        self.assertEqual(made[0].task, main.TASKS["yam"])
        self.assertFalse(made[0].task.endswith("."))

    def test_model_needs_an_owner(self):
        with (
            patch.dict(os.environ, {"REACTOR_API_KEY": "test"}),
            self.assertRaises(ValueError),
        ):
            PraxisClient(SyntheticRobot(DROID), DROID, model="praxis-1-droid")


def no_stamps():
    raise AssertionError("a stepped session must not read time_micros")


class SteppedClientTest(unittest.TestCase):
    def run_client(self, embodiment, chunks=3, episodes=1, fake=None, **kw):
        joins = JoinWarnings()
        logger = logging.getLogger("praxis")
        logger.addHandler(joins)
        try:
            reactor, robot, results, pending = run_session(
                SteppedPraxisClient,
                fake or FakeSteppedReactor,
                embodiment,
                robot=kw.pop("robot", None) or RecordingRobot(embodiment),
                chunks=chunks,
                episodes=episodes,
                **kw,
            )
        finally:
            logger.removeHandler(joins)
        self.assertEqual(joins.joins, [])  # join=newest is not warned
        self.assertEqual(pending, [], "a task outlived the session")
        return reactor, robot, results

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
        # The first read is the camera check before connecting; it is not sent.
        shown = robot.shown[1:]
        for name in DROID.cameras:
            decoded = reactor.window[name]
            # Every observed tick, in order; the last chunk's frames are kept for
            # a step that never comes.
            self.assertEqual(len(decoded), STACK + 21)
            candidates = np.stack([f[name] for f in shown]).astype(np.int16)
            for i, got in enumerate(decoded):
                errors = np.abs(candidates - got.astype(np.int16)).mean(axis=(1, 2, 3))
                # Closest to the frame observed at that tick, not a neighbour: a
                # one-frame shift fails here.
                self.assertEqual(int(errors.argmin()), i, f"{name} frame {i}")
                self.assertLess(errors[i], 4, f"{name}: mean error {errors[i]:.1f}")

    def test_each_row_is_observed_after_its_command(self):
        reactor, _, _ = self.run_client(DROID, robot=EchoRobot(DROID))
        _, second, third = ([p[0] for p in s["ee_pose"]] for s in reactor.sent_states)
        self.assertEqual(second, [-1] * 4 + list(range(21)))
        self.assertEqual(third, [20, *range(24)])

    def test_yam_steps_with_three_cameras_and_both_arms(self):
        reactor, _, [chunks] = self.run_client(YAM, chunks=2)
        self.assertEqual(
            reactor.media_counts[0], {"top": 25, "wrist_left": 25, "wrist_right": 25}
        )
        self.assertEqual(set(reactor.sent_states[-1]), {"left", "right"})
        self.assertEqual(set(chunks[1].rows[0].targets), {"left", "right"})

    def test_second_episode_resets_context(self):
        reactor, robot, results = self.run_client(
            DROID, chunks=2, episodes=2, robot=EchoRobot(DROID)
        )
        self.assertEqual(reactor.commands.count("reset_context"), 1)
        self.assertEqual([c.step for c in results[1]], [2, 3])
        self.assertEqual([len(c.rows) for c in results[1]], [21, 24])
        reset = reactor.commands.index("reset_context")
        self.assertEqual(reactor.commands[reset + 1], "step")
        third = reactor.step_states[2]  # the second episode's first step
        self.assertNotIn("executed_step", third)
        self.assertEqual(reactor.media_counts[2], {"exterior": 25, "wrist": 25})
        # Computed from the 25 still rows after the arm's last command.
        self.assertEqual([p[0] for p in reactor.used_states[2]["ee_pose"]], [23] * 25)
        self.assertEqual(robot.holds, 3)

    def test_failures_raise_prediction_failed(self):
        class Fails(FakeSteppedReactor):
            async def step(self, state=None, commands=None, media=None):
                self.commands.append("step")
                data = {"step": 0, "code": "task_too_long", "reason": "over 64 tokens"}
                self._send({"type": "prediction_failed", "data": data})
                return types.SimpleNamespace(step=1, error=None)

        class Errors(FakeSteppedReactor):
            async def step(self, state=None, commands=None, media=None):
                self.commands.append("step")
                return types.SimpleNamespace(step=1, error="CUDA error")

        _, robot, [error] = self.run_client(DROID, fake=Fails)
        self.assertIsInstance(error, PredictionFailed)
        self.assertEqual((error.step, error.code), (0, "task_too_long"))
        self.assertEqual(robot.holds, 2)
        _, _, [error] = self.run_client(DROID, fake=Errors)
        self.assertIsInstance(error, PredictionFailed)
        self.assertEqual(error.code, "step_failed")
        self.assertIn("CUDA error", str(error))

    def test_timeout_says_what_the_session_waits_for(self):
        class Silent(FakeSteppedReactor):
            async def step(self, state=None, commands=None, media=None):
                self.commands.append("step")
                self.waiting_for = "the model server"
                return types.SimpleNamespace(step=1, error=None)

        _, _, [error] = self.run_client(
            DROID, fake=Silent, chunks=1, chunk_timeout_s=0.05
        )
        self.assertIsInstance(error, TimeoutError)
        self.assertIn("waiting for: the model server", str(error))

    def test_stale_failure_does_not_end_the_next_episode(self):
        class LateFailure(FakeSteppedReactor):
            """The second call outlives the client's timeout, then fails."""

            async def step(self, state=None, commands=None, media=None):
                if len(self.step_states) == 1:
                    self.commands.append("step")
                    self.step_states.append(dict(state))
                    data = {"step": self.next_step, "code": "timeout", "reason": "late"}
                    self._send({"type": "prediction_failed", "data": data}, 0.15)
                    self.awaiting, self.reset = None, True
                    return types.SimpleNamespace(step=2, error=None)
                return await super().step(state, commands, media)

        async def wait():
            await asyncio.sleep(0.3)  # the late failure lands between episodes

        _, _, results = self.run_client(
            DROID,
            fake=LateFailure,
            chunks=2,
            episodes=2,
            chunk_timeout_s=0.1,
            between=wait,
        )
        self.assertIsInstance(results[0], TimeoutError)
        self.assertEqual([len(c.rows) for c in results[1]], [21, 24])

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
            task=None,
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
        self.assertIn("Synthetic robot on reactor/praxis-1-droid, stepped", text)
        self.assertIn("step=0 rows=21", text)
        self.assertIn("step=1 rows=24", text)
        self.assertRegex(text, r"latency=\d+ ms bytes=\d+ join=newest")
        self.assertIn("join=newest: expected in a stepped session", text)
        self.assertIn("Executed 45 rows; session closed", text)


if __name__ == "__main__":
    unittest.main()
