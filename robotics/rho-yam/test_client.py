# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
import asyncio
import json

import numpy as np
from PIL import Image
import pytest

import rho_client
from rho_client import RhoClient, observation, validate_reply, TRACKS
from run_policy import demo_observation, load_observation


def reply(request_id=0, stamp=10):
    _, state = demo_observation()
    return {"step": request_id, "source_capture_us": stamp, "view_skew_us": 0,
            "execution_horizon": 25, "actions": np.tile(state, (50, 1)).tolist()}


@pytest.mark.parametrize("field,value", [
    ("step", 9), ("source_capture_us", 11), ("view_skew_us", 1),
    ("execution_horizon", 50), ("actions", [[0] * 20] * 49),
    ("actions", [[float("nan")] * 20] * 50), ("actions", [[0] * 20] * 50),
])
def test_bad_reply_rejected(field, value):
    body = reply()
    body[field] = value
    with pytest.raises(ValueError):
        validate_reply(body, 0, 10)


def test_raw_gripper_outputs_preserved():
    body = reply()
    body["actions"][0][19] = 1.3
    assert validate_reply(body, 0, 10)[0, 19] == 1.3


def test_observations_are_snapshots_and_input_units_are_checked():
    frames, state = demo_observation()
    frozen_frames, frozen_state = observation(frames, state)
    frames[TRACKS[0]][:] = 255
    state[0] = 99
    assert frozen_frames[TRACKS[0]][0, 0, 0] != 255
    assert frozen_state[0] != 99
    for bad in (np.zeros(14), np.full(20, np.nan), np.zeros(20)):
        with pytest.raises(ValueError):
            observation(frames, bad)
    state[19] = 2
    with pytest.raises(ValueError, match="gripper"):
        observation(frames, state)
    frames, state = demo_observation()
    frames[TRACKS[0]] = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB"):
        observation(frames, state)
    with pytest.raises(ValueError, match="camera names"):
        observation({}, state)


def test_saved_observation_loading(tmp_path):
    frames, state = demo_observation()
    for name, frame in frames.items():
        Image.fromarray(frame).save(tmp_path / f"{name}.png")
    (tmp_path / "state.json").write_text(json.dumps(state.tolist()))
    loaded, values = load_observation(tmp_path)
    np.testing.assert_array_equal(values, state)
    for name in TRACKS:
        np.testing.assert_array_equal(loaded[name], frames[name])


class FakeTrack:
    def __init__(self):
        self.stamps = []

    async def publish(self):
        return self

    def push_frame(self, frame, *, capture_time_us):
        self.stamps.append(capture_time_us)


class FakeReactor:
    instances = []
    response_mode = "valid"

    def __init__(self, *args, **kwargs):
        self.commands = []
        self.tracks = {name: FakeTrack() for name in TRACKS}
        self.session_id = "test-session"
        self.closed = self.disconnected = False
        self.instances.append(self)

    def on_message(self, callback):
        self.callback = callback

    async def connect(self):
        pass

    async def request_schema(self):
        return {"paths": {name: {"post": {"operationId": name}} for name in
                          ("reset", "set_task_description", "set_state_json")}}

    def track(self, name):
        return self.tracks[name]

    async def send_command(self, name, payload):
        self.commands.append((name, payload))
        if name == "set_state_json":
            request = json.loads(payload["state_json"])
            if self.response_mode == "silent":
                return
            if self.response_mode == "error":
                self.callback({"type": "command_error", "data": "invalid state"})
                return
            body = reply(request["chunk_id"], request["capture_us"])
            body["actions"][0][19] = 1.3
            if self.response_mode == "stale":
                body["step"] = -1
            self.callback({"type": "action_prediction", "data": body})

    async def disconnect(self):
        self.disconnected = True

    def close(self):
        self.closed = True


@pytest.fixture
def fake_service(monkeypatch):
    FakeReactor.instances = []
    monkeypatch.setattr(FakeReactor, "response_mode", "valid")
    monkeypatch.setattr(rho_client, "Reactor", FakeReactor)
    monkeypatch.setenv("REACTOR_API_KEY", "test-only")
    async def no_sleep(seconds):
        pass
    monkeypatch.setattr(rho_client.asyncio, "sleep", no_sleep)
    return FakeReactor


def test_two_predictions_are_correlated_and_session_closes(fake_service):
    async def exercise():
        frames, state = demo_observation()
        async with RhoClient("pick up block") as client:
            for i in range(2):
                state[0] += 0.01
                result = await client.predict(frames, state, seed=100 + i)
                assert result["request"]["chunk_id"] == i
                assert result["request"]["seed"] == 100 + i
                assert result["request"]["proprio"] == state.tolist()
                assert result["invalid_gripper_rows"] == [0]
                assert result["prediction"]["actions"][0][19] == 1.3
                assert result["actions_executed"] == 0
        server = fake_service.instances[-1]
        assert server.closed and server.disconnected
        assert [name for name, _ in server.commands] == [
            "reset", "set_task_description", "set_state_json", "set_state_json"]
        assert server.tracks[TRACKS[0]].stamps == server.tracks[TRACKS[1]].stamps
        assert server.tracks[TRACKS[0]].stamps == server.tracks[TRACKS[2]].stamps
    asyncio.run(exercise())


@pytest.mark.parametrize("mode,error", [("silent", TimeoutError), ("stale", ValueError), ("error", RuntimeError)])
def test_protocol_failure_prevents_session_reuse(fake_service, monkeypatch, mode, error):
    monkeypatch.setattr(fake_service, "response_mode", mode)
    async def exercise():
        frames, state = demo_observation()
        async with RhoClient("pick up block", timeout=0.01) as client:
            with pytest.raises(error):
                await client.predict(frames, state)
            with pytest.raises(RuntimeError, match="fresh"):
                await client.predict(frames, state)
        assert fake_service.instances[-1].closed
    asyncio.run(exercise())


def test_key_has_no_local_file_fallback(monkeypatch):
    monkeypatch.delenv("REACTOR_API_KEY", raising=False)
    with pytest.raises(ValueError, match="REACTOR_API_KEY"):
        rho_client.api_key()
