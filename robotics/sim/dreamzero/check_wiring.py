"""Exercise the gateway end-to-end minus Isaac and the model: drive the real
WebSocket protocol server over loopback with a synthetic RoboLab request, and
verify every contract function round-trips.

Needs numpy, msgpack and websockets only, and is deliberately importable
without reactor-sdk, aiortc, a simulator or a GPU, so this runs anywhere the
repo is checked out:

    python check_wiring.py [port]

A green run means the gateway is protocol-correct and the observation mapping
survives a round trip. It says nothing about task success, which needs the
real simulator.
"""

import asyncio
import sys

import numpy as np
import websockets.asyncio.client

from dreamzero_sim.contract import (
    ACTION_SHAPE,
    FRAME_HW,
    JOINT_DIM,
    OBS_KEY_TO_TRACK,
    TRACKS,
    action_reply,
    decode_chunk,
    extract_frames,
    extract_state,
)
from dreamzero_sim.gateway import RoboLabPolicy
from dreamzero_sim.policy_server import PolicyServerConfig, WebsocketPolicyServer
from dreamzero_sim import msgpack_numpy

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18991
H, W = FRAME_HW
rng = np.random.default_rng(0)

# ── contract: the camera mapping, which is the trap in this example ─────────
# RoboLab numbers exteriors from 0, the checkpoint from 1. Getting this
# backwards feeds the model a black primary view and does not error.
left = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
wrist = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
black = np.zeros((H, W, 3), dtype=np.uint8)
obs_frames = {
    "observation/exterior_image_0_left": left,
    "observation/exterior_image_1_left": black,
    "observation/wrist_image_left": wrist,
}
frames = extract_frames(obs_frames)
assert list(frames) == list(TRACKS), "frames must come back in the declared order"
assert np.array_equal(frames["exterior_1"], left), (
    "RoboLab's exterior_image_0_left (its REAL left camera) must land on "
    "exterior_1, not exterior_2"
)
assert np.array_equal(frames["exterior_2"], black)
assert np.array_equal(frames["wrist"], wrist)
assert all(f.dtype == np.uint8 for f in frames.values())
print("extract_frames ok:", {k: v.shape for k, v in frames.items()})

# A camera the client omits becomes black, matching the training-time dropout.
partial = extract_frames({"observation/exterior_image_0_left": left})
assert np.array_equal(partial["exterior_1"], left)
assert not partial["exterior_2"].any() and not partial["wrist"].any()

# A batched temporal stack keeps the newest frame only.
stacked = np.stack([black, left])
batched = extract_frames({**obs_frames, "observation/wrist_image_left": stacked})
assert np.array_equal(batched["wrist"], left), "a 4-d stack must take the newest frame"

for bad, why in [
    ({}, "no camera frames at all"),
    ({"observation/wrist_image_left": np.zeros((H, W))}, "a 2-d frame"),
]:
    try:
        extract_frames(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"extract_frames accepted {why}")
print("extract_frames ok: black fill, newest-of-stack, malformed rejected")

# ── contract: state extraction, clamping and the zero-state meaning change ──
joints, gripper = extract_state(
    {
        "observation/joint_position": np.arange(JOINT_DIM, dtype=np.float32),
        "observation/gripper_position": np.array([0.4], dtype=np.float32),
    }
)
assert joints == [float(i) for i in range(JOINT_DIM)], joints
assert abs(gripper - 0.4) < 1e-6, gripper
# The model clamps to [0, 1]; clamp here so a rounding overshoot cannot make
# it reject the whole state update.
assert extract_state({"observation/gripper_position": [1.2]})[1] == 1.0
assert extract_state({"observation/gripper_position": [-0.1]})[1] == 0.0
# Missing state falls back to zeros, which silently changes the predicted
# joints from absolute targets to relative deltas, so the gateway logs it.
assert extract_state({}) == ([0.0] * JOINT_DIM, 0.0)
try:
    extract_state({"observation/joint_position": [np.nan] * JOINT_DIM})
except ValueError:
    pass
else:
    raise AssertionError("extract_state accepted non-finite joints")
print("extract_state ok: 7 joints + clamped gripper, zeros when absent")

# ── contract: a chunk without obs_seq must be refused, not guessed at ───────
chunk = np.linspace(0.0, 1.0, ACTION_SHAPE[0] * ACTION_SHAPE[1]).reshape(ACTION_SHAPE)
actions, obs_seq, chunk_index, seconds = decode_chunk(
    {
        "actions": chunk.tolist(),
        "obs_seq": 12,
        "chunk_index": 3,
        "inference_seconds": 0.26,
    }
)
assert actions.shape == ACTION_SHAPE and obs_seq == 12 and chunk_index == 3
assert abs(seconds - 0.26) < 1e-9
try:
    decode_chunk({"actions": chunk.tolist()})
except RuntimeError:
    pass  # no obs_seq: a fresh chunk cannot be told from one in flight
else:
    raise AssertionError("decode_chunk accepted a chunk with no obs_seq")
try:
    decode_chunk({"actions": np.full(ACTION_SHAPE, np.nan).tolist(), "obs_seq": 1})
except ValueError:
    pass
else:
    raise AssertionError("decode_chunk accepted non-finite actions")
print("decode_chunk ok: obs_seq required, non-finite rejected")

assert action_reply(chunk)["actions"].dtype == np.float32
assert action_reply(chunk[0])["actions"].shape == (1, ACTION_SHAPE[1])

# ── the msgpack codec, which has to match RoboLab's byte for byte ───────────
round_tripped = msgpack_numpy.unpackb(msgpack_numpy.packb({"a": left, "b": 0.5}))
assert np.array_equal(round_tripped["a"], left) and round_tripped["a"].dtype == np.uint8
assert round_tripped["b"] == 0.5
print("msgpack_numpy ok: uint8 frames survive the round trip")


# ── the protocol, over a real socket, with a stub bridge ────────────────────
class StubBridge:
    """Stands in for bridge.Bridge without a session or a model."""

    class Diag:
        requests = 0
        resets = 0

    def __init__(self):
        self.diag = StubBridge.Diag()
        self.seen = []
        self.resets = []

    async def predict(self, frames, joints, gripper, prompt):
        self.seen.append((frames, joints, gripper, prompt))
        self.diag.requests += 1
        return chunk

    async def reset_episode(self, reason):
        self.resets.append(reason)


def build_request(session_id: str) -> dict:
    """One infer request, keyed exactly as RoboLab's client packs one."""
    return {
        **obs_frames,
        "observation/joint_position": rng.normal(0, 0.3, JOINT_DIM).astype(np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.array([0.4], dtype=np.float32),
        "prompt": "pick up the marker and put it in the cup",
        "session_id": session_id,
        "endpoint": "infer",
    }


async def exercise_protocol() -> None:
    stub = StubBridge()
    policy = RoboLabPolicy(stub)
    server = WebsocketPolicyServer(
        policy=policy, server_config=PolicyServerConfig(), host="127.0.0.1", port=PORT
    )
    task = asyncio.create_task(server.run())
    await asyncio.sleep(0.4)  # let the listener bind

    try:
        async with websockets.asyncio.client.connect(
            f"ws://127.0.0.1:{PORT}", compression=None, max_size=None
        ) as ws:
            # The handshake the client blocks on before its first query.
            metadata = msgpack_numpy.unpackb(await ws.recv())
            assert metadata["action_space"] == "joint_position", metadata
            assert metadata["n_external_cameras"] == 2, metadata
            assert metadata["needs_wrist_camera"] is True, metadata
            assert tuple(metadata["image_resolution"]) == FRAME_HW, metadata

            for query, session in enumerate(["s-1", "s-1", "s-2"]):
                await ws.send(msgpack_numpy.packb(build_request(session)))
                raw = await ws.recv()
                assert not isinstance(raw, str), f"server returned an error:\n{raw}"
                reply = msgpack_numpy.unpackb(raw)
                got = np.asarray(reply["actions"])
                assert got.shape == ACTION_SHAPE, got.shape
                assert np.isfinite(got).all()
                assert np.allclose(got, chunk, atol=1e-6)
                print(f"  query {query} (session {session}): {got.shape} returned")

            # A changed session id has to end the episode, or the model would
            # carry its causal cache into the next one.
            assert len(stub.resets) == 1, stub.resets
            assert "s-2" in stub.resets[0]

            # ... and so does RoboLab's reset endpoint.
            await ws.send(
                msgpack_numpy.packb({"endpoint": "reset", "session_ids": ["s-2"]})
            )
            assert await ws.recv(), "the client blocks unless reset replies"
            assert len(stub.resets) == 2, stub.resets
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert stub.diag.requests == 3
    # The mapping the bridge would have published, checked at the far end of
    # the whole round trip rather than in isolation.
    published, joints_seen, _, prompt_seen = stub.seen[0]
    assert np.array_equal(published["exterior_1"], left)
    assert len(joints_seen) == JOINT_DIM
    assert prompt_seen.startswith("pick up the marker")
    print(f"protocol round trip ok: 3 queries + 2 episode resets on :{PORT}")


print(f"exercising the openpi protocol over loopback on :{PORT} ...")
asyncio.run(exercise_protocol())

assert set(OBS_KEY_TO_TRACK.values()) == set(TRACKS)
print("\nWIRING OK")
