"""Exercise the gateway end-to-end minus Isaac and the network: feed a
synthetic RoboLab observation in on one thread, resolve it like the bridge
would on another, and verify every contract function round-trips.

Needs only numpy, and is deliberately importable without reactor-sdk,
aiortc, or a GPU, so this runs anywhere the repo is checked out."""

import json
import threading

import numpy as np

from cosmos_droid_sim.contract import (
    ACTION_DOF, ACTION_HORIZON, TRACKS,
    decode_chunk, encode_executed_step, encode_proprio, split_composite,
)
from cosmos_droid_sim.gateway import GatewayState

# ── contract: composite split ───────────────────────────────────────────────
H, W = 540, 640  # RoboLab's composite: wrist 360x640 on top, 2x 180x320 below
comp = np.arange(H * W * 3, dtype=np.uint32).reshape(H, W, 3).astype(np.uint8)
views = split_composite(comp)
assert list(views) == list(TRACKS), "views must come back in the declared track order"
assert views["wrist_view"].shape == (360, 640, 3)
assert views["exterior_view_1"].shape == (180, 320, 3)
assert views["exterior_view_2"].shape == (180, 320, 3)
assert np.array_equal(views["wrist_view"], comp[:360])
assert np.array_equal(views["exterior_view_2"], comp[360:, 320:])
print("split_composite ok:", {k: v.shape for k, v in views.items()})

# ── contract: proprio + echo + chunk decode ─────────────────────────────────
obs = {
    "observation/image": comp,
    "observation/joint_position": [0.1, -0.2, 0.3, -1.5, 0.0, 1.2, 0.7],
    "observation/gripper_position": [0.5],
    "prompt": "put the banana in the bowl",
}
proprio = json.loads(encode_proprio(obs))
assert proprio["joint_position"] == [[0.1, -0.2, 0.3, -1.5, 0.0, 1.2, 0.7]]
assert proprio["gripper_position"] == [[0.5]]

chunk = np.linspace(0.0, 1.0, ACTION_HORIZON * ACTION_DOF).reshape(ACTION_HORIZON, ACTION_DOF)
step, action = decode_chunk({"data": {"action": chunk.tolist(), "step": 7}})
assert step == 7 and action.shape == (ACTION_HORIZON, ACTION_DOF)
assert decode_chunk({"type": "state_update", "data": {}}) is None
assert decode_chunk("not json {") is None
echo = json.loads(encode_executed_step(step, action))
assert echo["step"] == 7 and len(echo["action"]) == ACTION_HORIZON
print("proprio / chunk / echo ok")

# ── gateway hand-off: infer() blocks until a 'bridge' resolves it ───────────
gw = GatewayState(chunk_timeout_s=5.0)


def fake_bridge():
    req = gw.take_pending(timeout=5.0)
    assert req is not None, "bridge never saw the request"
    assert req.task == obs["prompt"]
    assert json.loads(req.proprio_json) == proprio
    req.step, req.action = 7, chunk
    req.done.set()


t = threading.Thread(target=fake_bridge)
t.start()
reply = gw.infer(obs)
t.join(timeout=5.0)
assert np.array_equal(reply["action"], chunk)
assert gw.diag.requests == 1 and gw.diag.chunks_returned == 1 and gw.diag.last_step == 7

frame, seq = gw.frame_reader("wrist_view")()
assert frame is not None and frame.shape == (360, 640, 3) and seq == 1
print("gateway hand-off ok:", gw.diag)

print("\nWIRING OK")
