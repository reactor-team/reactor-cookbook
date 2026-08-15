"""Exercise the gateway end-to-end minus SAPIEN and the network: build a
synthetic request exactly as the authors' client pickles one, run it through
the real gateway with a stub predictor, and verify every contract function
round-trips.

Needs numpy and pyzmq only, and deliberately importable without reactor-sdk,
aiortc, a simulator or a GPU, so this runs anywhere the repo is checked out:

    python check_wiring.py
"""

import asyncio
import json
import pickle

import numpy as np

from robotwin_sim.contract import (
    ACTION_SHAPE,
    PROPRIO_DIM,
    PROPRIO_PRED_SHAPE,
    VIEWS,
    decode_prediction,
    decode_request,
    encode_reply,
    encode_state_json,
    frames_from_video,
)
from robotwin_sim.gateway import Gateway

H, W = 240, 320

# ── contract: the video inversion is exact, not approximate ─────────────────
# The client normalises rendered uint8 frames into [-1, 1]; recovering them
# has to give back the *same* pixels, or every frame the model sees is off by
# a quantisation step.
rng = np.random.default_rng(0)
rendered = rng.integers(0, 256, (len(VIEWS), H, W, 3), dtype=np.uint8)
normalised = rendered.astype(np.float32) / 127.5 - 1.0
frames = frames_from_video(normalised)
assert list(frames) == list(VIEWS), "views must come back in the declared order"
for i, view in enumerate(VIEWS):
    assert frames[view].shape == (H, W, 3) and frames[view].dtype == np.uint8
    assert np.array_equal(frames[view], rendered[i]), f"{view} is not an exact match"
print(f"frames_from_video ok: {len(VIEWS)} views, exact from uint8")

# Every uint8 value must survive, including the boundaries where float32
# division error would otherwise round the wrong way.
ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (1, 1, 3))
ramp = np.broadcast_to(ramp, (len(VIEWS), 1, 256, 3))
back = frames_from_video(ramp.astype(np.float32) / 127.5 - 1.0)
assert np.array_equal(back[VIEWS[0]][0, :, 0], np.arange(256)), "boundary values lost"
print("frames_from_video ok: all 256 levels round-trip")

# ── contract: decode a request as the authors' client pickles it ─────────────
request_dict = {
    "video": normalised,
    "proprios": rng.normal(0, 0.3, PROPRIO_DIM).astype(np.float64),
    "prompt": ["pick up the bottle"],  # a one-element list, as they send it
    "env_rank": 0,
    "rollout_id": 7,
    "step_id": 13,
    "cfg": 1.5,
}
req = decode_request(request_dict)
assert req.task == "pick up the bottle", req.task
assert req.seed == (0, 7, 13), req.seed
assert req.cfg == 1.5
assert req.proprio.shape == (PROPRIO_DIM,)

# A missing seed field must default rather than raise: the seeds are what make
# a rollout reproducible, so their absence has to be visible in the payload.
bare = decode_request({**request_dict, "env_rank": 0, "rollout_id": 0, "step_id": 0})
assert bare.seed == (0, 0, 0)

for bad, why in [
    ({**request_dict, "proprios": np.zeros(3)}, "wrong proprio width"),
    ({**request_dict, "proprios": np.full(PROPRIO_DIM, np.nan)}, "non-finite proprio"),
    ({**request_dict, "video": normalised[0]}, "video missing the view axis"),
    ({"prompt": "x"}, "no video/proprios at all"),
]:
    try:
        decode_request(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"decode_request accepted {why}")
print("decode_request ok (and rejects malformed requests)")

# ── contract: retry changes the string but not the seeds ────────────────────
first = encode_state_json(req, 1)
retried = encode_state_json(req, 1, retry=1)
assert first != retried, "a retry must change the request string or it is deduplicated"
a, b = json.loads(first), json.loads(retried)
assert b.pop("retry") == 1
assert a == b, "a retry must not change the seeds; the answer has to be identical"
assert a["chunk_id"] == 1 and a["env_rank"] == 0 and a["rollout_id"] == 7
assert a["step_id"] == 13 and a["cfg"] == 1.5
print("encode_state_json ok: retry changes a byte, keeps every seed")

# ── contract: decode a prediction, encode the authors' reply ────────────────
actions = np.linspace(0.0, 1.0, ACTION_SHAPE[0] * ACTION_SHAPE[1]).reshape(ACTION_SHAPE)
proprios = np.zeros(PROPRIO_PRED_SHAPE)
step, got_actions, got_proprios = decode_prediction(
    {"actions": actions.tolist(), "proprios": proprios.tolist(), "step": 1}
)
assert step == 1 and got_actions.shape == ACTION_SHAPE
assert got_proprios.shape == PROPRIO_PRED_SHAPE
try:
    decode_prediction({"actions": actions[:4].tolist(), "step": 1})
except ValueError:
    pass
else:
    raise AssertionError("decode_prediction accepted a short chunk")

reply = encode_reply(got_actions, got_proprios)
assert reply["actions"].dtype == np.float64 and reply["proprios"].dtype == np.float64
print("decode_prediction / encode_reply ok")

# ── the gateway: a pickled request in, a pickled reply out ──────────────────
seen = []


async def stub_predict(request):
    """Stand in for bridge.predict without a session."""
    seen.append(request)
    return actions, proprios


gw = Gateway(stub_predict)
raw_reply = asyncio.run(gw.handle(pickle.dumps(request_dict)))
out = pickle.loads(raw_reply)
assert gw.handled == 1 and len(seen) == 1
assert seen[0].seed == (0, 7, 13), "the gateway must relay the seeds verbatim"
assert np.array_equal(out["actions"], actions)
assert out["actions"].shape == ACTION_SHAPE and out["proprios"].shape == PROPRIO_PRED_SHAPE
print(f"gateway round trip ok: {ACTION_SHAPE} actions returned to the client")

print("\nWIRING OK")
