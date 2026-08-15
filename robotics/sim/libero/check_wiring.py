"""Exercise the harness end-to-end minus the network: feed a synthetic chunk in,
verify the env executes it and the echo comes back correctly shaped."""

import json

import numpy as np

from libero_sim.contract import (
    ACTION_DOF, ACTION_HORIZON, SEED_SKIP_STEPS, VIEWS,
    decode_chunk, encode_executed,
)
from libero_sim.env import EnvConfig, LiberoEnv
from libero_sim.loop import RolloutState

print("VIEWS (order matters):", list(VIEWS.items()))

env = LiberoEnv(EnvConfig(suite="libero_10", task_id=0))
print("language:", repr(env.language))

rollout = RolloutState(env, exec_steps=ACTION_HORIZON)

frames = {n: rollout.frame_source(n)() for n in VIEWS}
for n, f in frames.items():
    assert f is not None, f"{n} not primed"
    print(f"primed {n}: {f.shape} {f.dtype} contiguous={f.flags['C_CONTIGUOUS']}")

chunk = [[i * 1e-3, 0.0, -0.05, 0.0, 0.0, 0.0, -1.0] for i in range(ACTION_HORIZON)]
assert len(decode_chunk(chunk)) == ACTION_HORIZON
assert decode_chunk("garbage") == []
assert decode_chunk([[1, 2]])[0].shape == (ACTION_DOF,)
print("\ndecode_chunk ok")

assert rollout.is_episode_start(), "should start as episode-start"
rollout.submit_chunk({"action": chunk, "step": 0})
assert not rollout.is_episode_start(), "chunk should clear episode-start"

echo = None
for _ in range(ACTION_HORIZON * 3):
    rollout._tick()
    echo = rollout.take_pending_echo()
    if echo is not None:
        break

assert echo is not None, "no echo produced"
rows = json.loads(echo)
print(f"echo: {len(rows)} steps x {len(rows[0])} dof, {len(echo)} chars")
expected = ACTION_HORIZON - SEED_SKIP_STEPS
assert len(rows) == expected, f"expected {expected} steps, got {len(rows)}"
assert all(len(r) == ACTION_DOF for r in rows)
assert np.allclose(rows[0], chunk[SEED_SKIP_STEPS], atol=1e-4), (
    f"echo mismatch: {rows[0]} vs {chunk[SEED_SKIP_STEPS]}"
)

assert rollout.take_pending_echo() is None, "echo should only be returned once"
assert encode_executed([]) == "", "empty echo must be the empty string"

print("diag:", rollout.diag)
assert rollout.diag.steps_executed == expected
assert rollout.diag.chunks_received == 1

after = rollout.frame_source("agentview")()
moved = not np.array_equal(after, frames["agentview"])
print(f"agentview changed after execution: {moved}")

env.close()
print("\nWIRING OK")
