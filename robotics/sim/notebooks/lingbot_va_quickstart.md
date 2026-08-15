# LingBot-VA quickstart

Get robot actions from the hosted LingBot-VA model in a few minutes.

LingBot-VA is a [causal video-action world model](https://github.com/Robbyant/lingbot-va)
on the LIBERO
embodiment (the LIBERO benchmark's simulated Franka setup). You stream two
camera views and echo back what you executed, and it returns a 16-step, 7-DoF
action chunk (6 end-effector deltas plus a gripper command) in about 208 ms
per chunk on the hosted deployment. The
[paper](https://arxiv.org/abs/2601.21998) reports 98.5% on LIBERO-Long.

The script drives the hosted model with five recorded examples:
observations recorded against this same deployment, stored together with the
chunks it returned at the time. It sends the same observations again and
checks the answers.

## Setup

Two things: a Reactor API key (create one at [reactor.inc/account/api-keys](https://reactor.inc/account/api-keys)) and the
client package.

```bash
cd robotics/sim/notebooks && uv sync --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python lingbot_va_quickstart.py   # then run it
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`lingbot_va_quickstart.py`](./lingbot_va_quickstart.py) reports only
its length and the API endpoint. The optional LIBERO section is the separate
closed-loop benchmark path.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
client = LingbotVaClient()      # model="lingbot-va", 20 fps tracks, 128x128
await client.connect()          # handlers -> connect -> await READY -> tracks -> ping
```

Capacity: HTTP 429 `no available capacity` on session creation means no capacity is free right
now. Wait and retry, or ask Reactor for additional capacity.

This model's wire: two tracks, two commands, one event, one reply type. There
is no proprioception field; the observation is the video.

| Direction | Message | Notes |
|---|---|---|
| → | publish tracks `agentview`, `eye_in_hand` | 128×128, LIBERO's native size. Order matters: mapped positionally to the server's camera keys. |
| → | `set_task_description {task_description}` | ≤300 chars, once per episode |
| → | `set_executed_action_json {executed_action_json: ""}` | clears the echo |
| → | `reset {}` | episode boundary; rebuilds the KV cache (the model's per-episode memory), `step` → 0 |
| ← | `action_prediction {action: [16,7], step}` | the seed chunk, emitted unprompted |
| → | `set_executed_action_json` with the rows you executed | this is what advances the episode |
| ← | `action_prediction {action: [16,7], step}` | one chunk per echo |

`step` is the model's own inference counter, not an echo of anything you
sent; this model has no request id. The protocol is lock-step: one chunk,
then silence until you report what you executed. With no echo sent, nothing
arrives at all.

LingBot-VA's wire departs from the generic
[contract](./robot-policy-client-contract.md): different state fields, no
`chunk_id`, no echoed request id. Read this guide as the authority for this
model.

## Get actions

Five rules of this wire; breaking any of them produces no error.

| # | Rule | What happens if you break it |
|---|---|---|
| 1 | The echo must change value | An echo identical to the previous one reads as "nothing new" and produces no chunk at all. |
| 2 | `reset` takes an empty payload | An unknown field makes it a no-op: `reset {"sampling_seed": 0}` leaves `step` climbing without re-anchoring, while `reset {}` restarts `step` at 0. |
| 3 | Clear the echo before `reset` | Otherwise the previous episode's last echo lands as this episode's first executed chunk. |
| 4 | From the second echo on, send exactly 16 rows | The server reshapes to `(4,4,7)` and drops the exception: the episode stops without an error. |
| 5 | Do not drain before the seed chunk | It arrives unprompted, so clearing your queue first throws away the only chunk you will get, then waits forever for an echo you cannot send. |

The seed chunk is an episode's first, emitted unprompted after `reset`. Its
first 4 rows are a fixed seed: placeholder values, not predictions. Execute
12 of 16. Executing the placeholders instead commands the midpoint of the
training range, which is real motion (`dx` +0.12 four times over).
`predict()` applies the skip for you and exposes the result as
`pred.executable`.

Action rows are deltas in raw LIBERO action units; the server already
un-normalized the model's `[-1, 1]` output through its training quantiles.
Channel order:

```
(dx, dy, dz, droll, dpitch, dyaw, gripper)
```

One raw unit is not a metre. LIBERO drives robosuite's `OSC_POSE` controller,
which rescales `[-1, 1]` to its own `output_max` inside the controller, and
that scale is not part of the served contract. See Physical deployment at the
bottom of this guide before pointing this at hardware.

### What `predict()` does

The client is
[`reactor_robotics/lingbot_va.py`](./reactor_robotics/lingbot_va.py). Two
details matter:

**The frame-window hold.** The server commits a window of video frames per
chunk: nominally 16 (`frame_chunk_size` 4 × `action_per_frame` 4), 12 on the
seed chunk. An observation has to stay on the tracks long enough for that
window to fill with it, or the model predicts from a mix of this observation
and the last one. `window_s` is 20 frame periods, covering the 14-19 frame
jitter of the server's variable commit window. Frames are sent over video
while the echo is sent over the data channel, and nothing ties the two
together, so this ordering is the only thing pairing an observation with its
chunk. A robot never pays the hold: its cameras fill the window while it
executes the previous chunk.

**The seed chunk is different.** It is emitted unprompted after `reset`, so
you neither echo it nor hold the observation window for it, and it may already
be queued by the time you look. Do not drain before it, or you discard the
only chunk you will get.

## Check the results

The model is unseeded, so a replay does not reproduce the recorded numbers.
The check compares the 4 fixed seed rows exactly and verifies the rest
structurally: shape, finite values, increasing `step`, rows inside the
training range, and no duplicate echoes.

The L2 distance to the recorded chunks is reported for reference only: its
band is about 55% of a chunk's own magnitude (the model's own run-to-run
spread at recording time, see
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md)), loose enough to catch
only gross drift.

The script prints a row per check and then its verdict, which is the line to
read:

```
RESULT: PASS
```

After the verdict it runs the two failure modes you are most likely to hit: an
echo identical to the previous one produces no chunk at all, and a 25 s idle
does not drop the session, because `session.py` has been pinging every 10 s
the whole time.

## Latency

Steady-state latency (echo to chunk) measured a p50 of 208 ms at recording
time, the same number on its model card; the seed chunk is timed from `reset`
and also contains the server gathering its first 12 frames, so the two are
never pooled. Per observation the script also pays the 1.00 s frame-window hold,
which is this model's frame budget rather than its latency: a real client is
executing the previous chunk during that window and its cameras fill it as a
side effect. A LIBERO rollout at 20 Hz renders exactly 16 frames while
executing 16 actions.

## Run the simulator

Everything above replayed synthetic frames, so none of it measures task
competence. The paper reports 98.5% on LIBERO-Long with its own client; this
harness, over the production wire, measured 96.4% (482/500). The harness is in
this repository: [`../libero/`](../libero). It
drives a real [LIBERO](https://libero-project.github.io/) (robosuite/MuJoCo)
environment closed-loop, lock-step against this same hosted model, and it is
the working reference for the protocol above.

### Commands

```bash
cd ../libero
uv sync --python 3.10          # LIBERO needs Python 3.10; uv fetches it

# LIBERO is not on PyPI, so clone and install from source.
git clone https://github.com/Lifelong-Robot-Learning/LIBERO vendor/LIBERO
touch vendor/LIBERO/libero/__init__.py      # missing in the upstream tree
uv pip install --no-deps -e vendor/LIBERO

# LIBERO prompts interactively for a config path on first import. Pre-seed it.
mkdir -p .libero
cat > .libero/config.yaml <<EOF
assets: $PWD/vendor/LIBERO/libero/libero/./assets
bddl_files: $PWD/vendor/LIBERO/libero/libero/./bddl_files
benchmark_root: $PWD/vendor/LIBERO/libero/libero
datasets: $PWD/vendor/LIBERO/libero/libero/../datasets
init_states: $PWD/vendor/LIBERO/libero/libero/./init_files
EOF
export LIBERO_CONFIG_PATH="$PWD/.libero"

# Offline check first: no network, no API key. Confirms the LIBERO install
# and the env wrapper before you spend a session on it.
uv run python check_wiring.py

# Then the real thing.
export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
uv run python -m libero_sim.main --task-id 0 --record out.mp4
```

### Expected output

```
libero.env    libero_10 task 0: 'put both the alphabet soup and the tomato sauce
              in the basket' (50 init states)
libero.bridge published track agentview
libero.bridge published track eye_in_hand
libero.bridge re-attached engine with task set
libero        connected; running (Ctrl-C to stop)
libero.loop   episode done (init_state=0 success=True steps=172); ending run
libero        diagnostics: RolloutDiagnostics(chunks_received=11, steps_executed=172,
              echoes_sent=10, episodes=0, successes=1, short_chunks=0, stray_chunks=0)
```

The counts vary with episode length. `episodes` counts re-runs of the same
init state, so a first-try success leaves it at 0.

`--record out.mp4` writes both published camera views side by side, which is
the quickest proof the rollout is really running rather than merely
connected.

### Requirements

| | |
|---|---|
| Install time | 20-40 minutes, mostly the LIBERO clone plus MuJoCo/robosuite wheels |
| GPU | None. MuJoCo renders offscreen on CPU; the policy's GPU is Reactor's |
| Wall-clock per episode | a few minutes; the sim holds still between chunks (lock-step) |
| `stray_chunks` in the diagnostics | should be 0. Non-zero means a chunk landed while the previous one was still executing, so some steps were chosen against a state the sim had already left |

Check these two settings; a mistake produces a wrong result instead of an
error.

- Image orientation. MuJoCo's offscreen renderer returns bottom-up arrays.
  `libero_sim/env.py` applies a vertical flip (`img[::-1]`), not the 180°
  rotation (`img[::-1, ::-1]`) some other LIBERO harnesses use; the two
  differ by a horizontal mirror. Get it wrong and every observation is
  mirrored against the training distribution while the policy keeps emitting
  confident, wrong actions. Only the task success rate will tell you.
- Main-thread env. The env is constructed and stepped on the main thread and
  the Reactor bridge gets its own. On macOS, using MuJoCo's offscreen GL
  context off its creating thread segfaults the process rather than raising.

The harness sends `reset {}` with an empty payload, which is rule 2 above.
`reset {"sampling_seed": 0}` is a no-op that leaves the KV cache intact, so
every episode after the first is predicted against the previous episode's
history, while a task change still re-anchors the episode and masks the
failure.

## Physical deployment

The wire protocol does not change when the frames come from a camera instead
of an `.npz`. To drive an arm:

1. Publish `agentview` and `eye_in_hand` from your two cameras, execute
   `pred.executable`, echo what you executed, and repeat.
2. Supply the `[-1, 1]` → metres scale yourself; it is not in the served
   contract. LIBERO drives robosuite's `OSC_POSE` controller, which rescales
   inside the controller. Robosuite's own defaults
   `(0.05, 0.05, 0.05, 0.5, 0.5, 0.5)` are a starting point for a site
   calibration, not a verified value for this model.
3. Supply IK (inverse kinematics: computing joint angles from an
   end-effector target). The model has no joint channel: its 7 channels are 6
   end-effector deltas plus a gripper, so there is no joint target to send.
4. Follow [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the wire rules while you build the rig.

Steps 2 and 3 are why you cannot command an arm from the bare `predict()`
above: this model is trained on LIBERO in simulation and its chunks are
normalized deltas.

Also expect out-of-distribution behaviour on physical hardware: a sim-trained
policy at 128×128 pointed at a real room is out of distribution by
construction on scene, lighting, camera geometry and rate (16 actions is
0.8 s of LIBERO time at its 20 Hz, against 1.07 s at a 15 Hz client). Reactor
support for physical deployment is coming soon.
