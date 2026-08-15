# GR00T N1.7 quickstart

Get robot actions from the hosted GR00T N1.7 model in minutes.

[`GR00T N1.7`](https://github.com/NVIDIA/Isaac-GR00T) is a
vision-language-action model: a
Qwen3-VL-architecture backbone with a flow-matching action head, served on the
`oxe_droid` embodiment of NVIDIA's
[`GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) checkpoint.
Send two camera views
plus measured state; it emits a 40-step chunk of absolute joint targets every
~100 ms, continuously.

The script drives it with five recorded examples: real Franka Research 3 rig
frames, stored with the chunks this deployment returned.

## Setup

From a clone of this repository, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and create a
[Reactor API key](https://reactor.inc/account/api-keys).

```bash
cd robotics/sim/notebooks && uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python groot_n17_quickstart.py   # then run it
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`groot_n17_quickstart.py`](./groot_n17_quickstart.py) reports only its
length and the API endpoint.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
client = GrootN17Client()      # model="groot-n17", 15 fps tracks
await client.connect()         # handlers -> connect -> await READY -> tracks -> ping
```

Capacity: HTTP 429 `no available capacity` at session creation means none is
free now. Wait and retry, or ask Reactor for more.

This model's wire:

| Direction | Message | Notes |
|---|---|---|
| → | publish tracks `exterior_view`, `wrist_view` | any size; server resizes to 256×256 |
| → | `set_task_description {task_description}` | ≤300 chars |
| → | `set_state_json {state_json}` | ≤2000 chars, every observation |
| → | `reset {}` | clears frame buffers, `step` → 0 |
| ← | `action_prediction {eef_9d, gripper_position, joint_position, step}` | `[40,9]`, `[40,1]`, `[40,7]` |

`state_json` is a dict of named vectors, not a flat `proprio` list:

```
{"eef_9d": [<9 floats>], "gripper_position": [<1>], "joint_position": [<7>]}
```

17 floats across three keys; `eef_9d` is xyz plus a 6D rotation. Send all three
every observation. A key the server cannot parse becomes zeros: one warning per
session, no error, and the policy acts on a fabricated state. `encode_state()`
validates locally instead.

How this differs from the generic
[contract](./robot-policy-client-contract.md):

| Generic contract | GR00T N1.7 |
|---|---|
| `state_json` is `{proprio: [N floats], chunk_id}` | named state vectors, no `chunk_id` |
| one `actions` array `[K, A]` | three named fields |
| `step` echoes your `chunk_id` | the model's own inference counter (0 at reset) |
| one reply per request | free-running broadcast |
| a malformed request is dropped | the bad key becomes zeros |

A generic client will not drive this model unchanged. This guide documents the
model's own protocol, which is stable.

## Get actions

GR00T N1.7 does not wait for requests. Once a task is set and both tracks have
frames, it predicts every engine tick and broadcasts, measured at ~100 ms per
chunk on this deployment. A robot executes the newest chunk. The script wants
the chunk that saw one observation, and nothing links chunk to request, so
`predict()` pairs by engine ordering.

It conditions on a 2-frame window strided 15 engine ticks apart (~1 s at 15
fps, the DROID training-time temporal window), and the window persists across
chunks by design. `predict()` therefore:

1. sends the state, then the frames;
2. holds the observation for `window_s` = `(15 + 1 + margin)/fps`, so every
   slot in the server's 16-frame buffer becomes it;
3. drains the queued chunks, all of which predate the hold;
4. discards one more, and returns the next.

Step 4 is exact because the engine is serial: drain conditioning, one
inference, emit, loop. Chunk *B*'s conditioning was drained after *A* was
emitted, so it falls after the hold. The one assumption: the hold exceeds a
round trip. No model here tags a chunk with its source frame, so ordering is
all a client has.

The actions are absolute joint targets: the checkpoint predicts relatively, the
server converts using that tick's `state_json`. So `joint_position` rows are
poses, not deltas, and row 0 sits near the state you sent, within ≤0.055 rad
over 15 samples at recording time. That check catches broken wiring. Lose the
streamed state and rows go relative, with no error.

### What `predict()` does

The client is
[`reactor_robotics/groot_n17.py`](./reactor_robotics/groot_n17.py). State goes
before frames: the engine drains conditioning at a tick boundary, so it must
already be there when the frames arrive.

## Check the results

The action head samples, so replaying an observation does not reproduce its
chunk; numbers do not compare one-for-one. The check verifies shapes, finite
values, `step` always increasing, FR3 joint limits and gripper range, then two
calibrated tolerances: row 0 within 0.0688 rad of the streamed state, per-step
joint motion within 0.0622 rad (both mean plus three standard deviations at
recording time, see [`examples/PROVENANCE.md`](./examples/PROVENANCE.md)). The
anchor check catches broken wiring.

The L2 distance to the recorded chunks is reported only: its band (2.55, about
15% of a chunk's magnitude) is the model's run-to-run spread, so one value
outside it is normal. Repeated ones suggest the deployment changed.

The script prints a row per check, then the verdict, the line to read:

```
RESULT: PASS
```

Then the likeliest failure modes. A malformed `state_json` is not rejected: the
affected key becomes zeros and the model keeps predicting. A 25 s idle does not
drop the session. `reset` restarts the inference counter.

## Latency

The model's own cadence is the ~100 ms chunk period. The script prints
something else: the pairing cost (one discarded chunk plus the next, about 1.5
chunk periods) on top of the 1.40 s frame-window hold per observation. A robot
pays neither: its cameras fill the window while it executes the previous chunk.

## Run the simulator

There is no closed-loop simulator for this model in this repo today.

| | |
|---|---|
| The frames above | real FR3 rig captures, so the observations are genuine |
| The numbers above | the wire contract and the model's self-consistency, not task success |
| The upstream evaluation | NVIDIA's Isaac-GR00T repository ships real-robot control scripts and publishes offline error metrics; its published success rates are simulation (RoboLab). That harness is upstream, not Reactor's, and is not wired to a Reactor-served policy. |
| What Reactor has | a client for a Franka Research 3 rig, which produced the frames above. Not published in this repository. |
| A closed-loop simulator | does not exist in this repo today |

Two siblings here drive other Reactor-served policies against real simulators,
both the same client with a different frame source:

- [`../cosmos-droid/`](../cosmos-droid): NVIDIA's RoboLab DROID
  benchmark on Isaac Sim, same embodiment as this model, driven through an
  openpi-compatible gateway.
- [`../libero/`](../libero): a real LIBERO/robosuite environment,
  closed-loop. Needs no GPU.

Pointing either at `groot-n17` means porting its observation mapping to this
model's two tracks and 17-dim `state_json`. The transport and pairing are
already what this guide shows.

## Physical deployment

Same client, frames from your cameras, `state_json` from your arm. The protocol
is the same from a camera as from an `.npz`. Chunks are absolute joint targets,
so there is no IK and no unit calibration before a joint command. Reactor's own
FR3 client runs it as its demo path and does command the arm. The capture
behind `examples/groot_n17_examples.npz` is available from your Reactor contact
if you want more than five observations.

To drive an arm:

1. Publish `exterior_view` and `wrist_view` from your two cameras, and send all
   three `state_json` keys every observation.
2. Execute chunks as `predict()` returns them. Do not command an arm from the
   bare `predict()` above; it has none of the guards in step 3.
3. Add each guard below, at the value Reactor's FR3 client uses.
4. Follow [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the wire rules as you build the rig. GR00T N1.7 has a row in its table.

| Guard | Value it uses |
|---|---|
| per-tick joint delta clamp | 0.05 rad (0.75 rad/s at 15 Hz) |
| joint-limit margin | shrinks the FR3 range at each end before clamping |
| action-semantics guard | classifies the first chunk against the measured state, e-stops if it does not look absolute |
| starvation e-stop | no usable chunk for 2 s |
| staleness drop | discards a chunk older than 0.6 s at adoption |
| execution horizon | adopts a new chunk every 8 ticks, starting at row 2 to offset latency already spent. The rig run raised it to 24: at 8, the grasp intents at chunk steps 24-40 never execute |

Reactor support for physical deployment is coming soon.
