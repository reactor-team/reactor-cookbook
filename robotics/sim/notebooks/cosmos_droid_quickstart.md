# Cosmos-Nano-Policy-DROID quickstart

Get robot actions from the hosted Cosmos-Nano-Policy-DROID model in a few
minutes. This tests that the hosted Reactor deployment is alive and speaking
its API contract correctly -- it says nothing about how well the policy does
tasks. That number comes from the simulator, at the end of this guide.

[`Cosmos3-Nano-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
(NVIDIA) is a vision-language-action policy on the
DROID/Franka embodiment, built on a Cosmos3 world-foundation-model backbone.
You stream three camera views plus proprioception (the robot's measured joint
and gripper state), and it returns a 32-step chunk of absolute joint targets:
2133 ms of motion at DROID's 15 Hz.

The script compares the model's answers against a recorded-example fixture
(observations recorded against this deployment, stored together with the
chunks it returned at the time, see
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md)). If the fixture is
missing it falls back to live checks against the hosted model. The script
detects which mode it is in.

## Setup

From a clone of this repository, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and create a
[Reactor API key](https://reactor.inc/account/api-keys).

```bash
cd robotics/sim/notebooks && uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python cosmos_droid_quickstart.py   # then run it
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`cosmos_droid_quickstart.py`](./cosmos_droid_quickstart.py) reports
only its length and the API endpoint. The optional RoboLab section is the
separate, heavyweight benchmark path.

The script falls back to the recording script's own deterministic observations
when `examples/cosmos_droid_examples.npz` is not present, so it runs either
way. Without calibrated tolerances it runs only the checks that need none and
reports the rest.

If the session connects but every prediction times out, don't assume your
client is wrong. The silence semantics below make a broken deployment look
identical to a bad echo: the model answers nothing in either case. Rule out
your side first -- three tracks published, `proprio_json` parses, the echoed
`step` strictly increased -- and if all three hold, the deployment itself is
not answering, and no client change will fix that. Report it.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
client = CosmosDroidClient()      # model="cosmos-nano-policy-droid", 15 fps
await client.connect()
```

Capacity: one B200 serves one session. HTTP 429
`no available capacity` on session creation means no capacity is free right
now. Wait and retry, or ask Reactor for additional capacity; the script
retries the connect a few times before giving up.

This model's wire: three tracks, three commands, one reply type, and no
`reset` event at all.

| Direction | Message | Notes |
|---|---|---|
| → | publish tracks `wrist_view`, `exterior_view_1`, `exterior_view_2` | the checkpoint's declaration order |
| → | `set_task_description {task_description}` | ≤300 chars, on change |
| → | `set_proprio_json {proprio_json}` | ≤8000 chars, per chunk |
| → | `set_executed_step_json {executed_step_json}` | ≤8000 chars, per chunk |
| ← | `action_prediction {action: [32,8], step}` | 7 absolute joint positions (rad) + gripper |

There is no episode state to manage: no KV cache, no `reset`. The task and
the proprio are sent with every prediction, so a new episode or a task change
needs nothing extra: just call `predict()` with the new task.

## Get actions

The first prediction needs only task + proprio + a full frame set. Every one
after it needs you to report what you executed:

```
proprio_json         {"joint_position": [[<7 floats>], ...],
                      "gripper_position": [[<float>], ...]}
executed_step_json   {"step": <int>, "action": [[...]]}
```

Both proprio keys are lists of rows so you can send a short history in one
update; the last row is the current state.

The model answers only when the echoed `step` is strictly greater than the
last value it advanced on. A repeat, a lower value, or malformed JSON
produces no chunk and no error. This stops the model running ahead of a
client that is still executing, and stops a stalled control loop's retry from
causing a spurious re-prediction. The reply's own `step` is the model's
prediction counter from 0; echo the `step` of the chunk you executed and the
counters line up, which is what `predict()` does. A full 32×8 chunk echo
serialises to about 5100 characters, well inside the field's 8000-char limit.

The actions are 7 absolute joint positions in radians plus a gripper command,
in DROID's joint-position convention. Rows are poses, not deltas, and row 0
sits near the proprio you sent, which is exactly what the check below
confirms.

Two ways this differs from the generic
[contract](./robot-policy-client-contract.md):

| Generic contract | Cosmos-Nano-Policy-DROID |
|---|---|
| `state_json` is `{proprio: [N floats], chunk_id}` | `proprio_json` (row lists) + `executed_step_json`, no `chunk_id` |
| `step` echoes your `chunk_id` | `step` is the model's own prediction counter; you echo it back |

Everything else matches the generic contract: lock-step, one chunk per
request, stateless.

### What `predict()` does

The client is
[`reactor_robotics/cosmos_droid.py`](./reactor_robotics/cosmos_droid.py). Two
orderings matter:

- Frames and proprio before the echo. The model predicts only when it holds
  a full frame set and proprio it can parse, and it snapshots both on the
  tick it predicts. If the echo lands first, the prediction comes from the
  previous observation.
- Drain before echoing. An answer to an earlier, timed-out request can still
  arrive late. Draining first keeps that stale chunk from being taken as the
  answer to this request.

## Check the results

The policy samples, so replaying an observation does not reproduce its chunk
and the returned numbers cannot be compared one-for-one. The check verifies:
correct shape, finite values, `step` always increasing, joint targets inside
FR3 limits, gripper near [0, 1], and row 0 close to the proprio you sent.

The gripper is checked against [0, 1] with a small tolerance, not as a hard
bound: the wire relays the model's raw sampled output, and gripper training
data sits at the bounds (open/closed), so raw output lands slightly outside
them about as often as slightly inside. Clamp before actuating (see Physical
deployment below); a large excursion, not a small one, is the regression
signal.

The anchor and per-step tolerances are calibrated at recording time (the mean
plus three standard deviations over 3 passes of the same observations, see
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md)). Without the recorded
fixture they are unknown, so those two numbers are reported without a
pass/fail: every tolerance comes from a measurement. The L2 distance to the
recorded chunks is reported the same way. Its band is the model's own
run-to-run spread, so a single value outside the band is normal. Repeated
values outside it suggest the deployment has changed.

The script prints a row per check and then its verdict, which is the line to
read:

```
RESULT: PASS
```

After the verdict it runs the failure modes you are most likely to hit: a
non-increasing echoed `step` produces no chunk, a 25 s idle does not drop the
session, and a task change needs no ceremony because the model is stateless.

## Latency

One chunk is 2133 ms of motion (32 rows at 15 Hz), so the number that decides
whether this policy can drive a robot in real time is headroom against that
budget. The reference figures for this deployment are ~568 ms of model
compute per chunk and ~745 ms p50 think + wire; both are reference points,
not guarantees, and the p50 is also quoted in
[`../cosmos-droid/README.md`](../cosmos-droid/README.md). Whole-chunk
open-loop execution is the measured optimum for this policy, not a
compromise: success strictly increases with open-loop horizon and mid-chunk
replanning collapses it.

## Run the simulator

Everything above replayed synthetic frames, so none of it measures task
competence. That number comes from NVIDIA's RoboLab DROID benchmark on Isaac
Sim, and the harness that measures it is in this repository:
[`../cosmos-droid/`](../cosmos-droid).

The gateway leaves RoboLab unmodified. Isaac Sim owns its own process, python
and episode loop, and the seam it exposes for a remote policy is an openpi
WebSocket port; the example is that port, relaying each request to this
hosted model. Keeping the simulator unmodified is what makes any success rate
you measure comparable to RoboLab's own published numbers.

Served in-process, cosmos-nano-policy-droid scored 40.0% over RoboLab's 120
tasks × 3 rollouts, a Reactor measurement. RoboLab's
[current leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)
reports 36.8% (441/1200) under default language specificity; the different
rollout counts make that an external anchor, not a parity comparison. Over the
production wire it made 8/10 solves with a p50 of 745 ms think+wire and 0
stalls in 150 chunks. The same Reactor figures are in
[`../cosmos-droid/README.md`](../cosmos-droid/README.md).

### Commands

```
# 1. the gateway (this repo; host side, or any machine RoboLab can reach)
cd ../cosmos-droid
uv sync --python 3.12

uv run python check_wiring.py          # offline: no GPU, no network, numpy only

export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
uv run python -m cosmos_droid_sim.main --port 8000

# 2. RoboLab, unmodified, in its own container
docker run --rm --entrypoint /isaac-sim/python.sh --net host --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v $HOME/RoboLab:/workspace/robolab -w /workspace/robolab \
  robolab:<tag> policies/cosmos3/run.py --task BananaInBowlTask \
  --headless --num-envs 1 --num-runs 1 \
  --remote-host localhost --remote-port 8000
```

RoboLab writes the scored episode (a success boolean plus videos) to
`output/<ts>_cosmos3/<task>/`.

Note the API URL: `cosmos-droid` defaults to the production API,
`https://api.reactor.inc`. Set `REACTOR_API_URL` or pass `--api-url` to point
it somewhere else.

### Expected output

```
cosmos_droid.bridge published track wrist_view
cosmos_droid.bridge published track exterior_view_1
cosmos_droid.bridge published track exterior_view_2
cosmos_droid.bridge relay running
cosmos_droid.bridge task: 'put the banana in the bowl'
... one chunk per RoboLab request, ~2.1 s apart ...
```

### Requirements

| | |
|---|---|
| GPU | An RTX-class GPU is required for Isaac's renderer. L40S validated. Datacenter A100/H100/B200 have no graphics engines and cannot render at all. |
| Driver | NVIDIA 580.x. 595.x segfaults Isaac at boot. |
| Container | the `robolab` image built with Isaac 5.1 |
| Install time | hours, dominated by Isaac Sim and the RoboLab assets |
| Wall-clock | ~2.1 s per chunk, so a 120-task × 3-rollout sweep is a long run |

These requirements do not fit in a `pyproject.toml`, which is why they are
listed here rather than installed.

If predictions look one observation stale: frames are sent over WebRTC video,
commands are sent over the data channel, and the model pairs whichever frame
is newest with the request. `--settle` (default 0.1 s) pauses between pushing
frames and sending the echo so the fresh frame lands first. Raise it. This
reduces the risk but does not remove it: none of these models tags a chunk
with the frame it came from, so timing is the only thing pairing them.

## Physical deployment

Two properties of this model carry over to hardware directly: its chunks
are absolute joint targets (no IK and no unit calibration between the
model's output and a joint command), and each prediction is stateless (a
reconnect, a retry or a new episode cannot corrupt anything, because there
is nothing to corrupt).

To drive an arm:

1. Publish `wrist_view`, `exterior_view_1` and `exterior_view_2` from your
   three cameras, at the same track names and order used above.
2. Send the arm's measured joint and gripper state as `proprio_json`, and
   report what you executed as `executed_step_json`, exactly as the script
   does. The wire does not change when the frames come from a camera instead
   of an `.npz`.
3. Add the safety layer `predict()` does not have: joint clamps, a
   joint-limit margin, an action-semantics guard, and a starvation e-stop.
   Do not command an arm from the bare `predict()` above; it has none of
   these.
4. Follow [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the wire rules while you build the rig.

Reactor support for physical deployment is coming soon.
