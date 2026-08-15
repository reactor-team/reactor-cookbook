# X-WAM quickstart

Get robot actions from the hosted X-WAM model in a few minutes.

X-WAM is a bimanual manipulation policy: a Wan2.2-TI2V-5B world-action model
(it predicts video and actions together) fine-tuned on RoboTwin 2.0. You
stream three camera views and the robot's current pose, and it returns a
32-step, 14-DoF action chunk in about 163 ms per chunk on the hosted
deployment, 1.4× faster than the 229 ms for the authors' own stack on the
same GPU (a Reactor measurement).

The script drives the hosted model with five recorded examples: real
observations captured during a RoboTwin 2.0 evaluation run, stored together
with the actions the model returned at the time. The script sends the same
observations again and checks that the answers match. The client in
`reactor_robotics/` is the same client an evaluation harness or a controller
for a physical robot uses; the wire protocol does not change when the frames
start coming from a camera instead of a recorded file.

## Setup

Two things: a Reactor API key (create one at [reactor.inc/account/api-keys](https://reactor.inc/account/api-keys)) and the
client package.

```bash
cd robotics/sim/notebooks && uv sync --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python xwam_quickstart.py      # then run it
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`xwam_quickstart.py`](./xwam_quickstart.py) reports only its length and
the API endpoint. The optional simulator section is the separate, heavyweight
RoboTwin path.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
client = XwamClient()          # model="xwam", 15 fps tracks
await client.connect()         # handlers -> connect -> await READY -> tracks -> ping
```

X-WAM's tracks are `head_view`, `left_wrist_view` and `right_wrist_view`.
Keep them flowing between requests; the track repeats the current observation
for you. Each prediction is then: set the task once per episode
(`set_task_description`), send one request per distinct `state_json`
(`proprio` + `chunk_id`), and read the `action_prediction` reply, discarding
any reply whose `step` does not echo your `chunk_id`.

X-WAM is the reference implementation of the generic
[robot-policy client contract](./robot-policy-client-contract.md); if the
served model and the contract text ever disagree, trust the served behaviour
and report the mismatch.

## Get actions

The recorded examples below replay a slice of a real RoboTwin 2.0 evaluation:
four consecutive chunks of one rollout, then the first chunk of the next,
which crosses an episode boundary and carries a different instruction. Two
details pair an observation with its reply, and both produce plausible-looking
wrong results if skipped:

**The frame settle.** Nothing tags a chunk with the frame it came from, so
ordering is the only thing pairing an observation with its reply. Push the
frames, let them clear the encoder, then send the request. Send it too early
and the model answers from the tail of the encoder queue, which is the
previous observation, and the reply still looks plausible.

**A retry must change the request.** The model answers once per distinct
`state_json`. Re-sending the exact same state gets no reply at all: the model
cannot tell it apart from the continuous re-delivery of unchanged state, so
it treats it as a duplicate. The session hangs, with no error. Retry keeps the
same `chunk_id` and bumps a `retry` counter; because the noise seed is a pure
function of the seed fields, the retried answer reproduces the lost one when
frames are fed directly. Over the video transport the re-encoded frames make
it equal within tolerance instead.

### What `predict()` does

The client is [`reactor_robotics/xwam.py`](./reactor_robotics/xwam.py).

## Check the results

The check compares the returned actions with the recorded ones and fails if
any element differs by more than `5e-2`. The deltas are not zero because the
recorded actions were produced with tensors fed directly to the model, while
here the same frames travel as H.264 video over WebRTC, so the model decodes
a lossy re-encoding of them. Fed directly, the delta is at most `4.2e-3`;
over the video path, nine runs of these examples measured `1.2e-3` to
`3.7e-2`, so the `5e-2` tolerance has about 1.3× headroom over the worst
observation (see [`examples/PROVENANCE.md`](./examples/PROVENANCE.md)). For
scale, `5e-2` on a delta joint action is well inside the noise a real arm's
controller absorbs, and the 79.3% closed-loop evaluation ran over this same
transport.

Predicted robot states (`proprios`) are not checked. Nothing executes them
(the robot executes `actions`), and their delta is dominated by the same
transport transient without the action head's smoothing.

The last step before `close()` also shows two recovery properties: a retry
with a bumped `retry` field is answered again with the same seeds, and
re-sending the exact same state gets no reply because the model treats it as
a duplicate.

The script prints a row per example and then its verdict, which is the line
to read:

```
RESULT: PASS
```

## Latency

The model is the fast part: inference is ~163 ms per chunk, against 229 ms
for the same work on the authors' stack on the same GPU, 1.4× faster. The
~290 ms measured above is that 163 ms plus two client-side costs: frame
delivery, which this script's own frame rate sets, and transport.

| term | cost | what it is |
|---|---|---|
| model chunk inference | ~163 ms | the deployment's own figure, actual compute |
| frame delivery | up to 67 ms at this script's 15 fps | waiting for a fresh frame on every view |
| WAN hop + TURN relay + H.264 encode/decode | the remainder | transport |

Frame delivery is set by the client's publishing rate, not by the model. The
engine answers a request only once every view has delivered a frame that
arrived after that request, so a cold push waits for the next frame on the
slowest of three views: up to one frame period at whatever rate the client
publishes, and nearer a whole period than half of one because it is the
worst of three independent phases.

This script publishes at 15 fps, so that wait is up to 67 ms. The deployment
accepts 30 fps (`XwamClient(fps=30)`), which halves the period and shaved a
measured ~35 ms off the round trip, back to back. The script stays at 15 fps
because that is the rate the published evaluation ran at. A robot client,
streaming continuously, pays almost none of this either way.

None of it costs task success: the closed-loop RoboTwin 2.0 evaluation ran
over this exact wire, same transport and pacing, and scored 79.3% against
77.0%, computed from the paper's per-task results for the same ten tasks.
For scale, the paper's own robot rig ran at ~300 ms per chunk. A round trip
far above ~300 ms is GPU contention: worth noting, not worth chasing.

## Run the simulator

Everything above replayed recorded examples. Closing the loop means running
the RoboTwin 2.0 authors' own evaluation client against this same hosted
model, which scored 79.3% success on the 10 hardest RoboTwin 2.0 tasks
(reference: 77.0%, computed from the paper's per-task results for the same
ten tasks). The harness is in this repository:
[`../robotwin/`](../robotwin).

```
RoboTwin env (upstream, UNMODIFIED)                xwam model (Reactor)
  the authors' eval client  ──pickle/zmq──▶  robotwin_sim gateway
                                                │  3 video tracks
                                                └──── WebRTC ────▶
```

### Commands

```bash
# 1. the gateway, in its own venv (binds the port their client expects)
cd ../robotwin
uv sync --python 3.12

uv run python check_wiring.py         # offline: numpy and pyzmq only

export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
uv run python -m robotwin_sim.main --port 10086

# 2. the authors' client, unmodified, in the sim env, from their checkout
cd <X-WAM checkout>/evaluation
python robotwin_client.py --task_name <task> --task_config demo_randomized \
    --num_evals_per_worker 50 --server_port 10086 --save_root_dir <out>
```

Wait for the gateway's `tracks published` line before starting the client.
The client scores the episodes itself and writes them under
`--save_root_dir`.

### Expected output

```
robotwin_sim: connecting to xwam at https://api.reactor.inc
robotwin_sim.bridge: connected to xwam at https://api.reactor.inc;
                     tracks published: head_view, left_wrist_view, right_wrist_view
robotwin_sim.gateway: listening on tcp://*:10086; point the authors' client
                      here with --server_port 10086
robotwin_sim.bridge: task: '<the instruction their client sampled>'
robotwin_sim.bridge: chunk 1 (rollout 0 step 0): 290 ms
... one line per request, then the session summary on Ctrl-C ...
```

### Requirements

| | |
|---|---|
| Environments | two, isolated: the sim env (RoboTwin 2.0, SAPIEN, curobo, the authors' client) and the gateway env (this package) |
| numpy | the pins are incompatible, which is why the envs are split: `numpy==1.23.5` in the sim env, `numpy>=1.26` in the gateway env. The two processes exchange only pickled dicts over a local socket. |
| Install time | multi-hour on a CUDA GPU machine: SAPIEN, a renderer, curobo's compiled kernels and the task assets. It is not a `pip install`. |
| Gateway | no simulator, no GPU and no model weights; Python 3.10 or newer, 3.12 in the command above |
| Upstream | [github.com/sharinka0715/X-WAM](https://github.com/sharinka0715/X-WAM) (Apache-2.0), whose `evaluation/` holds the client. Evaluation numbers are pinned to commit `72cfb86b`. |
| Session start | `--ready-timeout` defaults to 300 s, because a cold deployment schedules a GPU and stages weights |

None of the simulator side is needed for anything above. The gateway's
`predict()` is the same `predict()` shown above; only the frame source
changes, from an `.npz` to the simulator's renders.

## Physical deployment

The same client, with frames from your cameras and `proprio` from your arm.
The protocol does not change. To drive an arm:

1. Publish `head_view`, `left_wrist_view` and `right_wrist_view` from your
   three cameras, and keep them flowing between requests.
2. Set the task once per episode with `set_task_description`.
3. Send one request per distinct `state_json` (`proprio` + `chunk_id`), after
   the frame settle, and discard any reply whose `step` does not echo your
   `chunk_id`.
4. Follow [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the wire rules while you build the rig. X-WAM is its reference
   implementation, so the contract text and this model agree.

Reactor support for physical deployment is coming soon, and the contract is
published now so integration work can start against a stable target.
