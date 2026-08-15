# DreamZero quickstart

Get robot actions from the hosted DreamZero model in a few minutes.

DreamZero (NVIDIA GEAR Lab) is a 14B autoregressive world-action model for
closed-loop robot control on the DROID/Franka embodiment: it predicts future
video and actions together. You stream three camera views and the measured
robot state, and every replan cycle it returns a 24-step action chunk whose
rows are 8 wide (7 joint targets for the 7-DoF arm, plus gripper), with ~267 ms
per replan on the hosted deployment.

The script drives the hosted model with 5 recorded examples: observations
recorded against this deployment, stored together with the action chunks it
returned at the time. The script sends the same observations again and
checks the results.

## Setup

From a clone of this repository, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and create a
[Reactor API key](https://reactor.inc/account/api-keys).

```bash
cd robotics/sim/notebooks && uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python dreamzero_quickstart.py   # then run it
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`dreamzero_quickstart.py`](./dreamzero_quickstart.py) reports only its
length and the API endpoint. The optional RoboLab/Isaac section is the
separate, heavyweight benchmark path.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
client = DreamZeroClient()          # model="dreamzero", 15 fps tracks
await client.connect()
```

Capacity: HTTP 429 `no available capacity` on session creation means no capacity is free right
now. Wait and retry, or ask Reactor for additional capacity.

DreamZero's tracks are `exterior_1`, `exterior_2` and `wrist`, and the names
shift by one against RoboLab's: the checkpoint numbers its video keys from 1
while RoboLab numbers cameras from 0.

```
RoboLab observation/exterior_image_0_left  ->  Reactor exterior_1   (real view)
RoboLab observation/exterior_image_1_left  ->  Reactor exterior_2   (black by default)
RoboLab observation/wrist_image_left       ->  Reactor wrist
```

`exterior_1` is the real left exterior view. Getting the mapping backwards
feeds the model a black primary view and does not error. `exterior_2` being
black is expected: RoboLab's default `--cam2-source black` matches the
checkpoint's training-time camera dropout.
[`../dreamzero/`](../dreamzero) implements the same mapping.

## Get actions

DreamZero does not wait for a request. Once a prompt is set and every camera
has a frame, it sends a new chunk whenever it sees fresh frames: free-running
broadcast, in the model's own terms. There is no request, so nothing
correlates a chunk to one.

| | DreamZero's wire |
|---|---|
| Style | free-running broadcast, no request/reply pairing |
| Tracks | `exterior_1`, `exterior_2`, `wrist` |
| Task | `set_prompt {prompt}` |
| State | `set_joint_position` + `set_gripper_position` (there is no `state_json`) |
| Reply | `action_chunk`, carrying `obs_seq`, `chunk_index` and `inference_seconds` |
| Chunk | `(24, 8)`: 7 absolute joint targets + gripper |
| Episode boundary | `reset`, acked with `episode_reset`; `obs_seq` restarts at 0 |

A robot never notices any of this: it simply executes the newest chunk. A
script does notice, because the obvious implementation is wrong:

```python
push_frames(obs); chunk = await next_chunk()      # WRONG
```

The chunk that arrives next was already in flight when you pushed, computed
from the previous observation. It has the right shape, finite values and a
plausible trajectory, so nothing about it looks wrong.

### Which chunk saw your frames

The client ignores chunks computed from old frames. The rule: record the
largest `obs_seq` seen so far, push the new frames, then ignore every
arriving chunk until one carries a strictly larger `obs_seq`. That one saw
the new frames. `obs_seq` is the model's count of camera frames consumed
within the episode (one counter across all three cameras, from 0, reset by
`reset`), stamped with the highest count in that chunk's snapshot.

What this guarantees: the returned chunk consumed at least one post-push
frame on every camera, because the model waits for all three before
inferring. What it cannot guarantee: the model takes the 4 newest frames per
camera as a rolling window that persists across chunks by design, so no
synchronous client can get a chunk whose whole window is its own observation.

DreamZero's wire departs from the generic
[robot-policy client contract](./robot-policy-client-contract.md): different
command names, no `state_json`, no `chunk_id` echo, and free-running
broadcast instead of one reply per request. Its protocol is documented and
stable, but it is model-specific, so a client written against that contract
will not drive it unchanged. Bringing DreamZero onto the generic contract is
future work.

### What `predict()` does

The client is
[`reactor_robotics/dreamzero.py`](./reactor_robotics/dreamzero.py). Two
orderings matter:

- State before frames. The model snapshots the latest joint/gripper value
  together with the frames it consumes, so the state has to be in place
  before the new frames land.
- Drain before pushing. The pre-push `obs_seq` mark must account for every
  chunk already emitted, otherwise a chunk computed from the previous
  observation can still slip through.

## Check the results

DreamZero's pipeline is unseeded: replaying an observation does not reproduce
its actions, so the returned numbers cannot be compared one-for-one. The
check verifies: correct shape, finite values, `obs_seq` and `chunk_index`
always increasing, joint targets inside Franka limits, gripper in range, and
continuity across chunk boundaries. A large boundary step points at one
specific failure: losing the streamed robot state turns the model's absolute
joint targets into relative deltas, which jump by radians rather than
hundredths.

The L2 distance to the recorded chunks is reported for reference only. Its
band is the model's own run-to-run spread at recording time (the mean plus
three standard deviations over 15 samples, see
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md)), so a single value
outside the band is normal. Repeated values outside it suggest the deployment
has changed.

The script prints a row per check and then its verdict, which is the line to
read:

```
RESULT: PASS
```

After the verdict it ends the episode with `reset`, and confirms the new
episode really is new: `chunk_index` and `obs_seq` both restart, which is why
the client's high-water mark has to be reset along with them.

## Latency

The number to watch is the model-reported `inference_seconds`: the replan
cost, excluding transport. The deployment's advertised operating point is
267 ms per replan. That is the planning rate, not the execution rate: the arm
executes at 15 Hz, so a 24-step chunk is 1.6 s of motion, and at a 267 ms
replan only the first few rows of each chunk ever execute. A median above the
operating point usually means contention on the shared deployment: worth
noting, not worth chasing.

## Run the simulator

Everything above replayed recorded examples, so none of it measures task
competence. Task competence comes from NVIDIA's RoboLab benchmark (Isaac Sim);
the [RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)
reports 25.7% (308/1200) over all 120 tasks. The
harness is in this repository: [`../dreamzero/`](../dreamzero).

### Commands

```
# 1. the gateway (this repo; host side, or any machine RoboLab can reach)
cd ../dreamzero
uv sync --python 3.12

uv run python check_wiring.py         # offline: no simulator, no GPU, no key

export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
uv run python -m dreamzero_sim.main --port 5000

# 2. RoboLab, unmodified, in its own container
docker run --rm --entrypoint /isaac-sim/python.sh --net host --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v $HOME/RoboLab:/workspace/robolab -w /workspace/robolab \
  robolab:<tag> policies/dreamzero/run.py \
    --task BananaInBowlTask --headless \
    --num-envs 1 --num-runs 10 --open-loop-horizon 24 \
    --remote-host localhost --remote-port 5000
```

Two settings decide whether the number you get is real:

- `--open-loop-horizon 24`, not 8. Horizon 24 executes exactly one full chunk
  per inference, the shape the checkpoint was trained for. Horizon 8 matches
  the training-time frame stride instead and measurably loses: ~6-7x more
  dropped objects, and no successes at all on the task it was measured on.
- Frames go out one per camera per query, with no repeats, which is what the
  gateway does. The model's temporal context is its last 4 pushes, so a track
  that repeated its frame to keep video flowing would fill that window with
  four copies of one observation and delete the context.

The gateway binds `0.0.0.0`, so it does not have to share a host with the
simulator; pass the gateway machine's address as `--remote-host`.

### Expected output

```
dreamzero_sim: gateway serving openpi on 0.0.0.0:5000 -> dreamzero
dreamzero_sim.policy_server: listening on 0.0.0.0:5000
... RoboLab boots Isaac (minutes), then sends its first observation ...
dreamzero_sim.bridge: connecting to dreamzero at https://api.reactor.inc
dreamzero_sim.bridge: tracks published: exterior_1, exterior_2, wrist
dreamzero_sim.bridge: episode started (prompt='put the banana in the bowl' ...)
dreamzero_sim.bridge: request 1 -> chunk 3: (24, 8) obs_seq=2 (floor -1)
... one chunk per RoboLab request ...
```

RoboLab writes scored episodes under its `output/` directory.

### Requirements

| | |
|---|---|
| GPU | An RTX-class GPU is required for Isaac's renderer. Datacenter A100/H100/B200 have no graphics engines and cannot render at all. |
| Driver | NVIDIA 580.x. 595.x segfaults Isaac 5.x's renderer at startup. Isaac 5.1 declares a minimum of 570.169, so 580.x is inside its window. |
| Container | RoboLab and its Isaac Sim 5.1 image |
| Install time | hours, dominated by Isaac Sim and the RoboLab assets |
| Gateway | no simulator, no GPU and no model weights; Python 3.12 |
| Session start | minutes: the session holds two GPU workers, so `--ready-timeout` defaults to 900 s. A busy cluster answers with HTTP 429 `no available capacity` rather than queueing; retry after a short wait. |

None of the simulator side is needed for anything above.

## Physical deployment

To drive an arm:

1. Publish `exterior_1`, `exterior_2` and `wrist` from your cameras, keeping
   the camera-index mapping above.
2. Send the arm's measured state with `set_joint_position` and
   `set_gripper_position` on every observation.
3. Adopt the newest chunk and discard older queued chunks. The `obs_seq` gate
   is for pairing scripted replays, not continuous robot control.
4. Add hardware safety: validate actions, enforce motion and joint limits,
   reject stale chunks, and add a watchdog/e-stop. The bare quickstart client
   is not safe to command an arm.
5. Read [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the generic contract. DreamZero has a row in its table recording where
   its own wire departs from it.

Reactor support for physical deployment is coming soon.
