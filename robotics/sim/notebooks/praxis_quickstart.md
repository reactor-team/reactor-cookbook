# Praxis-1 quickstart

Get robot actions from the hosted Praxis-1 model in a few minutes, on either
embodiment:

| Embodiment | Endpoint | Cameras | Frame size |
| --- | --- | --- | --- |
| DROID (one Franka arm) | `reactor/praxis-1-droid` | `exterior`, `wrist` | up to 512 × 288 |
| YAM (bimanual) | `reactor/praxis-1-yam` | `top`, `wrist_left`, `wrist_right` | 16:10, up to 416 on the long side |

Praxis-1 (Runway) is a vision-language-action policy. You stream camera views
and the measured end-effector pose of each arm, and it returns a chunk of
absolute end-effector pose and gripper targets: 21 rows for the first chunk of
an episode, then 24 rows (1.6 s at 15 Hz) for every chunk after it.

The script drives the hosted model with 5 recorded examples: observations
recorded against this deployment, stored with the chunks it returned at the
time. It sends the same observations again and checks the results. The
observations are synthetic, so this checks the connection and the reply
contract, not how well the policy does tasks.

## Setup

From a clone of this repository, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and create a
[Reactor API key](https://reactor.inc/account/api-keys).

```bash
cd robotics/sim/notebooks && uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python praxis_quickstart.py --embodiment droid
uv run python praxis_quickstart.py --embodiment yam
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`praxis_quickstart.py`](./praxis_quickstart.py) reports only its
length and the API endpoint.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
session = ReactorSession("reactor/praxis-1-droid", fps=15, frame_size=(288, 512))
await session.connect(
    ("exterior", "wrist"),
    subscribe=("action_prediction", "prediction_failed", "session_status", "context_reset"),
)
await session.send("set_task", {"task": "Put the red block in the bowl."})
await session.send("set_pose_frame", {"pose_frame": "fr3_hand"})   # DROID only
```

Capacity: HTTP 429 `no available capacity` on session creation means no
worker is free right now. The script retries a few times; otherwise wait and
retry, or ask Reactor for additional capacity.

DROID poses can be in the Franka FR3 hand frame (`fr3_hand`, the default),
NVIDIA RoboLab's DROID frame (`robolab_droid`), or MolmoSpaces' DROID frame
(`molmo_spaces_droid`). The model converts them and returns targets in the
frame you chose. YAM has no frame command: both arms report in the rig's
world frame, and targets come back in it.

## Get actions

Praxis-1 is lock-step. One chunk is outstanding at a time, and only your echo
releases the next one:

```
first chunk of a session   frames (25 per camera) + state (25 rows) -> step 0, 21 rows
every later chunk          state, then set_executed_step(step)        -> step + 1, 24 rows
start a new episode        state, then reset_context                  -> next step, 21 rows
```

Each prediction uses the last 25 frames of every camera and the last 25 rows
of state, so `request_chunk()` in the script publishes the new frames, waits
for the windows to fill with them (about 1.9 s), sends the state, and only
then sends the trigger. Sending the trigger before the frames have filled the
window gives you a chunk computed partly from older frames.

Robot state is a JSON **string** in `robot_state_json`. Positions are in
meters and quaternions are scalar first:

```python
# DROID: one arm
{"ee_pose": [[x, y, z, qw, qx, qy, qz], ...25 rows], "grip": [g, ...25 values]}
# YAM: both arms, rig world frame
{"left": {"ee_pose": [...], "grip": [...]}, "right": {"ee_pose": [...], "grip": [...]}}
```

The reply carries `poses` and `grips` (DROID), or `left_poses`, `left_grips`,
`right_poses` and `right_grips` (YAM). Each pose row is `[x, y, z]` followed
by the 3 × 3 rotation matrix row by row. `step` arrives as a JSON number that
the SDK may decode as a float, so compare it with `int(step)`. DROID replies
also carry `control_json`, the impedance gains of the DROID demonstrations.

## Check the results

The policy samples, so replaying an observation does not reproduce its chunk.
The check verifies: 21 rows and then 24, 12 values per pose row, finite
values, `step` counting 0, 1, 2, …, rotation blocks that are rotation
matrices, row 0 close to the pose you sent, and adjacent rows close to each
other.

Targets are absolute, so row 0 sitting near the measured pose is what catches
a wrong frame, swapped units, or a camera on the wrong track. The row 0 and
adjacent-row tolerances are calibrated at recording time: the mean plus three
standard deviations over 3 passes of the same observations, see
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md). Without the recorded
fixture those two numbers are reported without a pass/fail. The rotation check
uses a fixed tolerance of 0.001, because the blocks are orthonormal to float
precision and precision noise is not worth calibrating.

The L2 distance to the recorded chunks is reported only. Its band is the
model's own run-to-run spread, so a single value outside it is normal.
Repeated values outside it suggest the deployment has changed.

The script prints a row per check and then its verdict, which is the line to
read:

```
RESULT: PASS
```

After the verdict it shows the flow control: `get_status` reports that the
session is waiting for the robot to execute the last step, and
`reset_context` starts a new episode whose first chunk has 21 rows while
`step` keeps counting. If every prediction times out, send `get_status`: its
`waiting_for` field says what the session still needs.

## Latency

One chunk is 1.6 s of motion (24 rows at 15 Hz). The model computes the next
chunk only after your echo, so the echo-to-chunk time is a wait your
controller must cover by holding the robot. At recording time it was about
250 ms p50 on DROID and 270 ms on YAM, measured from a laptop; see the
manifests in [`examples/`](./examples) for the full numbers. These are
reference points, not guarantees.

## Run the simulator

This repository has no simulator adapter for Praxis-1 yet. The
[DROID integration guide](https://docs.reactor.inc/robotics/praxis/droid/integration)
and [YAM integration guide](https://docs.reactor.inc/robotics/praxis/yam/integration)
describe what an adapter must do. The DROID pose frames `robolab_droid` and
`molmo_spaces_droid` let an adapter for those simulators send poses without
converting them.

## Physical deployment

To drive a robot:

1. Publish every camera track from cameras placed like the training rig, and
   check which image goes to which track.
2. Send the measured state, not the last requested targets, with at least 25
   rows at 15 Hz, and send it before each `set_executed_step`.
3. Validate each chunk before executing it: finite values, the `step` you
   expect, and targets inside your robot's position, speed, acceleration, and
   workspace limits. Convert the pose targets to joint motion with your own
   controller, and convert gripper values to commands your gripper accepts.
4. On `prediction_failed` or a lost connection, hold or stop the robot. Keep
   the emergency stop active. Resetting the model does not reset the robot.
5. Read [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the generic contract.
