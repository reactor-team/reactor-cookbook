# Praxis-1 closed-loop client

A reference client for the hosted Praxis-1 robot policies. Copy it instead of
writing the loop yourself: the policy pairs camera frames with arm state by
capture time, and a client that gets the stamping wrong still runs, but feeds
the model frames that are 1–4 ticks away from the state it is planning from.

| Model | Robot | Cameras | Frame limit |
| --- | --- | --- | --- |
| `reactor/praxis-1-droid` | One Franka arm (DROID) | `exterior`, `wrist` | 512 × 288 |
| `reactor/praxis-1-yam` | Two YAM arms | `top`, `wrist_left`, `wrist_right` | 416 px on the longer side |

Both run at **15 Hz**. You implement a small `Robot` interface (read the
cameras, read the arm state, execute one command row); `PraxisClient` runs the
loop. `main.py` runs it against a synthetic robot so you can check the
protocol before you connect hardware.

## Run the demo

You need [uv](https://docs.astral.sh/uv/) and a Reactor API key from the
[API keys page](https://reactor.inc/account/api-keys). No GPU, robot or
weights are needed.

```sh
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/robotics/praxis-1
uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'
uv run python main.py --robot droid
uv run python main.py --robot yam --task 'Pick up the pen and place it in the bag.'
```

Keep the key in the environment; do not commit it. `REACTOR_API_URL` overrides
the default endpoint, `https://api.reactor.inc`. `--model owner/name` targets
another deployment of the same model.

The synthetic robot streams moving test images and holds a fixed pose. The demo
runs one episode of four chunks (`--chunks`, `--episodes`), then closes the
session. Each chunk prints one line:

```text
episode=0 step=0 rows=21 latency=<ms> join=capture_time first arm xyz=[...]
episode=0 step=1 rows=24 latency=<ms> join=capture_time first arm xyz=[...]
```

- `step` — the chunk's number in the session.
- `rows` — command rows in the chunk: 21 for the first chunk of an episode,
  24 after.
- `latency` — from the request that triggered the chunk (the state and the
  step echo) to the chunk's arrival. It includes the network and the model;
  the first chunk also includes the 25 start frames reaching the model.
- `join` — how the model paired frames and state for that chunk (below).
  Anything other than `capture_time` is logged as a warning.

The output checks the protocol, not task quality: the images are synthetic and
the robot does not move.

## The stamping rule

The cameras are video tracks; the arm state is a command. They reach the model
on different channels, with different delays, so the model cannot pair them by
arrival order. Pairing "the newest 25 frames with the last 25 state rows"
puts the frames 1–4 ticks off the state, depending on how long the client waits
between its last frame and its state. The model therefore pairs them by
**capture time**, and the client must stamp both with the same value:

```python
from reactor_sdk import time_micros

# Once per 15 Hz tick:
t = time_micros()  # read ONCE, the engine's clock
frames = robot.read_cameras()  # capture every camera...
state = robot.read_arm_state()  # ...and the arm state, now
for name, frame in frames.items():
    reactor.track(name).push_frame(frame, capture_time_us=t)
history.append((t, state))  # this row's time_us is t
```

- Use `reactor_sdk.time_micros()`, not `time.time()` or `time.monotonic()`. It
  is the clock the media engine stamps frames on; any other clock puts the
  stamps seconds away from the frames and the model falls back to `newest`.
- Read it once per tick and give the same value to every camera and to the
  state row. Do not read it per camera.
- Each row's frames must actually be pushed. A row whose stamp no camera
  carries has no frames to pair with.

`PraxisClient._observe` is the one place this happens.

## The loop

```text
connect, publish every camera, set_pose_frame (DROID), set_task
repeat 25 ticks: observe (stamp, push frames, keep the state row)     # robot still
set_robot_state_json (the 25 rows)            -> action_prediction step 0
for each chunk:
    for each row: execute(row); wait one tick; observe
    set_robot_state_json (the last 25 rows)
    set_executed_step (the chunk's step)      -> action_prediction step n+1
```

1. **Episode start.** The model needs 25 frames and 25 state rows. The robot
   holds still for 25 ticks; each tick is observed as above, so the rows repeat
   the start pose with the stamps of frames that were actually pushed.
2. **Execute the chunk.** Each row is one tick. After each row the client waits
   one tick and observes: that row is the state the command reached.
3. **Report it.** After the last row, send the last 25 rows. Row 0 is the
   previous call's current row; the 24 after it are the rows just executed.
   Then echo the chunk's `step` with `set_executed_step`. The state must be sent
   before the echo: the echo is what releases the next call.
4. **Wait.** The cameras keep streaming while the client waits for the next
   chunk. Those frames are stamped but have no state rows; the model skips
   them.
5. **Check the join.** Poll `get_status` now and then (`status_every`, every 10
   chunks by default) and warn if `join` is not `capture_time`. The poll runs
   while the robot executes, so it does not delay the loop.

A new episode sends a new task, holds still for 25 ticks, sends the state, then
`reset_context`. The last chunk of the previous episode is not echoed, so the
model keeps waiting until the reset.

### `join` values

| `join` | Meaning | What to do |
| --- | --- | --- |
| `capture_time` | Every state row was paired with the frames stamped with it. | Nothing; this is the expected value. |
| `nearest` | Some row had no frame within 33 ms after waiting 0.5 s. A frame was dropped, the video arrived over 0.5 s late, or the stamps are slightly off. The nearest frames stood in. | Occasional: network. Every call: check that every camera is pushed every tick with the row's stamp. |
| `newest` | No frame within 1 s of `time_us`. The stamps are not on the `time_micros()` clock, or a frame was pushed without `capture_time_us`. The newest frames were used. | Fix the clock. |
| `cameras` | No `time_us` was sent. The model paired the newest moments all cameras share with the last rows. | Send `time_us`. |
| empty | No call yet. | |

## Implement your robot

A sketch; `exterior_cam`, `wrist_cam` and `arm` stand for your own drivers.

```python
from praxis_client import DROID, ArmState, CommandRow, PraxisClient


class MyRobot:
    def read_cameras(self):
        # Newest RGB uint8 (H, W, 3) frame per camera, scaled down to the limit.
        return {"exterior": exterior_cam.latest(), "wrist": wrist_cam.latest()}

    def read_arm_state(self):
        # [x, y, z, qw, qx, qy, qz] in metres, quaternion scalar first;
        # gripper absolute, 0 = open. DROID's single arm is named "arm".
        return {"arm": ArmState(ee_pose=arm.ee_pose(), grip=arm.gripper())}

    def execute(self, row: CommandRow):
        target = row.targets["arm"]  # position (3,), rotation (3, 3), grip
        arm.servo_to(target.position, target.rotation, target.grip, row.control)


async with PraxisClient(MyRobot(), DROID, pose_frame="fr3_hand") as client:
    await client.run_episode("Put the red block in the bowl.", chunks=20)
```

For YAM, use `YAM`, cameras `top`, `wrist_left`, `wrist_right`, and arms
`left` and `right`, both in the rig's world frame. Every method must return well
inside one tick (66 ms); run slow capture or control in your own threads and
return the newest values.

### The chunk

`action_prediction` carries one row per 15 Hz tick:

| Field | DROID | YAM |
| --- | --- | --- |
| `step` | chunk number | chunk number |
| `poses` / `left_poses`, `right_poses` | per row `[x, y, z]` then the rotation matrix, row-major (12 values) | the same, per arm |
| `grips` / `left_grips`, `right_grips` | per row, absolute, 0 = open | per arm |
| `timestamps` | seconds from the first row | the same |
| `control_json` | the impedance gains to run the chunk with, or empty | — |

DROID poses come back in the frame set with `set_pose_frame`: `fr3_hand`
(default), `robolab_droid` or `molmo_spaces_droid`, the same frame you report
`ee_pose` in. `parse_chunk` turns a chunk into `CommandRow`s. Validate targets
against your robot's limits before you command hardware.

### Failures

`prediction_failed` (`step`, `code`, `reason`) means the chunk could not be
computed. The model retries by itself after 0.5 s, and its next chunk starts a
new context. The client logs it and keeps waiting. Codes: `timeout`, `busy`,
`too_large` (frames over the limit), `rejected`, `task_too_long` (over 64
tokens), `invalid_input`. A chunk that does not arrive within `chunk_timeout_s`
(60 s) raises `TimeoutError`; `get_status` then says what the session waits for
(`waiting_for`, `awaiting_step`).

## Commands and messages

| Command | Payload | Reply |
| --- | --- | --- |
| `set_task` | `{"task": str}` (300 characters at most) | — |
| `set_pose_frame` (DROID) | `{"pose_frame": str}` | — |
| `set_robot_state_json` | `{"robot_state_json": str}` | — |
| `set_executed_step` | `{"executed_step": int}` | — |
| `reset_context` | `{}` | `context_reset` (`next_step`) |
| `get_status` | `{}` | `session_status` (`waiting_for`, `join`, `next_step`, `awaiting_step`) |

`robot_state_json` is a JSON string with at least 25 rows, oldest first:

```json
{"ee_pose": [[x, y, z, qw, qx, qy, qz], ...], "grip": [g, ...], "time_us": [t, ...]}
{"left": {"ee_pose": [...], "grip": [...]}, "right": {...}, "time_us": [t, ...]}
```

The first is DROID, the second YAM; YAM's `time_us` is shared by both arms.
`time_us` must be strictly ascending, one value per row.

## Offline checks

```sh
cd robotics/praxis-1
uv run python -m unittest discover -s tests -v
```

The tests replace the SDK with a fake session that pairs frames and state by
stamp the way the model does. They check that every sent row has frames with
its stamp on every camera, that row 0 is the previous call's current row, that
the state precedes the echo, the YAM state layout, the episode reset and the
`join` warning. They do not check task quality, latency or a hosted session.
