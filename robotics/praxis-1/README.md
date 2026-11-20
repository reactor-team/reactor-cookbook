# Praxis-1 closed-loop client

> **Warning:** Praxis-1 has been validated in simulation only. On a real
> robot, keep an emergency stop within reach and your controller's limits
> (workspace, speed, force) active at all times.

A reference client for the hosted Praxis-1 robot policies. Copy it instead of
writing the loop yourself: the policy pairs camera frames with arm state by
capture time, and a client that gets the stamping wrong still runs, but feeds
the model frames that are 1–4 ticks away from the state it is planning from.

| Model | Robot | Cameras | Frame limit |
| --- | --- | --- | --- |
| `reactor/praxis-1-droid` | One Franka arm (DROID) | `exterior`, `wrist` | 512 × 288 |
| `reactor/praxis-1-yam` | Two YAM arms | `top`, `wrist_left`, `wrist_right` | 16:10, 416 px on the longer side |

Both run at **15 Hz**. You implement a small `Robot` interface (read the
cameras, read the arm state, execute one command row, hold still);
`PraxisClient` runs the loop. `SteppedPraxisClient` runs the same loop in a
stepped session (see [Streamed or stepped](#streamed-or-stepped)). `main.py`
runs either against a synthetic robot so you can check the protocol before you
connect hardware.

The model references list every command, message and field:
[Praxis-1 DROID](https://docs.reactor.inc/robotics/praxis/droid/reference) and
[Praxis-1 YAM](https://docs.reactor.inc/robotics/praxis/yam/reference).

### Terms

- **Tick** — one 15 Hz control period (66.7 ms). The client observes the robot
  once per tick.
- **Row** — one tick's data: a state row is the arm state observed at a tick; a
  command row is one tick's targets in a chunk.
- **Chunk** (or **step**) — the command rows one policy call returns, numbered
  by `step`: 21 rows for an episode's first chunk, 24 after.
- **Echo** — `set_executed_step`: telling the model which chunk the robot has
  executed. The model computes the next chunk only after the echo.
- **Episode** — one task from a still start: 25 still ticks, then chunks. A
  new episode starts a new context with `reset_context`.
- **Join** — how a call paired camera frames with state rows
  (`session_status.join`).

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
uv run python main.py --robot yam --task 'Pick up the pen from the center of the table, then place the pen in the bag'
```

Keep the key in the environment; do not commit it. `REACTOR_API_URL` overrides
the default endpoint, `https://api.reactor.inc`. `--model owner/name` targets
another deployment of the same model.

Write the task the way each model was trained. YAM was trained on captions
like the one above, with no closing period, and the task reaches the model
verbatim. Each robot's default `--task` already has that form.

The synthetic robot streams moving test images and holds a fixed pose. The demo
runs one episode of four chunks (`--chunks`, `--episodes`), then closes the
session. It prints one line per chunk, then a summary:

```text
Synthetic robot on reactor/praxis-1-droid, streamed (not task quality)
episode=0 step=0 rows=21 latency=<n> ms join=capture_time first arm xyz=[<x>, <y>, <z>]
episode=0 step=1 rows=24 latency=<n> ms join=capture_time first arm xyz=[<x>, <y>, <z>]
episode=0 step=2 rows=24 latency=<n> ms join=capture_time first arm xyz=[<x>, <y>, <z>]
episode=0 step=3 rows=24 latency=<n> ms join=capture_time first arm xyz=[<x>, <y>, <z>]
Executed 93 rows; session closed
join=capture_time: every state row was paired with the frames stamped with it
```

- `step` — the chunk's number in the session.
- `rows` — command rows in the chunk: 21 for the first chunk of an episode,
  24 after.
- `latency` — how long the robot waited for the chunk: from the last row the
  client observed before the call (the end of the previous chunk, or the 25th
  start tick) to the chunk's arrival. It includes sending the state, the
  network and the model. The client waits for the state command's reply
  before it sends the echo, so that round trip is in it too. The first chunk
  also includes the 25 start frames reaching the model.
- `join` — how the model paired frames and state for that chunk (below).
  Anything other than `capture_time` is logged as a warning.
- `xyz` — the first row's target position from the model.

The output checks the protocol, not task quality: the images are synthetic and
the robot does not move.

## Streamed or stepped

The client talks to the model in one of two ways. Both use the same `Robot`,
the same 15 Hz execution and the same chunks.

| | Streamed (`PraxisClient`) | Stepped (`SteppedPraxisClient`) |
| --- | --- | --- |
| Cameras | Video tracks, streamed all the time | The chunk's frames, sent with the call |
| State and echo | Two commands after each chunk | In the same call as the frames |
| Stamps | Every frame and state row, from `time_micros()` | None |
| Expected `join` | `capture_time` | `newest` |
| Call round trip, median, 12 robots under load | 211 ms | 304 ms (about 90 ms more) |
| From the end of a chunk to the next, one robot | about 170–280 ms | about 370–730 ms, with encoding |
| Video per call | none (streamed) | about 60–130 KB with real camera frames |

**Choose streamed** for the lowest latency. The video is already on the model
when the chunk ends, so a call carries only the state. In exchange, the client
must stamp every frame and state row with the same capture time (see
[The stamping rule](#the-stamping-rule)).

**Choose stepped** for the simplest correct client. After each chunk, one
`Reactor.step` carries the chunk's frames of every camera, the last 25 state
rows and the step echo. Frames and state arrive together, so nothing is
stamped and nothing can drift apart. The cost is latency: the chunk's video is
encoded and uploaded with the call.

```sh
uv run python main.py --robot droid                  # streamed (default)
uv run python main.py --robot droid --mode stepped
uv run python main.py --robot yam --mode stepped
```

In code, swap the class; the rest is the same:

```python
async with SteppedPraxisClient(MyRobot(), DROID, pose_frame="fr3_hand") as client:
    await client.run_episode("Put the red block in the bowl.", chunks=20)
```

### How a stepped call works

```text
connect (generation="stepped"), publish every camera, set_pose_frame (DROID), set_task
repeat 25 ticks: observe (keep each camera's frame and the state row)   # robot still
step(state=25 rows, media=25 frames per camera)          -> action_prediction step 0
for each chunk:
    for each row: execute(row); wait one tick; observe
    step(state=25 rows + executed_step, media=the chunk's frames) -> step n+1
```

- `state` holds `robot_state_json` with the last 25 rows and no `time_us`.
  After the first call it also holds `executed_step`, the step just executed.
- `media` holds, per camera, one frame per row since the previous call: 25 at
  episode start, then 21 after the first chunk and 24 after the others. The
  model pairs the last 25 state rows with the 25 newest frames.
- Each camera's frames go as one H.264 stream that starts at a keyframe, one
  `EncodedFrame` per frame. H.264 needs an even width and height. Raw frames
  would make a call about 22 MB; encoded, it is about 60–130 KB with real
  camera frames (the demo's synthetic frames encode to about 11–20 KB).
  `encode_h264` does this with [PyAV](https://github.com/PyAV-Org/PyAV).
- The chunk arrives as an `action_prediction` message, as in a streamed
  session. A new episode sends `reset_context` before its first call.

Do not send `time_us` in a stepped session. Stamps pair a state row with frames
that carry the same `capture_time_us`; the stepped frames carry none, so the
model would wait 0.5 s per call for frames that never come.

A stepped session computes a chunk only inside `step()`, and nothing retries a
call by itself. A call the runtime refuses raises `StepRefusedError` (nothing
it carried was applied); a call that runs and fails raises `PredictionFailed`.
Start a new episode to go on. Any message still queued from an earlier call is
dropped before each call.

The demo also prints the encoded video each call carried:

```text
Synthetic robot on reactor/praxis-1-droid, stepped (not task quality)
episode=0 step=0 rows=21 latency=<n> ms bytes=<n> join=newest first arm xyz=[<x>, <y>, <z>]
episode=0 step=1 rows=24 latency=<n> ms bytes=<n> join=newest first arm xyz=[<x>, <y>, <z>]
...
Executed 93 rows; session closed
join=newest: expected in a stepped session: each call carries its own frames and state, and the last 25 rows were paired with the call's 25 newest frames
```

`latency` is measured as in a streamed session: from the last observed row to
the chunk's arrival. Here that includes encoding and uploading the video.

### What `join` means in each mode

- **Streamed:** `join` says how the model matched the streamed frames to the
  state rows. Expect `capture_time`; anything else is logged as a warning (see
  [`join` values](#join-values)).
- **Stepped:** expect `newest`. Each call carries its own frames, so the model
  pairs the last 25 rows with that call's newest frames. This is correct, and
  the client does not warn about it.

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

### Camera latency

The stamp is the moment the client reads the camera, not the moment the frame
was exposed. If your camera pipeline delivers frames late (often 30–120 ms),
the frame read at `t` shows the scene at `t` minus that delay, and the model
pairs it with the arm state at `t`. `join` still reports `capture_time`: the
stamps match, so the model cannot see the delay.

If your frames arrive late, stamp each frame with its exposure time converted
to the `time_micros()` clock, and give each state row a stamp within 33 ms of
its frames (for example, the state your controller recorded nearest that
exposure time). Convert with an offset measured once:
`offset = time_micros() - driver_clock_now()`, then
`capture_time_us = exposure_time + offset`, where `driver_clock_now()` reads the
clock the driver stamps exposures on, in microseconds. This client stamps read
time; change `_observe` for such a rig.

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

1. **Episode start.** The model needs 25 frames and 25 state rows. Bring the
   robot to its start pose and let it come to rest before you call
   `run_episode`: the 25 start rows record whatever the arm does. The client
   observes 25 ticks, so the rows repeat the start pose with the stamps of
   frames that were actually pushed.
2. **Execute the chunk.** Each row is one tick. After each row the client waits
   one tick and observes: that row is the state the command reached.
3. **Report it.** After the last row, send the last 25 rows. After a 24-row
   chunk, row 0 is the previous call's current row and the 24 after it are the
   rows just executed. After the episode's first chunk (21 rows), the first 4
   rows are the last start ticks, so the previous call's current row is row 3.
   Then echo the chunk's `step` with `set_executed_step`. The state must be
   sent before the echo: the echo is what releases the next call.
4. **Wait.** The cameras keep streaming for the first 17 ticks (1.1 s) of
   the wait for the next chunk, then pause until it arrives. Those frames are
   stamped but have no state rows; the model skips them. The model keeps 50
   frames per camera, and every frame streamed during a wait lies between two
   rows of the next call: 25 rows + 17 frames + the 8 frames that can arrive
   while the model waits up to 0.5 s for late video fit in 50. If more frames
   were streamed, the frame of the call's row 0 would leave the window and the
   call would fall back to `nearest`, delaying it by 0.5 s and making the next
   wait longer still.
5. **Check the join.** Poll `get_status` now and then (`status_every`, every 10
   chunks by default) and warn if `join` is not `capture_time`. The poll runs
   while the robot executes; the client lets it finish before it sends the next
   state (it gives up on it after 0.2 s), so the poll reports this chunk's call.

A new episode sends a new task, holds still for 25 ticks, sends the state, then
`reset_context`. The last chunk of the previous episode is not echoed, so the
model keeps waiting until the reset. Any message still queued from the
previous episode is dropped before the reset.

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
        # An absolute target, safe to hold indefinitely; never a velocity.
        target = row.targets["arm"]  # position (3,), rotation (3, 3), grip
        arm.set_target(target.position, target.rotation, target.grip, row.control)

    def hold(self):
        # Hold still: the measured pose becomes the target.
        arm.set_target_to_current_pose()


async with PraxisClient(MyRobot(), DROID, pose_frame="fr3_hand") as client:
    await client.run_episode("Put the red block in the bowl.", chunks=20)
```

For YAM, use `YAM`, cameras `top`, `wrist_left`, `wrist_right`, and arms
`left` and `right`, both in the rig's world frame. Every method must return well
inside one tick (66 ms); run slow capture or control in your own threads and
return the newest values.

### Safety

- `execute()` sets an **absolute target** that is safe to hold indefinitely.
  The arm holds the last row's target while the client waits for the next
  chunk (about 0.1–0.7 s, longer if the model is slow) and after any error.
  Never send velocities from `execute()`: a velocity keeps the arm moving
  while nothing is commanded.
- `hold()` holds the arm still: set its current measured pose as the target,
  or stop it safely. The client calls it when an episode ends, on any error
  during one (including `PredictionFailed`, `TimeoutError` and an exception
  from your own methods), and when the client exits. It may be called more
  than once.
- Validate every target against your robot's limits before you command
  hardware.

### Camera frames

- Return **RGB**, `uint8`, shape `(H, W, 3)`. OpenCV and RealSense often
  deliver BGR: convert it (`frame[..., ::-1]`). The SDK reads a 4-channel
  array as BGRA, so the client refuses anything but 3 channels.
- Keep each camera's native aspect: **crop, do not stretch**, then scale down.
  DROID frames must fit 512 × 288. YAM views must be 16:10 with the longer
  side 416 px (for example 1920 × 1200 scaled to 416 × 260); the model trims
  each view to its 416 × 256 tile and refuses other aspects.
- A stepped session needs an even width and height (H.264).

The client checks the frames once before it connects, then every tick, so a
frame the model would refuse stops the client before the arm moves.

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
`ee_pose` in. `parse_chunk` turns a chunk into `CommandRow`s.

### Failures

`prediction_failed` (`step`, `code`, `reason`) means the chunk could not be
computed. In a streamed session the model retries by itself after 0.5 s, and
its next chunk starts a new context.

| Code | Meaning | Streamed client |
| --- | --- | --- |
| `busy`, `timeout` | The model was busy or slow | Logs it and keeps waiting |
| `too_large` | Frames over the limit | Raises `PredictionFailed` |
| `task_too_long` | The task is over 64 tokens | Raises `PredictionFailed` |
| `invalid_input` | The state or the frames could not be used | Raises `PredictionFailed` |
| `rejected` | The model refused the call | Raises `PredictionFailed` |

A retry cannot fix the last four, so the client stops the episode (the robot
holds) instead of waiting. Fix the cause and start a new episode. A chunk that
does not arrive within `chunk_timeout_s` (60 s) raises `TimeoutError`; its
message says what the session is waiting for (`get_status`'s `waiting_for`).

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
`time_us` must be strictly ascending, one value per row. Every value must be
finite, and each quaternion non-zero.

## Offline checks

```sh
cd robotics/praxis-1
uv run python -m unittest discover -s tests -v
```

The tests replace the SDK with a fake session that keeps the model's 50 frames
per camera and pairs frames and state by stamp the way the model does. They
check that every sent row has frames with its stamp on every camera, which row
is the previous call's current row, that each row is observed after its
command, that the state precedes the echo and the status poll precedes the
next state, that a slow model does not push row 0's frame out of the window,
the YAM state layout, the episode reset (state, then `reset_context`, and the
first chunk computed from the still rows), the failure codes, timeouts, stale
messages, `hold()`, the frame and state checks and the `join` warning. For
stepped sessions, a second fake decodes each call's video. Those tests check
that the state has no `time_us`, the step echo, the frame count per camera (25,
then one per executed row), that each camera's stream starts at a keyframe and
decodes to the frame observed at the same tick, failures, and that `newest` is
not warned. They do not check task quality, latency or a hosted session.
