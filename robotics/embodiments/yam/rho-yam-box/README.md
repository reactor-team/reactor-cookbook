# Rho (YAM box) quickstart

This client drives `reactor/rho-yam-box`, Microsoft's Rho vision-language-action
policy for the bimanual [YAM](../README.md) robot. It streams three RGB camera
views, sends the end-effector state of both arms and a task instruction, and
receives **50 × 20 absolute end-effector targets**. Execute the first 25 rows
at 15 Hz, then ask again. The example drives a mock robot; it moves no
hardware.

Start with the CLI below, then read [Connect your robot](#connect-your-robot)
or the [Python API example](#call-the-api-from-your-own-python-code).

## Read before you use it on a robot

- The weights are [`microsoft/rho-yam-box`](https://huggingface.co/microsoft/rho-yam-box)
  (MIT), served with the inference code from
  [`microsoft/rhobotics`](https://github.com/microsoft/rhobotics) (MIT) as
  published: bfloat16, 10 flow steps, no quantization or other lossy change.
- It is a **midtrained** checkpoint, which Microsoft publishes as a starting
  point for finetuning. Its bundled data config describes simulation data. How
  well it drives a real YAM without finetuning is **unknown**. There is no
  model card.
- Upstream does not document the units, the frames, or the arm order. The
  checkpoint statistics suggest that positions are meters, probably in each
  arm's own base frame, and that the gripper value is the raw gripper joint
  value (about -1.22 to 1.47 in the training data), not a 0 to 1 fraction.
  Left arm first follows upstream's docstrings and camera names. Confirm all
  three on your robot before you close the loop.

## Run the quickstart

You need [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/),
and a Reactor API key from the
[API keys page](https://reactor.inc/account/api-keys). Your account needs
access to `reactor/rho-yam-box`; if a valid key cannot open a session,
contact Reactor to request access. No local GPU, simulator, or weights are
needed. The commands below let uv install Python 3.12; the client supports
Python 3.10+.

```sh
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/robotics/embodiments/yam/rho-yam-box/client-python
uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'
uv run python main.py --requests 5
```

Keep your key in the environment; do not paste it into a Python file or
commit it. The client prints no credentials.

You should see output like this (times vary):

```text
Synthetic protocol smoke test (not task quality); no robot moves
step=0 shape=(50, 20) execute=25 model=73.7 ms RTT=257.3 ms view skew=0 us first-row jump L=165 mm R=77 mm
step=1 shape=(50, 20) execute=25 model=79.0 ms RTT=163.0 ms view skew=0 us first-row jump L=42 mm R=98 mm
...
Reset check: step=5 shape=(50, 20) execute=25 model=73.2 ms RTT=161.3 ms view skew=0 us first-row jump L=109 mm R=61 mm
PASS: 6 valid predictions; session closed
Median model=73.7 ms; median request RTT=165.4 ms
```

The script makes five requests, resets the episode, makes one more request,
and closes the session, also on failure or Ctrl+C. Between requests the mock
robot steps through the 25 executed rows at 15 Hz (about 1.7 s), so the loop
runs at the real-time cadence. Add `--no-realtime` to skip those waits.

All inputs are synthetic: seeded camera-like 480 × 640 images (a gradient
and colored blocks that move a few pixels per frame) and a fixed start
pose. They test the API, not task success. `first-row jump` is the distance
from the current position of each arm to the first target. With random
images it is often several centimeters. A real controller must refuse a chunk
whose first row is far from the measured pose.

The endpoint defaults to `https://api.reactor.inc`. Set `REACTOR_API_URL` to
use another environment. The lockfile pins the tested Python SDK version.

## Call the API from your own Python code

Save the following as `example.py` next to `client.py` and run
`uv run python example.py`. `RhoClient`, `SyntheticCameras`, and `MockYam` are
helpers in this folder, not SDK imports.

```python
import asyncio

from client import RhoClient, split_arms
from main import MockYam, SyntheticCameras


async def main():
    cameras, robot = SyntheticCameras(), MockYam(realtime=False)
    async with RhoClient() as client:
        prediction = await client.predict(
            cameras.read(),         # {"scene_view": ..., "left_wrist_view": ..., "right_wrist_view": ...}
            robot.get_state(),      # 20 values, left arm first
            "pick up the red cube and place it in the box",
            seed=0,
        )
        print(prediction.actions.shape)                    # (50, 20)
        print(split_arms(prediction.actions[0])["left"])   # xyz, 3x3 rotation, gripper
        await robot.execute(prediction.to_execute)         # the first 25 rows
        await client.reset()                               # before the next episode


asyncio.run(main())
```

Keep the context open for a whole session: capture, `predict`, execute, and
repeat. The instruction is sent again only when `task` changes. If a request
fails or times out, leave the context and find the cause before you open a
new session.

## Connect your robot

Replace the two stand-ins in [`main.py`](client-python/main.py):

- `SyntheticCameras.read` returns one `uint8` RGB frame for each of
  `scene_view`, `left_wrist_view`, and `right_wrist_view`. Any frame size
  works; 480 × 640 is typical. Keep the roles fixed: a wrist frame
  on `scene_view` gives wrong actions, not an error.
- `MockYam.get_state` returns the measured 20-value end-effector state.
  `MockYam.execute` sends target rows to the arms at 15 Hz. A real driver
  needs forward and inverse kinematics, limits, and a startup guard; see
  [Drive a real YAM](../README.md#drive-a-real-yam).

`arm_vector`, `pack_state`, and `split_arms` in
[`client.py`](client-python/client.py) convert between a position, a 3 × 3
rotation matrix, and a gripper value and the 20-value layout.

The client sends camera frames from the asyncio event loop. Run your
controller in its own thread or process, or make it `async`, so that it does
not block the loop.

## Wire contract and timing

The model follows the request/reply form of the
[robot policy client contract](../../../sim/notebooks/robot-policy-client-contract.md),
with the differences below.

1. Register handlers, connect, wait for `READY`, then publish the three video
   tracks. Send frames at a steady 10–30 fps (the client uses 15) and repeat
   the current observation between requests.
2. Send `set_task_description` with
   `{"task_description": "pick up the red cube and place it in the box"}`
   (1–300 characters). The model answers no request until it is set. A change
   applies to the next request.
3. Push the three views of the new observation with one shared capture
   time, then send `set_state_json` at once, with the same value as
   `capture_us` and a JSON **string** as its `state_json` field:

   ```python
   from reactor_sdk import time_micros

   now = time_micros()  # the SDK clock, not time.time()
   for view, frame in observation.items():
       tracks[view].push_frame(frame, capture_time_us=now)
   ```

   ```json
   {"proprio": [0.4, 0.0, 0.2, 1, 0, 0, 0, 1, 0, 0.3,
                0.4, 0.0, 0.2, 1, 0, 0, 0, 1, 0, 0.3], "chunk_id": 3,
    "capture_us": 1759912345678901}
   ```

   `proprio` is the 20-value state, left arm first. `chunk_id` is a
   nonnegative request ID. `capture_us` is the capture time of the three
   frames, in microseconds. The optional `seed` (0 to `2**63 - 1`) fixes the
   sampling noise and defaults to `chunk_id`. There is no `cfg` field and no
   seed triple.
4. Wait for `action_prediction` whose `data.step` equals `chunk_id`:

   ```text
   {"type": "action_prediction",
    "data": {"actions": [[...20 values...], ...50 rows...],
             "execution_horizon": 25, "step": 3, "inference_seconds": 0.075,
             "source_capture_us": 1759912345678901, "view_skew_us": 0}}
   ```

   Row k is the absolute target for control step k + 1 after the
   observation, in the layout of `proprio`. Execute `execution_horizon` (25)
   rows, then send the next request. The other 25 rows are a preview. The
   reply has no predicted states (`proprios`). `source_capture_us` echoes
   the request's `capture_us`. `view_skew_us` is how far apart the three
   frames the model used were captured (0 when one capture time stamped all
   three). Integer fields can arrive as floats (`"step": 3.0`,
   `"execution_horizon": 25.0`), so compare them by value.

Rules:

- With `capture_us`, the model answers as soon as each of the three tracks
  has delivered a frame captured at or after `capture_us`, from the frame of
  each track nearest to it. Frames that arrived before the request count, so
  there is no need to wait between the push and the request. If a track has
  no frame within 75 ms of `capture_us` (for example, a lost frame), the
  model drops the request with `command_error`; send a new observation.
- Without `capture_us`, the model answers only after each of the three
  tracks has delivered a frame that arrived after the request. Push the new
  frames, wait a few frame periods, then send the request.
- Keep one request outstanding. Do not pipeline.
- A byte-identical `state_json` gets no second reply. To retry after a lost
  reply, keep `chunk_id` and change another byte, for example a `"retry": 1`
  field; the model ignores unknown keys. A retry that keeps `capture_us` runs
  on the same frames while the model still holds them (about 1 s).
- A malformed request (wrong length, a non-finite number, a missing
  `chunk_id`) is dropped with **no reply**. The model never fills in a
  missing state. This client checks the state before it sends it.
- `command_error` with `command: "state_json"` means the model could not
  answer that request. The session stays open; send it again with a byte
  changed. `command: "reset"` means a reset failed; reconnect.
- `reset` starts a new episode. It drops an unanswered request and keeps the
  task and the tracks. Use a new `chunk_id` after it.
- The model scales each frame to fit 224 × 224, keeping the aspect ratio, and
  pads the rest. It clips each state value to slightly past the range of its
  training data, so a state far outside that range is answered as if it were
  at the edge.

Timing: the model takes about 75 ms per request on the serving GPU
(`inference_seconds`). The round trip from `set_state_json` to the reply
also includes network time and the time for the observation's frames to
arrive. The client pushes the frames and sends the request at the same
moment, so the round trip is also the time from capture to actions. From a
laptop on home internet we measured 165 ms median with the camera-like
frames of `main.py` and 257 ms with real 480 × 640 robot camera images (a
real image is larger after encoding). The first request with an
instruction of a new length can take about 0.3 s more, once. The arm holds
its last target while it waits for the next chunk. These are small-sample
observations, not a latency guarantee.

With the same decoded frames, state, instruction, and seed, the model gives
the same actions. Video encoding can change decoded pixels, so repeated
requests with the same source frames can still differ slightly.

## What this checks

- Matching request IDs, a finite `(50, 20)` chunk, and `execution_horizon`.
- A request after `reset`.
- Input checks before a request is sent: three named `uint8` RGB views, 20
  finite values, and 6D rotation columns that are not zero or parallel.
- Session cleanup on success and failure.

It does not check task success, units, or motion on a real robot.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `REACTOR_API_KEY` is missing | Export a key in the same terminal that runs `uv run`. |
| Authentication or access failure | Confirm the key and `REACTOR_API_URL`. If a valid key still cannot connect, contact Reactor to request access to `reactor/rho-yam-box`. Do not share the key in logs. |
| HTTP 429 `no available capacity` | Wait for capacity, then retry. An open session reserves a worker. |
| No SDK wheel or import errors | Run `uv sync --locked --python 3.12` in this directory. Linux wheels need glibc 2.34 or newer. |
| `TimeoutError` with no `command_error` | The model dropped the request or never got the observation's frames. Check the three tracks, the task, and the state. `capture_us` must be a `time_micros()` value, the clock that stamps the frames. |
| `command_error` naming `capture_us` | A track had no frame within 75 ms of `capture_us` (a lost frame, or an old `capture_us`). Send a new observation. |

## Verification

On October 8, 2026, the quickstart command above ran against
`https://api.reactor.inc` from a Python 3.12 environment installed with
`uv sync --locked` (Python SDK 1.9.0): five requests and one after reset
returned valid `(50, 20)` chunks with matching request IDs, matching
`source_capture_us`, `view_skew_us` 0 and `execution_horizon` 25, and the
session closed. Median model time was 74 ms and median request round trip
165 ms. This checks the API wiring, not robot task quality.

## Offline checks

```sh
cd robotics/embodiments/yam/rho-yam-box/client-python
uv run python -m unittest discover -s tests -v
```

The tests use a fake SDK. They cover the request and reply, malformed
replies, invalid inputs, timeouts, cleanup, the state layout helpers, the
main loop, the Python example above, and the links in these pages.
