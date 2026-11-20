# Rho on YAM

Call **`reactor/rho-yam-box`** with three camera images, the robot state, and
a task instruction. The model returns 50 action targets for both arms. Your
application decides how to execute them and when to send the next observation.

## Setup

Install [uv](https://docs.astral.sh/uv/) and `make`, then run:

```sh
cd robotics/embodiments/yam
make setup
make test
export REACTOR_API_KEY='YOUR_OWN_API_KEY'
make smoke
```

Requires Python 3.12+; uv can provision it. Create a key at
[Reactor API keys](https://reactor.inc/account/api-keys) for an account with
access to `reactor/rho-yam-box`. The client reads it from `REACTOR_API_KEY`.

`make smoke` sends synthetic images and an example state, saves the prediction
under `output/`, and disconnects. It checks the connection and response format.
`make test` runs offline with mocked services.

## Predict from a saved observation

Capture the three views and robot state together, then save them as:

```text
observations/example/
  scene_view.png
  left_wrist_view.png
  right_wrist_view.png
  state.json
```

Images must be **640×480 RGB**. `state.json` is an array of 20 numbers: 10 for
the left arm, then 10 for the right. Each arm's state contains:

- **Position:** gripper-body XYZ in meters, in that arm's base frame.
- **Rotation:** the first two columns of the rotation matrix, stored as
  `[R00, R10, R20, R01, R11, R21]`.
- **Gripper opening:** 0 closed, 1 open.

Use the gripper-body pose rather than the fingertip grasp point or joint angles.

```sh
uv run --locked python run_policy.py \
  --observation observations/example \
  --task 'Pick up the red block and place it in the blue tray.' \
  --seed 0 --output output/prediction.json
```

The command saves the request state, raw prediction, session ID, timing, and
indices of rows with out-of-range gripper values. The source images stay in
the observation directory.

## Use predictions in your application

`RhoClient.predict()` accepts a dictionary of three RGB `uint8` NumPy arrays,
keyed by camera name, and the 20-value state. Keep the session open to make
successive predictions. For example, wire in your observation and execution
callbacks:

```python
from rho_client import RhoClient

async def run_episode(read_observation, execute_actions, steps=10):
    async with RhoClient("Pick up the red block") as policy:
        for _ in range(steps):
            frames, state = await read_observation()
            result = await policy.predict(frames, state)
            prediction = result["prediction"]
            horizon = prediction["execution_horizon"]
            await execute_actions(prediction["actions"][:horizon])
```

In this example, `read_observation()` returns synchronized images and measured
state. `execute_actions()` returns after executing the selected targets through
your controller. Adapt that loop to your application's scheduling and execution
policy. Sessions have a 600-second budget; open a new session for longer runs.

Targets describe absolute end-effector poses using the same ordering and units
as the input state. Your controller handles frame conversion, IK, gripper
mapping, and any smoothing or limits. Raw gripper predictions outside `[0, 1]`
are preserved and listed in `result["invalid_gripper_rows"]`.

## Response and timing

| Field | Value |
|---|---|
| Camera tracks | `scene_view`, `left_wrist_view`, `right_wrist_view` |
| `actions` | `(50, 20)` absolute end-effector targets, left arm first |
| `execution_horizon` | 25, the model's recommended action prefix |
| `step` | The request's `chunk_id` |
| `source_capture_us` | The request's transport `capture_us` |
| `view_skew_us` | 0 for the synchronized snapshot publications |

Each call holds its input snapshot fixed while publishing it through the video
encoder. Repeated frames receive fresh transport timestamps for request/response
matching; those timestamps do not represent the original sensor capture time.
Account for that distinction when integrating a live camera and control loop.

The client validates the response shape, rotations, action prefix, and request
correlation. A timeout, model error, or mismatch invalidates the session;
reconnect before retrying.

## Model

Send measured state in the units described above. The hosted model handles
normalization; you don't need to normalize the values before calling `predict()`.

Microsoft publishes the upstream checkpoint at
[microsoft/rho-yam-box](https://huggingface.co/microsoft/rho-yam-box) and its code
at [microsoft/rhobotics](https://github.com/microsoft/rhobotics).
