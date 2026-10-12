# Hosted Rho YAM client

Scripts to call **`reactor/rho-yam-box`** on Reactor. Provide
three camera images, the robot state, and a task instruction; receive a
`(50, 20)` action prediction. No simulator, robot assets, weights download,
GPU, or robot drivers are installed. The script saves predictions and does
not execute them.

## Setup

Install [uv](https://docs.astral.sh/uv/) and `make`, then run:

```sh
cd robotics/embodiments/yam
make setup
make test
export REACTOR_API_KEY='YOUR_OWN_API_KEY'
make smoke
```

Python 3.12+ is required; uv can provision it. Use your own key from
[Reactor API keys](https://reactor.inc/account/api-keys). Your account must
have access to `reactor/rho-yam-box`; the model operator must arrange that
access separately. Do not send a developer or administrator key with the
scripts. Keys are read only from the environment in Python.

`make smoke` sends synthetic images and a valid example state to check the
hosted connection and response format. It is not a task-quality test. It
creates a live inference session, saves a JSON response under `output/`, and
disconnects. Local tests use mocked services and need no credentials.

## Send your own observation

Capture the three views and the state together. Put them in a directory:

```text
observations/example/
  scene_view.png
  left_wrist_view.png
  right_wrist_view.png
  state.json
```

Each image must be **640×480 RGB**. `state.json` is a JSON array of 20 numbers:
the left arm's 10 values followed by the right arm's 10 values. Each arm has
**XYZ in meters, a 6D rotation, and gripper opening**. XYZ and rotation describe
the gripper body in that arm's base frame. Rotation is the first two columns
of its rotation matrix, column by column:
`[R00, R10, R20, R01, R11, R21]`. Gripper opening is **0 closed, 1 open**.
Use measured poses, not joint angles or fingertip grasp-site coordinates.

```sh
uv run --locked python run_policy.py \
  --observation observations/example \
  --task 'Pick up the red block and place it in the blue tray.' \
  --seed 0 --output output/prediction.json
```

The JSON output includes the exact request state, raw prediction, session ID,
timing, and rows with out-of-range gripper values. Camera inputs stay in your
observation directory. No clipping, smoothing, normalization override, or
action execution is applied by this client.

## Call it from your own Python integration

Reuse a session for successive observations:

```python
from pathlib import Path
from rho_client import RhoClient
from run_policy import load_observation

async def predict_once():
    frames, state = load_observation(Path("observations/example"))
    async with RhoClient("Pick up the red block") as policy:
        result = await policy.predict(frames, state, seed=0)
        actions = result["prediction"]["actions"]
        return actions
```

Pass a dictionary of three NumPy RGB `uint8` arrays and the 20-value measured
state to `predict()`. You may call it again inside the same context with a
new observation after your application has handled the previous prediction.
The session budget is 600 seconds; open another session for longer work.

The client sends frozen snapshots, repeating them through the video encoder
with fresh transport timestamps. These timestamps establish correspondence
between the request and response; **they are not the original sensor capture
times of saved images**. This example does not provide a continuously streaming
camera loop, real-time scheduling, or a hardware controller.

## Response contract and current limitations

| Field | Expected value |
|---|---|
| Hosted model | `reactor/rho-yam-box` |
| Camera tracks | `scene_view`, `left_wrist_view`, `right_wrist_view` |
| `actions` | 50 rows × 20 values, absolute end-effector targets |
| Arm ordering | Left first, then right; same pose/gripper convention as state |
| `execution_horizon` | 25, the served recommended prefix |
| `step` | Matches the request's `chunk_id` |
| `source_capture_us` | Matches the request's transport `capture_us` |
| `view_skew_us` | 0 for these synchronized replay publications |

Actions are **not joint commands**. The gripper-body pose differs from a grasp
point between the fingers. A hardware integration must supply its own verified
frame mapping, IK/controller, gripper conversion, and observation timing.
The client retains raw out-of-range gripper outputs and flags their row
indices rather than silently repairing them. A timeout, model error, or
correlation mismatch invalidates the session; reconnect before retrying.

This contract matches the inspected 0.1.2 serving behavior. The hosted name is
not an immutable release pin; the client checks schema, response shape,
execution horizon, and correlation, but does not verify deployed weight or
normalization hashes. A model update may require a client update.

The earlier YAM experiments found that state normalization materially changed
behavior. This handoff sends measured state unchanged and uses production's
normalization. It does not carry the experimental simulator correction.
Grasp closure and reliable task success remain unverified; a valid response
or a completed smoke test does not establish either.

Microsoft's upstream model is
[microsoft/rho-yam-box](https://huggingface.co/microsoft/rho-yam-box), with code
at [microsoft/rhobotics](https://github.com/microsoft/rhobotics).

## What to share

Share this directory, including `uv.lock`, and the hosted model identifier.
Arrange model access for the recipient's account and have them set their own key.
Their first check is `make smoke`; then send a recorded observation and inspect
the raw output. Do not include credentials or private observation recordings
in the source package.
