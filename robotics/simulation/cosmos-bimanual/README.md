# Simulate AgiBot actions with Cosmos3 Nano Sim Bimanual

Send controller poses and gripper values to a hosted neural simulator and receive
video of the scene responding. This is action-conditioned simulation, not a
policy that produces robot actions. The checkpoint targets the AgiBot embodiment;
an image of another robot does not make that embodiment supported.

## Run a short rollout

Requires Python3.11+, uv, and a Reactor API key with access to
`reactor/cosmos3-nano-sim-bimanual`. No local GPU or model download is needed.
The Python SDK uses the unqualified model name in `session.py`.

Supply a PNG/JPEG head-camera image of a supported AgiBot scene that you have
permission to use. Start with a scene matching the public NVIDIA checkpoint's
examples. This repository does not redistribute robot captures.

```bash
cd robotics/simulation/cosmos-bimanual
export REACTOR_API_KEY='your-key'
uv sync --locked
uv run rollout.py --image /path/to/agibot.png \
  --prompt 'Move both robot arms carefully over the table.'
```

The script initializes the image, seed, and controller baseline; moves the left
controller by0.01m on its X axis and closes its gripper; then resets the world.
It saves initial, controlled, and reset preview snapshots under `output/`.
Text supplies task context. Arm movement comes from `set_teleop`, not the prompt.

Positions are absolute controller samples in meters, interpreted relative to the
initial baseline; orientation is quaternion `[x,y,z,w]`. Grip0 means open and1
means closed. The identity pose is a convenient input baseline, not an absolute
robot joint configuration. Controller-to-camera alignment requires calibration;
this example does not promise a particular tabletop direction or task success.

`reset` restarts from the current image and current controller baseline. It does
not restore the original controller values. Prompt changes also restart the world.

## What the example verifies

The client awaits command calls, receives preview frames, and waits for a fresh
step-zero status after reset. Every wait is bounded and the session is closed on
exit. Preview frames can repeat or be dropped; frame counts and snapshots are not
an exact, correlated policy-observation contract or a throughput benchmark.
A valid image and useful output still require embodiment-compatible inputs.

Hosted access failures and command refusals propagate as SDK exceptions. Check
model access and available capacity if connection fails. Sessions are capped at
five minutes to bound resource use. Never commit API keys.

Test target: hosted model0.2.0, runtime3.6.1, Python SDK1.10.0. The model slug follows
the active deployment, so later releases may change generated behavior.

Production verification (2026-10-20): the script received184 preview frames,
saved all three RGB snapshots, and observed a fresh step-zero status after reset.
This verifies API integration, not physical accuracy or manipulation success.
