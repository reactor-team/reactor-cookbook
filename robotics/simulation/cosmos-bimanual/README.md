# Simulate AgiBot actions with Cosmos3 Nano Sim Bimanual

Send controller poses and gripper values to a hosted neural simulator and receive
video of the scene responding. This is action-conditioned simulation, not a
policy that produces robot actions. The checkpoint targets the AgiBot embodiment;
an image of another robot does not make that embodiment supported.

## Run a short rollout

Requires Python3.11+, uv, and a Reactor API key with access to
`reactor/cosmos3-nano-sim-bimanual`. No local GPU or model download is needed.
The Python SDK uses the unqualified model name in `session.py`.

Use NVIDIA's public AgiBot example for your first run. The command below downloads
only the starting image from a pinned checkpoint revision, not model weights.
See the [source checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)
and its [declared license](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/blob/441c3ef075c8beee8bf4d04fb5cb423d982f75bc/README.md).
The image remains hosted by NVIDIA and is not redistributed in this cookbook.
Later, substitute your own compatible AgiBot head-camera PNG/JPEG.

```bash
cd robotics/simulation/cosmos-bimanual
export REACTOR_API_KEY='your-key'
uv sync --locked
mkdir -p inputs
curl --fail --location \
  'https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/441c3ef075c8beee8bf4d04fb5cb423d982f75bc/assets/sim_bimanual/agibot/images/agibot_original_long60_019_initial.png' \
  --output inputs/agibot.png
uv run rollout.py --image inputs/agibot.png \
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

## Try your own controller inputs

```bash
uv run interactive.py --image inputs/agibot.png \
  --prompt 'Move both robot arms carefully over the table.'
```

Enter `left` or `right` to select an arm. `x+`, `x-`, `y+`, `y-`, `z+`, and
`z-` adjust the selected controller position by1cm; this example clamps offsets
to±10cm. `open` and `close` set its gripper. `reset` starts a new world from the
current pose, `snap` saves the latest preview, and `quit` disconnects.
Each motion/gripper command waits for20 received preview frames and saves a PNG.
Open those files to inspect the scene. This is a terminal integration example,
not a live graphical viewer. Preview buffering means a saved frame is not a
correlated observation for a particular command.

The model continues simulating between inputs: not typing holds the latest
controller sample; it does not pause time. This example controls a learned
simulated scene only, not a physical robot. No policy-evaluation claim is made.

Interactive production verification (2026-10-20): selected both arms, changed
X/Y positions, closed/opened grippers, reset, saved six snapshots, and exited
cleanly after185 received preview frames.
