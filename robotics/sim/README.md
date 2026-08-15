# Robotics: Sim & Policy Quickstart Examples

Runnable examples of driving robotics policies served on Reactor. Start with
the quickstarts, which replay recorded observations against a hosted policy
and need no local install beyond Python and
[uv](https://docs.astral.sh/uv/). The five sim packages then drive a real
physics simulator against the same served policies, using the Python
[reactor-sdk](https://pypi.org/project/reactor-sdk/) transport.

## Quickstarts

**[`notebooks/`](./notebooks)** holds six runnable Python scripts, each with a
guide, that drive a **hosted** Reactor robotics policy by replaying recorded
observations. **No simulator, no GPU, no model weights**: `uv sync`, an API
key, and a few minutes. Open the one for your model; there is no reading order.

- [`lingbot_va_quickstart.md`](./notebooks/lingbot_va_quickstart.md):
  `lingbot-va`, LIBERO manipulation. Lock-step, driven by an executed-action
  echo: one `(16, 7)` chunk of end-effector deltas per echo, p50 208 ms. Its
  paper's benchmark is [`libero/`](./libero) below.
- [`cosmos_droid_quickstart.md`](./notebooks/cosmos_droid_quickstart.md):
  `cosmos-nano-policy-droid`, DROID/Franka on a Cosmos3 backbone. Stateless,
  with an executed-step echo as flow control and no `reset` on the wire:
  `(32, 8)` absolute joint targets, a 2133 ms chunk budget. Its paper's
  benchmark is [`cosmos-droid/`](./cosmos-droid) below.
- [`xwam_quickstart.md`](./notebooks/xwam_quickstart.md): `xwam`,
  bimanual manipulation. Lock-step: you ask, it answers a `(32, 14)` action
  chunk. Replays five real RoboTwin 2.0 evaluation requests and checks them
  numerically against what the authors' own serving stack returned. This is
  the reference implementation of the generic contract.
- [`groot_n17_quickstart.md`](./notebooks/groot_n17_quickstart.md):
  `groot-n17`, NVIDIA Isaac-GR00T N1.7 on DROID/Franka. Free-running: it
  predicts every engine tick (~100 ms), and the client filters chunks by
  engine ordering. `(40, 17)` split across three named fields; real FR3 rig
  frames.
- [`dreamzero_quickstart.md`](./notebooks/dreamzero_quickstart.md):
  `dreamzero`, a 14B world-action model on the DROID/Franka embodiment.
  Free-running: it broadcasts `(24, 8)` chunks and the client uses `obs_seq`
  to know which observation a chunk actually saw.
- [`xr1_robocasa365_quickstart.md`](./notebooks/xr1_robocasa365_quickstart.md):
  `xr1-robocasa365`, Xiaomi's XR-1 fine-tuned for RoboCasa365 kitchen
  manipulation. Lock-step, echo-gated from the first request: `(16, 60)`
  packed chunks of which the first 12 columns are live. Its paper's
  benchmark is [`robocasa365/`](./robocasa365) below.

[`notebooks/README.md`](./notebooks#choose-a-model) has a model matrix covering
the wire protocol, action chunk, and available closed-loop harness.

The client package the scripts import (`notebooks/reactor_robotics/`) is
the same client an eval harness or a controller for a physical robot uses; see
[robot-policy-client-contract.md](./notebooks/robot-policy-client-contract.md).

## The sim packages

The packages below are the other direction: a real simulator, driven locally,
with the policy served remotely. Each is the benchmark its policy's paper
used.

- [`libero/`](./libero): LIBERO (robosuite/MuJoCo), driven lock-step
  by `lingbot-va`.
- [`cosmos-droid/`](./cosmos-droid): RoboLab, NVIDIA's Isaac Sim
  DROID benchmark, driven by `cosmos-nano-policy-droid`, one whole chunk per
  request.
- [`robotwin/`](./robotwin): RoboTwin 2.0, driven lock-step by `xwam`
  through the authors' own evaluation client, unmodified.
- [`dreamzero/`](./dreamzero): RoboLab again, driven by `dreamzero`,
  which broadcasts action chunks instead of answering requests.
- [`robocasa365/`](./robocasa365): RoboCasa365 (robosuite/MuJoCo), driven
  lock-step by `xr1-robocasa365` through the vendor's own rollout loop,
  unmodified.

Each package is trimmed to the wiring: a wire contract (`contract.py`), video
tracks (`tracks.py`), a reactor-sdk bridge (`bridge.py`), and whatever owns
the sim side. `libero` wraps its simulator as a Python library, so it has
an env wrapper (`env.py`) and a rollout loop (`loop.py`). The other three
talk to a simulator that runs in its own process and owns its own episode
loop, so each is a gateway (`gateway.py`): the port the simulator connects
to. `robocasa365` needs neither: its simulator environment can host the
reactor-sdk directly, so it is a drop-in client (`client.py`) the vendor's
loop imports. No sim assets and no model weights are vendored here.

|              | `libero` | `cosmos-droid` | `robotwin` | `dreamzero` | `robocasa365` |
|--------------|--------------|--------------------|----------------|-----------------|------------------|
| Simulator    | LIBERO (robosuite/MuJoCo) | RoboLab (Isaac Sim) | RoboTwin 2.0 | RoboLab (Isaac Sim) | RoboCasa365 (robosuite/MuJoCo) |
| Policy       | `lingbot-va` | `cosmos-nano-policy-droid` | `xwam` | `dreamzero` | `xr1-robocasa365` |
| Layout       | env wrapper + rollout loop | gateway (openpi WebSocket) | gateway (the authors' pickle-over-zmq port) | gateway (openpi WebSocket) | drop-in client, imported by the vendor's loop |
| Protocol     | lock-step: execute a chunk, echo it, wait for the next | one chunk per request, executed in full | lock-step: one request out, block for the reply | free-running: the model broadcasts, and chunks are matched to observations by `obs_seq` | lock-step: echo-gated from the first request, one chunk per executed-step echo |
| Action chunk | `(16, 7)` end-effector deltas | `(32, 8)` absolute joint targets | `(32, 14)` | `(24, 8)` | `(16, 60)` packed, first 12 live |
| Video        | two tracks (`agentview`, `eye_in_hand`), one frame per env render, plus a heartbeat | three tracks, one frame per request, plus a heartbeat | three tracks, repeating the current frame between requests | three tracks, queue-fed, no repeats | three tracks, a 4-frame history per request, slots pushed as per-camera sets |

Threading comes up only where the simulator runs inside the example's own
process: `libero` keeps the env on the main thread and moves the reactor
bridge to a thread of its own, with a lock-guarded hand-off (`RolloutState`)
between them. `cosmos-droid` also gives the bridge its own thread,
because the openpi server blocks the main thread for the whole of every
request. Each package's README covers its threading where it affects usage.

## Setup

Each example is its own installable package with its own `pyproject.toml` and
README, which has its exact setup and run instructions. All five need Python
3.10+ (`libero` pins `<3.11`) and a real simulator install:

| Example | Simulator you install | Rough cost |
|---|---|---|
| [`libero/`](./libero) | LIBERO | no GPU, 20-40 min |
| [`cosmos-droid/`](./cosmos-droid) | the `robolab` Docker image | RTX-class GPU, hours |
| [`dreamzero/`](./dreamzero) | the `robolab` Docker image | RTX-class GPU, hours |
| [`robotwin/`](./robotwin) | RoboTwin 2.0 + the authors' eval client | CUDA GPU, hours |
| [`robocasa365/`](./robocasa365) | RoboCasa365 via a Xiaomi-Robotics-1 checkout | CUDA GPU, hours |

The three gateways install with no simulator dependency at all, so you can
`uv pip install -e .` and run their `check_wiring.py` on a laptop before
committing to the heavyweight half.

`robotwin` needs two virtualenvs rather than one: RoboTwin 2.0 pins
`numpy==1.23.5` and the gateway package installs `numpy>=1.26`, so the
simulator and the gateway cannot share an environment. Its README covers it.

## Your API key

Create one at [reactor.inc/account/api-keys](https://reactor.inc/account/api-keys).
`reactor-sdk` exchanges `REACTOR_API_KEY` for a session JWT by sending the key
to the Reactor API's `/tokens` endpoint over HTTPS. The examples never print or
log the key.
