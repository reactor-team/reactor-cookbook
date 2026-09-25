# FLUX 0.3.0 quickstart

This client drives `reactor/flux3-action-droid`, the hosted FLUX DROID/Franka
policy. It streams three RGB camera views, sends measured robot state and a
task instruction, and receives **32 × 8 absolute joint/gripper targets**. It
prints predictions; it does not send them to a robot.

Release **0.3.0** offers six checkpoints. Choose one when creating the client;
the client discovers the server's available choices and verifies the selection
before sending a prediction. Reset preserves that choice. Create a **new session**
to change checkpoints.

Start with the CLI below, then use [your own observations](#replay-your-own-observations)
or the [Python API example](#call-the-api-from-your-own-python-code).

## Run the quickstart

You need [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/),
and a Reactor account. Create a key on the
[API keys page](https://reactor.inc/account/api-keys).
`reactor/flux3-action-droid` is public on the production API; the quickstart
requires a valid API key and available serving capacity.
No local GPU, simulator, Docker, or weights are needed. The commands below let
uv install Python 3.12; the client supports Python 3.10+.

From a new terminal:

```sh
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/robotics/flux3-action-droid/client-python
uv sync --locked --python 3.12
export REACTOR_API_KEY='<your key>'
uv run python main.py --checkpoint gd-fp8 --requests 5
```

If you already cloned the repo, start from its root and run
`cd robotics/flux3-action-droid/client-python`. Keep your key in the environment;
do not paste it into a Python file or commit it. This client prints no credentials.

You should see output like this (times vary):

```text
Synthetic protocol smoke test (not task quality)
Available: base-bf16, base-fp8, gd-bf16, gd-fp8, sd-bf16, sd-fp8
Pinned: gd-fp8
step=0 checkpoint=gd-fp8 shape=(32, 8) model=137.2 ms RTT=240.6 ms
...
Reset check: step=5, checkpoint=gd-fp8, shape=(32, 8)
PASS: 6 valid predictions; session closed
```

The worker can take several minutes to become ready after a cold start; the
client waits for readiness before publishing cameras or requesting actions.

The default checkpoint is `base-bf16`. The script makes five predictions, resets
the episode, makes one more prediction with the same checkpoint, and closes the
session even on failure. All observations in this default run are synthetic:
seeded random RGB images and a fixed Franka pose. They test the API, not task
success. Nothing is actuated or scored as robot behavior.

The endpoint defaults to `https://api.reactor.inc`. Set `REACTOR_API_URL` to
use another environment, which must offer the selected checkpoint. The model
name resolves the currently deployed release; this command does not pin a
historical release version. This example targets the 0.3.0 contract.

This example pins **reactor-sdk 1.6.0**, using native `Track.push_frame` and SDK
keepalive. Its lockfile captures the tested dependencies. Run the setup commands
from this example's directory; the other [policy quickstarts](../sim/notebooks)
have their own project environment and dependency versions.

## Checkpoint choices

| `--checkpoint` | Weights | Sampling steps |
| --- | --- | ---: |
| `base-bf16` | Base, BF16 (default) | 4 |
| `base-fp8` | Base, FP8 | 4 |
| `gd-bf16` | Guidance-distilled, BF16 | 4 |
| `gd-fp8` | Guidance-distilled, FP8 | 4 |
| `sd-bf16` | Step-distilled, BF16 | 1 |
| `sd-fp8` | Step-distilled, FP8 | 1 |

These are different weight packages; lower latency does not establish equivalent
robot task quality. Always use discovery (`get_checkpoint`) for the choices on
your worker. An unavailable choice fails explicitly; there is no fallback.

The session API is:

```python
# After READY, before the first prediction, using an ordinary Reactor SDK object:
status = await reactor.send_command("get_checkpoint", {})
available = status["data"]["available"]
reply = await reactor.send_command("select_checkpoint", {"checkpoint": "gd-fp8"})
if (
    not reply
    or reply.get("type") != "checkpoint_selected"
    or reply.get("data", {}).get("checkpoint") != "gd-fp8"
    or reply.get("data", {}).get("locked") is not True
):
    raise RuntimeError("Checkpoint selection was not confirmed")
```

[`select_checkpoint`](client-python/client.py) adds availability and existing-pin
checks. Repeating the same choice is accepted. Switching a pinned session emits
`command_error`; resetting or reconnecting that session does not unlock it.
Without explicit selection, the first valid prediction pins the server default
(`base-bf16` in 0.3.0).

The optional `--seed` is an integer from 0 through `2**63 - 1`. Omitting it
uses the request's `chunk_id` as the seed. It controls sampling noise, not the
server's execution recipe.

## Replay your own observations

```sh
uv run python main.py --checkpoint base-bf16 --observations observations.npz --requests 5
```

Supply an NPZ with these arrays; load uses `allow_pickle=False`:

| Array | Shape and meaning |
| --- | --- |
| `wrist_view` | `(N, H, W, 3)` uint8 RGB wrist camera |
| `exterior_view_1` | `(N, H, W, 3)` uint8 RGB first exterior camera |
| `exterior_view_2` | `(N, H, W, 3)` uint8 RGB second exterior camera |
| `proprio` | `(N, 8)`: seven measured joint angles in radians, then gripper closed fraction (0 open, 1 closed) |
| `task` | `(N,)` Unicode strings, each 1–300 characters |

To exercise the file path before connecting your cameras, create a synthetic
NPZ in the same client directory and replay it:

```sh
uv run python - <<'PY'
import numpy as np
from client import VIEWS
from main import observations

rows = observations(None, 5)  # synthetic fixtures, not recorded robot behavior
np.savez(
    "observations.npz",
    **{view: np.stack([frames[view] for frames, _, _ in rows]) for view in VIEWS},
    proprio=np.stack([state for _, state, _ in rows]),
    task=np.asarray([task for _, _, task in rows], dtype=str),
)
PY
uv run python main.py --checkpoint gd-fp8 --observations observations.npz --requests 5
```

Replace those arrays with your own captured RGB frames, measured states, and
instructions using the same keys and shapes. Each row is one observation; the
client does not execute predictions or evolve a simulator between rows.
The synthetic example uses 360 × 640 frames; the model resizes inputs itself.
Keep all three camera identities consistent with the DROID observation. Do not
substitute DreamZero's differently named tracks or state commands.

## Call the API from your own Python code

Save the following as `example.py` next to `client.py` and run
`uv run python example.py`. `FluxClient` is a helper shipped in this cookbook,
not an import provided by the SDK. This complete example uses one synthetic
observation; replace the `observations` call with your camera/state capture.

```python
import asyncio

from client import FluxClient
from main import observations


async def main():
    frames, proprio, task = observations(None, 1)[0]
    async with FluxClient(checkpoint="gd-fp8") as client:
        prediction = await client.predict(frames, proprio, task, seed=0)
        print(prediction.actions.shape)       # (32, 8)
        print(prediction.actions[0].tolist()) # first target: seven joints + gripper
        print(prediction.checkpoint, prediction.step)
        await client.reset()                 # same checkpoint for the next episode


asyncio.run(main())
```

For a capture loop, keep the context open, capture all three views and state,
then `await client.predict(...)` for each observation. The instruction is sent
again when `task` changes. Close the context before opening a session with a
different checkpoint. If a request fails or times out, leave the context and
investigate before starting a new session; do not keep pipelining requests.

## Wire contract and timing

1. Register handlers, connect, wait for `READY`, select the checkpoint, then
   publish all three tracks at 15 Hz. Tracks repeat the current observation.
2. Send `set_task_description` with `{"task_description": "put the marker in the cup"}`.
3. Publish the observation, allow a short settling interval, then send
   `set_state_json` with a **string** as its `state_json` field:

   ```python
   import json

   await reactor.send_command("set_state_json", {"state_json": json.dumps({
       "proprio": [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8, 0.0],
       "chunk_id": 0,
       "seed": 0,
   })})
   ```

4. Wait for `action_prediction` whose `data.step` echoes `chunk_id`. Its `data`
   also contains `checkpoint`, `actions`, and `inference_seconds`. Keep only one
   request outstanding; the client serializes calls and discards delayed replies
   for other IDs. A byte-identical request is deduplicated, so IDs advance even
   across reset. Timeout/error ends the quickstart; it does not silently retry.

Each action row contains seven **absolute joint targets in radians**, followed
by gripper closed fraction. Row `k` targets `(k + 1) / 15` seconds after the
observation: 32 rows cover approximately 2.13 seconds. These are not deltas or
end-effector poses. A real controller must validate limits, observation age,
and choose its replan/execution horizon before commanding hardware.

The 300 ms settling interval (`--settle-s`) is a transport heuristic, **not an
acknowledgement of camera frame identity or synchronized capture**. The echoed
request ID correlates state and response; it does not prove which decoded pixels
were consumed. Video encoding can change those pixels. This client does not
claim frame-metadata synchronization behavior.

Printed model time comes from `inference_seconds`. Request RTT runs from state
submission to matching response and excludes camera settling, connection, and
checkpoint setup. It is not complete sensor-to-actuator latency. The first task
can take longer to encode. Treat these small-sample numbers as observations,
not a paper benchmark or latency guarantee.

## What this checks

- Checkpoint discovery, selection acknowledgement, and the checkpoint on every reply.
- Matching request IDs and finite `(32, 8)` actions.
- Reset retaining the checkpoint and a new request receiving a valid response.
- Session cleanup on success and failure.

Release 0.3.0 uses the **fast execution recipe**. A fixed `--seed` fixes sampling
noise, but does not promise bitwise identical actions across runs or workers.
Deterministic execution is a server configuration, not a per-session option.
The example deliberately has no golden-action equality assertion. Neither its
synthetic smoke test nor NPZ replay establishes closed-loop simulation success.
There is no FLUX closed-loop simulator integration included in this entry.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `REACTOR_API_KEY` is missing | Export a key in the same terminal that runs `uv run`. |
| Authentication/access failure, or model unavailable | Confirm the key and `REACTOR_API_URL`. FLUX is public at `https://api.reactor.inc`; another environment may not offer it. Contact Reactor if a valid key still cannot connect. Do not share the key in logs. |
| HTTP 429 `no available capacity` | Wait for capacity, including the previous session's worker release, before retrying. An open session reserves a worker. |
| No SDK wheel or import errors | Run `uv sync --locked --python 3.12` in this example's directory. SDK 1.6.0 requires a supported platform; Linux wheels require glibc 2.34+. |
| Checkpoint unavailable or discovery failed | The endpoint may serve an older release or a smaller checkpoint set. Inspect the reported choices; the client does not silently substitute one. |
| No action before the timeout, or `command_error` | Check all three RGB views, measured state, task, and transport. Read the error, close the session, and correct the cause before retrying. |
| Different actions with the same seed | Fast execution and encoded camera pixels can vary. This is not an exact-replay test. |

The quickstart does not create a retry loop or additional sessions. Ctrl+C exits
through the same cleanup path as an error. The environment variable
`REACTOR_API_URL` can override the default production endpoint, so check it when
results or available checkpoints differ from this guide.

### Verification

Verified against deployed release 0.3.0 on September 25, 2026 (UTC), with SDK
1.6.0: the documented NPZ creation and replay commands ran from a fresh Python
3.12 environment installed with `uv sync --locked`. All six checkpoints were
discoverable, `gd-fp8` was selected, and five requests plus one after reset
returned valid `(32, 8)` chunks with the expected checkpoint and request ID.
The session closed successfully. The five pre-reset requests had a 104 ms
median model time and 201 ms median request RTT. An earlier direct synthetic
run also passed four predictions, including reset.

After the model became public, the documented synthetic CLI run also passed
with a separately supplied API key: six valid predictions including reset,
all six checkpoints discovered, and successful session cleanup. Its five
pre-reset requests had a 104 ms median model time and 185 ms median request RTT.

The offline tests cover the client, malformed inputs, cleanup, NPZ loading,
links, and execution of the complete Python example against a fake SDK. The
live tests exercise API-key authentication and inference, not account creation
or billing setup. These checks establish API wiring,
not robot task quality or latency comparisons across checkpoints.

## Offline checks

```sh
cd robotics/flux3-action-droid/client-python
uv run python -m unittest discover -s tests -v
```
