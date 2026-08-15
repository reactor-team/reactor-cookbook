# robocasa365-sim

RoboCasa365, the kitchen-manipulation benchmark `xr1-robocasa365` was
fine-tuned for, driven by the Reactor-served model. The vendor's evaluation
stack runs **unmodified**: its rollout loop, task registry, seeding and
success criteria are imported from the Xiaomi-Robotics-1 checkout. Only the
client is swapped, so what the benchmark measures does not change; what
changes is that observations travel over WebRTC video tracks and action
chunks come back on a data channel, instead of the vendor's raw TCP socket.

## No gateway, one environment

RoboCasa365's environment tolerates `numpy>=1.26`, which the reactor-sdk
needs, so this package installs **into the simulator's environment** and the
vendor loop imports `ReactorEvalClient` directly. One environment, no relay
port, no second process.

## One `infer()` cycle

The vendor loop calls `client.infer(state_history, image_history,
instruction)` once per replan window and blocks for the chunk. Per call:

1. The task string is sent once, when it changes.
2. The 4-row state history is sent verbatim as `state_history_json`.
3. The loop's own sampled 4-frame histories are pushed, one frame per camera
   per history slot, on three named tracks: `left_agentview`,
   `right_agentview`, `wrist_view`. The model pairs the tracks
   frame-for-frame in arrival order, so the three frames of a slot are
   pushed as a set, and slots are spaced rather than burst.
4. After a short settle (`--settle-s`, default 0.15 s) the executed-step echo
   is sent. The model predicts only on a strictly increasing echo, so this is
   the request; the reply is one `(16, 60)` chunk, of which the vendor loop
   consumes the first 12 columns.

The model runs with `obs_interval: 1`, so its per-view history holds exactly
the four frames each `infer()` pushed: vendor-identical conditioning, with
episode boundaries handled by eviction rather than a reset. What differs
from the socket path, deliberately, is the codec: frames arrive
H264-compressed rather than lossless.

## Install the simulator

Follow upstream: a [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)
checkout plus its RoboCasa365 installation (robosuite/MuJoCo; CUDA GPU,
`MUJOCO_GL=egl`). This package vendors none of it and needs only the
checkout path at run time.

## Install this client

Into the simulator's environment:

```bash
uv pip install -e robotics/sim/robocasa365
```

## Run an evaluation

Point the client at a running `xr1-robocasa365` runtime (a local
`reactor run` / container, or any deployment you can reach; the SDK's local
mode is the path this example was verified over):

```bash
MUJOCO_GL=egl python -m robocasa365_sim.entry \
    --vendor-dir ~/Xiaomi-Robotics-1 \
    --api-url http://127.0.0.1:8080 \
    --task-name CloseBlenderLid --num-trials 3 \
    --save-root-dir eval_results/reactor
```

All vendor knobs (`--task-set`, `--num-trials`, `--replan-steps`, `--seed`,
...) are accepted and forwarded; `--task-set target50` runs the full
benchmark. `--crop-ratio` must match the served model's configured value,
because the crop happens model-side in this topology. The summary lands in
`<save-root-dir>/<run-id>/summary.json` with a `transport` block recording
the wire and per-inference latency.

Over the production wire, this path scored 56.8% / 60.2% episode success at
replan 16 / 8 against 56.4% / 59.2% for the vendor's own socket (1000 paired
episodes, seed-matched; the vendor's published anchor is 57.28%). Measured
on an earlier runtime version.

## Check the wiring

```bash
uv run python check_wiring.py   # offline: no GPU, no network, no simulator
```

It verifies the two mistakes a live run cannot surface, because both look
exactly like a bad policy: a camera mapped to the wrong track, and frames
pushed per-track instead of as per-slot sets (the model pairs its three
tracks frame-for-frame in arrival order).

## Failure modes

- **The session dies between tasks.** The runtime disconnects a client that
  sends nothing for 20 s, and RoboCasa can take longer than that building
  the next task's environment. The client therefore pings every 10 s for
  the whole session (`reactor-sdk==0.8.0` leaves keepalive to the client).
- **Soft, blurry observations tank success.** aiortc's default H264 bitrate
  (1 Mbps, hard max 3) visibly degrades the views; measured cost was ~25
  points of episode success, concentrated in visual-state tasks. The client
  pins 10 Mbps (`XR1_EVAL_H264_BITRATE` to override); on localhost/LAN
  there is no reason to starve the encoder.
- **Predictions stop arriving deep into a long run.** Sessions were observed
  to stall after roughly 1.4k predictions, so the client recycles its
  WebRTC session every 600 (`XR1_CLIENT_SESSION_RECYCLE`; `0` disables).
  Predictions are stateless server-side, so a recycle is invisible to the
  eval loop.
- **Predictions look one observation stale.** Frames travel on video tracks
  and commands on the data channel; `--settle-s` is the pause that lets a
  pushed frame set land before the echo opens the gate. Raise it on a slow
  link.

## Provenance and licensing

The vendor code this example imports (`eval_robocasa365/entry.py`) is
Apache-2.0, from the Xiaomi-Robotics-1 repository; it is imported from your
checkout, not vendored here. `XR1_DEBUG_DUMP_CLIENT=<dir>` dumps the exact
frames sent, per view per history slot, for comparison against a model-side
dump when auditing the wire.
