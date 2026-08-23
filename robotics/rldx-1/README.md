# RLDX-1: Synchronizing Multi-Camera Input with Frame Metadata

[RLDX-1](https://github.com/RLWRLD/RLDX-1) is a vision-language-action
(VLA) model with a Qwen3-VL backbone and a diffusion / flow-matching action
head. The client publishes three camera views and robot state; the model returns
16-step action chunks over the data channel. It does not generate video.

This client works with the published `rldx-1` platform model or the matching
[`models/rldx-1`](../../models/rldx-1) recipe. It also shows how to solve two
common multi-camera problems:

- keeping independently delivered camera views aligned; and
- identifying which observation produced an action chunk.

The model announces its exact contract in the `model_schema` handshake. For the
current checkpoint, that contract is:

- **Input:** three 256×256 RGB tracks (`left_view`, `right_view`, `wrist_view`),
  robot proprioception, and a language task. The model waits for at least one
  frame from every view.
- **Output:** `action_prediction` messages containing a 16-step chunk:
  `end_effector_position` `[16,3]`, `end_effector_rotation` `[16,3]`,
  `gripper_close` `[16,1]`, `base_motion` `[16,4]`, and `control_mode` `[16,1]`.
- **Commands:** `get_schema`, `reset`, `set_state_json`, and
  `set_task_description`.

The client reads the announced views, resolution, control rate, state layout,
and state carrier instead of hardcoding them.

## Why frame metadata matters

WebRTC delivers each camera view on a separate track, while action chunks arrive
on the data channel. Without extra metadata:

- the freshest frame on each track may represent a different moment;
- an action chunk does not identify the observation that produced it; and
- proprioception sent as a separate message becomes another stream that the
  server must pair with the camera frames.

Only the client knows when it sampled its sensors, so it must put that timing on
the wire.

## The synchronization pattern

### 1. Use one capture time for every view in a tick

Read the clock once, then apply that value to every frame in the observation:

```python
from reactor_sdk import time_micros

capture_us = time_micros()
for view, track in tracks.items():
    track.push_frame(frames[view], capture_time_us=capture_us)
```

The declared value—not the time each push happens—lets the server recognize the
frames as one observation. Use the engine's monotonic `time_micros()` clock;
`time.time()` is not a substitute.

### 2. Attach state to the frames

Serialize the proprioceptive snapshot once and attach the same bytes to every
frame from that tick:

```python
track.push_frame(
    frame,
    user_data=state_bytes,
    capture_time_us=capture_us,
)
```

This ties the state to its frames instead of asking the server to correlate a
separate state stream by arrival time.

### 3. Correlate each action with its source observation

Add `capture_us` and `seq` to the state JSON when the handshake announces those
keys. RLDX-1 echoes them as `source_capture_us` and `source_seq` on the action
chunk. The client can then calculate observation age on its own clock:

```python
age_ms = (time_micros() - chunk["source_capture_us"]) / 1000
```

No clock synchronization with the server is required.

## State metadata contract

The state JSON contains every vector announced in `state_dims`, at exactly the
announced length:

```json
{
  "end_effector_position_relative": [0.0, 0.0, 0.0],
  "end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
  "gripper_qpos": [0.0, 0.0],
  "base_position": [0.0, 0.0, 0.0],
  "base_rotation": [0.0, 0.0, 0.0, 1.0],
  "capture_us": 2155019618705,
  "seq": 41
}
```

Build the vectors from `state_dims`; a vector with the wrong length causes the
whole state payload to be rejected. The two optional tag keys are announced in
`state_tag_keys`:

| Key | Meaning | Echoed as |
| --- | --- | --- |
| `capture_us` | Snapshot time in microseconds on the client clock | `source_capture_us` |
| `seq` | Client tick counter | `source_seq` |

Only include keys announced in `state_tag_keys`. Unknown keys can prevent the
state payload from parsing.

The handshake also selects the state carrier:

- `state_source == "frame_metadata"`: attach state to each frame.
- Any other value, or no `state_source`: send state with
  `set_state_json` for compatibility with deployments older than RLDX-1 0.4.0.

The client keeps `capture_time_us` on the frames in both modes, so cross-view
alignment still works when state travels separately.

Newer action messages may include three correlation fields:
`source_capture_us`, `source_seq`, and `view_skew_us`. Any field may be null;
treat null as unavailable, not zero. The client coerces these JSON numbers with
`int()` because the SDK may deliver integer-valued fields as Python floats.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- `reactor-sdk >= 1.1.1` (1.1.0 accepted `capture_time_us` but dropped it on the wire)

Per-frame `user_data`, `capture_time_us`, and `time_micros()` are not available
in the 0.x SDK used by older cookbook examples. The SDK provides wheels for
Linux x86_64 / aarch64 with glibc 2.34+, macOS 11+ arm64 or 13+ x86_64, and
Windows 10+.

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Run locally

First start the matching model recipe, then run:

```bash
cd robotics/rldx-1/client-python
uv sync
uv run python main.py --local
```

## Run with Reactor Cloud

```bash
cd robotics/rldx-1/client-python
uv sync
uv run python main.py --api-key rk_your_key_here
```

Optionally provide a task:

```bash
uv run python main.py \
  --api-key rk_your_key_here \
  --task "put the cup on the tray"
```

The example publishes synthetic frames and synthetic state. Its actions are
well-formed but do not represent meaningful robot behavior; replace both data
sources before using it with a robot.

## What you should see

A deployment using frame metadata reports its discovered schema, selected
carrier, first correlated chunk, and a final summary:

```text
[schema] views=['left_view', 'right_view', 'wrist_view'] resolution=256x256 control_hz=20 ...
[carrier] frame_metadata — the handshake asks for state on the frames ...

[first chunk]
  source_seq=0  source_capture_us=1528765378051
  echo matches the tick we sent: True ...
  age of source snapshot on our clock: 197 ms
  view_skew_us=0 ...

===== RLDX-1 sync summary =====
ticks published: 787 ; action chunks received: 49
state carrier: frame metadata
inter-arrival ms: p50=803 p90=806 (1.2 chunks/s)
chunk age on our clock ms: p50=177 p90=196 (from 49 echoed chunks)
echo correlation: 49/49 chunks echoed a stamp matching the tick we sent
view_skew_us: p50=0 max=0 (across 49 chunks)
command_errors: none — the state carrier worked
```

Two numbers are worth reading carefully. `inter-arrival` should be close to 800
ms because this checkpoint re-plans once per execution horizon (16 steps at 20
Hz = 0.8 s), no matter how fast frames are published. `view_skew_us` should be
**0** when one capture time is stamped across a tick's views: every view then
contributed a frame from the same declared instant.

The metrics answer different questions:

| Metric | Meaning |
| --- | --- |
| `inter-arrival` | How often action chunks arrive; this is cadence, not per-chunk latency. |
| `chunk age` | How old the source observation is when the client receives its action chunk. |
| `echo correlation` | Whether each returned tag matches a tick the client actually sent. |
| `view_skew_us` | Capture-time spread across the views used for the chunk. Expect 0 when a tick is stamped once; a value near one control period means a view lagged a whole step and the rest were held back to match it. |

If the deployment does not announce `state_source`, the client falls back to
`set_state_json`. It reports chunk age and echo correlation as unavailable when
the deployment does not return the echo fields. `view_skew_us` can still be
available because the frames retain their shared `capture_time_us`.

## Apply the pattern in your own model

RLDX-1's server implementation is included in
[`models/rldx-1`](../../models/rldx-1), primarily in `robot_state.py` and
`pipeline.py`. A multi-track model needs three pieces:

1. **Read frame metadata.** Each inbound frame exposes `.metadata` (the bytes
   passed as `user_data`) and `.capture_time_us` (the client-declared stamp).
   Decode and validate the metadata in the model.
2. **Choose the freshest state tag.** Use the tag's `seq` when present, or its
   `capture_us` otherwise. This embedded ordering outranks the frame's
   `capture_time_us`, which in turn outranks arrival order. Keep the
   separate-message carrier as a compatibility fallback.
3. **Align the camera views.** Choose one recent frame per view near the latest
   capture time covered by every view. Return the selected spread as
   `view_skew_us`; if any view lacks a stamp, fall back to newest-per-view and
   report the skew as null.

Echo the selected tag's `capture_us` and `seq` on the result, and announce
`state_source`, `state_tag_keys`, and `state_dims` in the handshake. That lets
clients discover the contract and work across deployment versions.

## Notes

- Model messages use the shape `{"type": "action_prediction", "data": {...}}`.
  The client handles `model_schema`, `action_prediction`, and `command_error`.
- Always surface `command_error` messages. RLDX-1 uses `command="state"` when
  missing-state fallback engages; actions may continue with stale or zeroed
  state.
- The initial `model_schema` message can race the data channel opening. The
  client requests it with `get_schema`, waits, and starts publishing before its
  final retry.
- A deployment scaled to zero may take time to pull its image and weights and
  load the model. This is why the example uses a generous `--connect-timeout`.
