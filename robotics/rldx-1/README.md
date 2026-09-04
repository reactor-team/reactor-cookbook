# RLDX-1: Synchronizing Multi-Camera Input with Frame Metadata

[RLDX-1](https://github.com/RLWRLD/RLDX-1) is a vision-language-action
(VLA) model with a Qwen3-VL backbone and a diffusion / flow-matching action
head. The client publishes three camera views and robot state; the model returns
16-step action chunks over the data channel. It does not generate video.

This client connects to the published `rldx-1` model on Reactor. It also shows
how to solve three common robot-policy transport problems:

- keeping independently delivered camera views aligned; and
- identifying which observation produced an action chunk; and
- letting the client own inference timing with Real-Time Chunking (RTC).

The model announces its exact contract in the `model_schema` handshake. For the
current checkpoint, that contract is:

- **Input:** three 256×256 RGB tracks (`left_view`, `right_view`, `wrist_view`),
  robot proprioception, and a language task. The model waits for at least one
  frame from every view.
- **Output:** `action_prediction` messages containing a 16-step chunk:
  `end_effector_position` `[16,3]`, `end_effector_rotation` `[16,3]`,
  `gripper_close` `[16,1]`, `base_motion` `[16,4]`, and `control_mode` `[16,1]`.
- **Commands:** `get_schema`, `reset`, `set_state_json`,
  `set_task_description`, and, when RTC is enabled, `request_action`.

The client reads the announced views, resolution, control rate, state layout,
and state carrier instead of hardcoding them.

To deploy the matching B200 model with guided RTC, use the
[`models/rldx-1`](../../models/rldx-1) workspace.

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

### 4. Schedule RTC on the client control clock

When `model_schema.inference_trigger` is `client_request`, the example switches
from passive streaming to an RTC plan scheduler:

1. It holds safely while requesting the first plan with no prefix.
2. Every `exec_horizon` control steps, it requests a replacement and sends the
   next `rtc_delay` physical actions from the active plan.
3. It keeps executing that base plan while inference runs.
4. At `install_step`, it discards the returned prefix and installs the suffix.

Each request names the active `base_plan_id`; each response must echo the
request, plan, prefix, and install metadata exactly. A missing deadline or bad
plan chain triggers `reset` and returns the client to safe hold. The action
prefix is flattened in the handshake's `action_order` using `action_dims`.

Deployments that announce `inference_trigger: streaming` keep the existing
server-paced behavior automatically.

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

| Key          | Meaning                                           | Echoed as           |
| ------------ | ------------------------------------------------- | ------------------- |
| `capture_us` | Snapshot time in microseconds on the client clock | `source_capture_us` |
| `seq`        | Client tick counter                               | `source_seq`        |

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

## Run

```bash
cd robotics/rldx-1/client-python
uv sync
export REACTOR_API_KEY=rk_your_key_here
uv run python main.py --model <account-slug>/rldx-1 --duration 60
```

Replace `<account-slug>` with the account slug printed by Reactor when the
model is published.

Optionally provide a task:

```bash
uv run python main.py \
  --model <account-slug>/rldx-1 \
  --task "put the cup on the tray"
```

The example publishes synthetic frames and synthetic state. Its actions are
well-formed but do not represent meaningful robot behavior. Replace the frame
and state generators and the `Client.execute_action` method before using it
with a robot; that method is the local controller seam.

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
RTC: requests=<count> actions_executed=<count> resets=0
RTC request-to-response ms: p50=<measured> p99=<measured>
inter-arrival ms: p50=<measured> p99=<measured>
chunk age on our clock ms: p50=<measured> p99=<measured>
echo correlation: 49/49 chunks echoed a stamp matching the tick we sent
view_skew_us: p50=0 p99=0 max=0 (across 49 chunks)
command_errors: none — the state carrier worked
```

The placeholders above are populated by the test run. In server-paced streaming mode,
`inter-arrival` follows the server's execution horizon. In RTC mode, requests
follow the client's announced `exec_horizon`; a late response is rejected
instead of shifting that control timeline. `view_skew_us` should be **0** when
one capture time is stamped across a tick's views: every view then contributed
a frame from the same declared instant.

The metrics answer different questions:

| Metric                    | Meaning                                                                                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RTC request-to-response` | Time from sending `request_action` until its matching action chunk reaches the client.                                                                                                                   |
| `inter-arrival`           | How often action chunks arrive; this is cadence, not per-chunk latency.                                                                                                                                  |
| `chunk age`               | How old the source observation is when the client receives its action chunk.                                                                                                                             |
| `echo correlation`        | Whether each returned tag matches a tick the client actually sent.                                                                                                                                       |
| `view_skew_us`            | Capture-time spread across the views used for the chunk. Expect 0 when a tick is stamped once; a value near one control period means a view lagged a whole step and the rest were held back to match it. |

If the deployment does not announce `state_source`, the client falls back to
`set_state_json`. It reports chunk age and echo correlation as unavailable when
the deployment does not return the echo fields. `view_skew_us` can still be
available because the frames retain their shared `capture_time_us`.

## Apply the pattern in your own client

Read the model handshake, stamp every view from one observation with the same
`capture_time_us`, and attach that observation's state bytes to each frame when
`state_source` is `frame_metadata`. For streaming mode, use the echoed
`source_capture_us` and `source_seq`. For RTC, keep a logical control-step
cursor, execute the old plan during inference, and install only an on-time,
correctly chained suffix.

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
