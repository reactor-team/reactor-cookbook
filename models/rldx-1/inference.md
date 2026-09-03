# Running the RLDX-1 inference server + connecting a client

This is the run guide for the self-hosted `rldx-reactor` server. For the
overview see [README.md](./README.md).

## 1. Start the server

```bash
GPUS='"device=0"' WEIGHTS_DIR=/path/to/RLDX-1-FT-ROBOCASA \
  ./scripts/run_reactor.sh start
./scripts/run_reactor.sh status
./scripts/run_reactor.sh logs
```

Wait for the model to finish loading (the log shows checkpoint shards loading,
then the runtime listening on `:8080`).

Useful env knobs (see the script header for the full list):

| var | default | purpose |
|---|---|---|
| `WEIGHTS_DIR` | — (required) | host checkpoint dir, mounted at `/weights` |
| `REACTOR_HTTP_PORT` | `8080` | server port |
| `GPUS` | `all` | `docker --gpus` value, e.g. `'"device=0"'` |

The runtime itself is configured entirely by environment variables (the script
sets `HOST`, `PORT` and `ORPHAN_TIMEOUT_SECONDS`; `REACTOR_LOG_LEVEL`,
`STUN_SERVERS`, `TURN_SERVERS` and `WEBRTC_PORT_RANGE` are the other useful
ones) — there are no serve flags to pass.

Edit `config.yml` (resolution, `embodiment_tag`, `device_id`) and
`./scripts/run_reactor.sh restart` — no rebuild needed (it's bind-mounted).

## 2. Connect a client

The server speaks the standard Reactor WebRTC transport. The model needs at
least one frame on **every** view (`left_view`, `right_view`, `wrist_view`)
before it steps. With RTC disabled, inference runs continuously. With RTC
enabled, `model_schema.inference_trigger` is `client_request` and the client
sends one `request_action` command per plan.

### Python (recommended quick test)

Use the cookbook client
([`robotics/rldx-1/client-python`](https://github.com/reactor-team/reactor-cookbook/tree/main/robotics/rldx-1)).
Use its frame publication and schema-handshake pattern in the client that
connects to this local server. The cookbook example targets the hosted model;
the self-host transport endpoint is `localhost:8080`.

That client is also the worked example of this model's sync contract: it stamps a
tick's three views with one `capture_time_us`, tags each frame with the proprio
JSON (embedding `capture_us` / `seq` when `state_tag_keys` announces them), picks
its carrier from `state_source`, and turns the echoed `source_capture_us` back
into the age of the observation each chunk came from — on its own clock.

### C++

The `cpp_sdk` `rldx_example` publishes the same three views + per-frame state
and receives `ActionPrediction` via `on_message` — point it at the server.

## 3. Wire shape

### RTC request

The first request establishes a plan and carries no prefix. Keep the robot in a
safe hold while it runs; logical control step 0 starts when this first plan
arrives:

```python
await reactor.send_command("request_action", {
    "request_id": 0,
    "base_plan_id": -1,
    "install_step": 0,
    "rtc_prefix_len": 0,
    "action_prefix": [],
})
```

For every later request, `base_plan_id` is the active plan. `action_prefix`
contains exactly `rtc_delay` physical-unit actions that remain scheduled while
inference runs. Concatenate each row in the handshake's `action_order` and use
the dimensions in `action_dims`:

```python
await reactor.send_command("request_action", {
    "request_id": 1,
    "base_plan_id": 0,
    "install_step": current_control_step + rtc_delay,
    "rtc_prefix_len": rtc_delay,
    "action_prefix": scheduled_prefix,
})
```

Only one request may be in flight. For later requests, if a result arrives after
its install step, do not splice it into the active plan; send `reset`, hold the
robot, and start again with a cold request.

### Action response

`action_prediction` messages arrive as:

```json
{"type": "action_prediction",
 "data": {"end_effector_position": [[...],...16],   // [16,3]
          "end_effector_rotation": [[...],...16],    // [16,3]
          "gripper_close": [[...],...16],            // [16,1]
          "base_motion": [[...],...16],              // [16,4]
          "control_mode": [[...],...16],             // [16,1]
          "step": 0,
          "source_capture_us": 1755729600123456,     // echoed from the state tag
          "source_seq": 41,                          // echoed from the state tag
          "view_skew_us": 640,                       // capture spread, 3 views
          "request_id": 1,
          "plan_id": 1,
          "base_plan_id": 0,
          "install_step": 43,
          "rtc_prefix_len": 1}}
```

`source_capture_us` / `source_seq` are the `capture_us` / `seq` the client
embedded in the state tag this chunk was inferred from (null if it embedded
neither), so a chunk lands on the client's own timeline instead of on the
arrival time of the message.

For RTC, continue the base plan through `install_step`, then execute the
returned action chunk beginning at `rtc_prefix_len`. The identifiers are null
in ordinary streaming mode.

## Troubleshooting

- **`no kernel image` / CUDA arch errors** — this bundle targets the RTX PRO
  6000 (sm_120) and builds flash-attn from source for it. Confirm the NVIDIA
  driver supports CUDA 12.9 and that the container can see the GPU.
- **flash-attn build from source** — if no prebuilt wheel matches the image's
  Python tag/ABI, the Dockerfile falls back to a (slow) source build; the CUDA
  *devel* base provides nvcc + ninja for it.
- **No `action_prediction` messages** — confirm the client publishes all three
  views. If `inference_trigger` is `client_request`, also send `request_action`.
- **`request_action` command error** — reset and restart the plan chain. A later
  request must name the last returned `plan_id` and provide exactly `rtc_delay`
  finite action-prefix rows in the handshake's action order.
- **Port busy** — set `REACTOR_HTTP_PORT` and restart.
