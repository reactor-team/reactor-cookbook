# DreamZero-YAM robot bridge

Use this example to connect `dreamzero-yam-molmoact2` to a bimanual YAM
robot. It streams three camera views and the measured state of both arms, then
receives `(24, 14)` action chunks.

The script runs with placeholder cameras and state, so try it before adding
robot hardware. To integrate your robot, replace three methods:

- `CameraTrack.read_frame`
- `RobotInterface.get_state`
- `RobotInterface.execute_chunk`

## Run it

```sh
cd notebooks
uv sync --python 3.12

export REACTOR_API_KEY='<your key>'
uv run python dreamzero_yam_bridge.py
```

You should see `CONNECTING → WAITING → READY`, followed by
`prompt_accepted`, `episode_started`, and a stream of chunk messages. Press
Ctrl-C to stop. HTTP 429 `no available capacity` means no capacity is free;
wait and retry.

## Connect your robot

- `read_frame` returns the latest `uint8` RGB frame for each view. Keep the
  `top`, `left`, and `right` camera roles consistent.
- `get_state` returns six joint positions and one gripper position for each
  arm. Joint positions are radians; grippers use 0 for open and 1 for closed.
- `execute_chunk` replaces the pending action buffer with the newest chunk
  and sends it to your controller at its control rate.

Do not send the raw quickstart output directly to hardware. Add joint limits,
per-step motion limits, stale-chunk rejection, and a watchdog/e-stop first.

## Wire contract

The model is free-running and closed-loop. It uses:

- video tracks named `top`, `left`, and `right`;
- `set_prompt` once per episode;
- `set_left_joint_pos`, `set_left_gripper_pos`,
  `set_right_joint_pos`, and `set_right_gripper_pos` continuously;
- `action_chunk` replies containing `chunk_index`, `obs_seq`,
  `inference_seconds`, and `actions`.

Each action row has 14 values:

```text
[left joints × 6, left gripper, right joints × 6, right gripper]
```

## Control-loop rules

1. **Keep state flowing.** The example sends state at 10 Hz. If updates stop,
   the model eventually waits for fresh state.
2. **Send both arms.** An arm whose state is never streamed gets relative
   deltas from a zero default. Do not execute them as absolute targets.
3. **Replace, do not append.** A new chunk supersedes the pending chunk.
   Blend or rate-limit the seam before sending it to a stiff controller.

Always close the session when finished; the example does this in `finally`.
For the shared session lifecycle, see the
[robot policy client contract](./robot-policy-client-contract.md).
