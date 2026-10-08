# YAM (bimanual)

The [I2RT YAM](https://i2rt.com/products/yam-box) setup used by these policies
is two 6-joint arms with grippers and three cameras. This
page describes the robot as a Reactor client sees it: the camera views, the
state you send, the actions you get back, and what you must build to drive a
real robot from them.

## Policies

| Policy | Folder | Space | Protocol | Chunk |
| --- | --- | --- | --- | --- |
| `reactor/rho-yam-box` (Microsoft Rho) | [`rho-yam-box/`](./rho-yam-box) | end-effector pose, absolute | one request, one reply | `(50, 20)`, execute 25 at 15 Hz |
| `dreamzero-yam-molmoact2` | [DreamZero-YAM bridge](../../sim/notebooks/dreamzero_yam_bridge.md) | joint positions | free-running | `(24, 14)` |

The two policies use different track names, state layouts, and gripper
conventions. A client for one does not drive the other. The rest of this page
describes the Rho layout.

## Cameras

| View | Rho track | Content |
| --- | --- | --- |
| Scene | `scene_view` | The scene camera. Upstream calls it `agentview` or `cam_scene`. |
| Left wrist | `left_wrist_view` | The camera on the left arm's wrist. |
| Right wrist | `right_wrist_view` | The camera on the right arm's wrist. |

Send RGB frames; 480 × 640 is typical. The policy accepts any frame size and
scales each frame to fit 224 × 224, keeping the aspect ratio.

## State: 20 values, left arm first

| Index | Value |
| --- | --- |
| 0–2 | left `x, y, z` |
| 3–8 | left 6D rotation |
| 9 | left gripper |
| 10–12 | right `x, y, z` |
| 13–18 | right 6D rotation |
| 19 | right gripper |

The 6D rotation is the first two columns of the end-effector rotation matrix
`R`, column by column: `[R00, R10, R20, R01, R11, R21]`. The policy turns it
back into a rotation with Gram-Schmidt, so the columns need not be exactly
orthonormal, but they must not be zero or parallel.

```python
import numpy as np

rot6d = R[:, :2].T.reshape(6)                       # matrix -> 6D
arm = np.concatenate([xyz, rot6d, [gripper]])       # 10 values for one arm
state = np.concatenate([left_arm, right_arm])       # 20 values
```

[`client.py`](./rho-yam-box/client-python/client.py) has these conversions as
`rot6d_from_matrix`, `matrix_from_rot6d`, `arm_vector`, `pack_state`, and
`split_arms`.

Units and frames are not documented upstream. From the state statistics in
the checkpoint's `stats.json`, across both arms:

- `x, y, z` span about 0.12 to 0.66, -0.31 to 0.35, and -0.01 to 0.54. They
  look like meters, probably in each arm's own base frame.
- The gripper value is the raw gripper joint value (about -1.22 to 1.47), not
  a 0 to 1 open/closed fraction.
- Left arm first follows upstream's docstrings and camera names; the
  checkpoint does not state it.

Confirm the frames, the gripper scale, and the arm order on your robot before
you close the loop.

## Actions: 50 targets, execute 25

Each reply has 50 rows in the state layout. Row k is the **absolute** target
for control step k + 1 after the observation, not a delta. The checkpoint's
`policy.json` sets `robot_hz: 15`, `chunk_size: 50`, and `n_action_steps: 25`:
execute the first 25 rows at 15 Hz (about 1.7 s), then send a new
observation. The other 25 rows are a preview. The policy does not enforce the
rate.

The public YAM BusyBox finetuning recipe in `microsoft/rhobotics` uses 30 Hz
data and 32-step chunks. A checkpoint finetuned with that recipe has
different timing; check its config before you reuse these numbers.

## Drive a real YAM

The [DreamZero-YAM i2rt bridge](../../sim/notebooks/dreamzero_yam_bridge_i2rt.py)
drives the arms through the vendor's [i2rt](https://github.com/i2rt-robotics/i2rt)
stack, which reads and commands **joint** positions. Rho works in
end-effector space, so a driver needs:

1. **Forward kinematics** to compute each arm's end-effector pose from its
   measured joints, in the frame the policy expects.
2. **Inverse kinematics** to convert each target row into joint positions.
   Reject a target that IK cannot reach.
3. **Gripper mapping** between the policy's raw value and your gripper
   command. Conventions differ between policies and drivers; check the
   direction and the range.
4. **Safety**: joint and workspace limits, a per-step motion limit, a
   startup guard that refuses a chunk whose first row is far from the
   measured pose, rejection of a stale chunk, and a watchdog or e-stop.
5. **Timing**: send the 25 rows at 15 Hz. The arm holds its last target
   while the next request is in flight.

Do not send raw policy output to hardware. The Rho client's
[`MockYam`](./rho-yam-box/client-python/main.py) shows the interface: replace
`get_state` and `execute` with your driver.

There is no YAM simulator integration in this repository. To drive a
simulated YAM, use the same interface: read the end-effector pose from the
simulator and send the targets to its controller.
