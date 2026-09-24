# Cosmos3-Policy-DROID

NVIDIA's [Cosmos 3](https://github.com/NVIDIA/cosmos) robot manipulation
policy for the DROID platform (a Franka Panda arm with a Robotiq 2F-85
gripper), served as a Reactor model. The robot's controller publishes three
camera views and its joint state; the model answers with a chunk of 32 future
joint-position targets to execute at 15 Hz. It produces no video.

Two post-trained checkpoints share one serving path and one config file:

| Checkpoint | Size | Warm step on one B200 |
| --- | --- | --- |
| [`nvidia/Cosmos3-Edge-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) (default) | 4B, ~7 GB | ~210 ms per `[32, 8]` chunk |
| [`nvidia/Cosmos3-Nano-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) | 16B, ~35 GB | ~570 ms per `[32, 8]` chunk |

A 32-step chunk at 15 Hz is 2.13 s of motion, so either checkpoint predicts
well inside the budget of the chunk it replaces.

## How it works

Every prediction is a pure function of the newest frame on each camera view,
the robot's current proprioception, and the task string. The policy carries
nothing between predictions: the wrist view is composed above the two exterior
views into one 540x640 canvas, the current joint state becomes the first row
of a 33-row action tensor, and the model denoises the 32 rows after it in four
UniPC steps with classifier-free guidance 3.0. The result is absolute joint
positions plus a gripper command.

The client drives the loop. After the first chunk, the model predicts again
only when the client's `executed_step_json` echoes a strictly larger `step`
than the last one it acted on, so a controller still executing a chunk is
never run ahead of, and a stalled controller repeating an echo cannot trigger
a second prediction.

## Client contract

Inbound video tracks, named after the DROID training-time cameras:

| Track | View |
| --- | --- |
| `wrist_view` | Wrist-mounted camera |
| `exterior_view_1` | First exterior camera |
| `exterior_view_2` | Second exterior camera |

Any resolution works; the model resizes. Keep publishing the current
observation at a steady rate so every view has a recent frame when a
prediction is due.

Commands:

| Command | Purpose |
| --- | --- |
| `set_task_description` | The episode's language instruction (300 chars max). Takes effect on the next chunk. |
| `set_proprio_json` | `{"joint_position": [[<7 floats>], ...], "gripper_position": [[<float>], ...]}`; the last row is the current state. Refreshed each control step. A malformed or non-finite value is ignored, never zero-filled. |
| `set_executed_step_json` | `{"step": <int>, "action": [[...]]}` echoing the chunk just executed. The next chunk is predicted only once `step` strictly increases. |
| `reset` | Reopen the gate: the next chunk is `step` 0 again and needs no echo. |

Message, on the data channel:

| Message | Fields |
| --- | --- |
| `action_prediction` | `action`: `[32][8]`, 7 joint positions plus 1 gripper command per row; `step`: monotonic counter for the session |

A typical control loop: publish the three tracks, set the task, send the
current proprio, receive `action_prediction`, execute the chunk (or the
leading part of it), send fresh proprio and the echo `{"step": <received
step>, ...}`, receive the next chunk.

## Prerequisites

- One NVIDIA GPU with at least 24 GB of memory for Edge (80 GB for Nano),
  Hopper or Blackwell class for the prebuilt attention kernels.
- Docker with the NVIDIA container toolkit and the `reactor` CLI.
- Network access to Hugging Face on first load. The checkpoint and the
  `Wan-AI/Wan2.2-TI2V-5B` video tokenizer it depends on download into
  `runtime.weights_path` (`~/.cache/reactor_registry/cosmos3-policy-droid`)
  and are reused after that. No token is needed; the repositories are public.

## Run it

```sh
cd models/cosmos3-policy-droid
reactor build
reactor run --gpus device=0
```

The first load downloads the weights and runs two warmup predictions so
`torch.compile`'s first call (about 20 s) happens before the first session.
Drive it with `reactor-sdk`: publish the three tracks, then loop on
`set_proprio_json` / `set_executed_step_json` and read `action_prediction`
from `on_message`.

To serve Nano, edit `cosmos3_policy_droid.yaml`:

```yaml
checkpoint: nvidia/Cosmos3-Nano-Policy-DROID
format_prompt_as_json: null
guidance_interval: null
```

## Files

| File | Owns |
| --- | --- |
| `cosmos3_policy_droid.py` | The application half: tracks, the two gates, the `reset` command, the `action_prediction` message |
| `cosmos3_policy_droid_model.py` | The model half: the policy service behind `load()` / `generate()` / `reset()`, `PolicyInput`, `PolicyResult` |
| `cosmos3_policy_droid_types.py` | `PolicyMedia`, `PolicyState`, `ActionPrediction` |
| `cosmos3_policy_droid_assets.py` | Config parsing and Hugging Face checkpoint routing |
| `cosmos3_policy_droid.yaml` | Which checkpoint to serve and how to sample it |
| `cosmos_framework/` | Pruned upstream serving code, OpenMDW-1.1, byte-identical to upstream (`VENDOR.md`) |
| `PORTING.md` | The decisions behind the two-halves shape |
| `tests/` | The contract, every refusal, the gates, and the model half's bookkeeping, without a GPU |

## Notes

- Upstream's content guardrail is switched off at load: its checkpoint is
  approval-gated and its downloader shells out to a tool the image does not
  carry. Moderation is the deployment's concern.
- `COSMOS_TRAINING=0` in the image keeps the vendored framework's
  training-only imports off the serving path; the vendored file set was
  measured with it set.
- The vendored tree pins Python 3.13 and CUDA 12.8 because the prebuilt
  `flash-attn`, `flash-attn-3-nv`, and `natten` wheels exist for exactly
  that pair.
