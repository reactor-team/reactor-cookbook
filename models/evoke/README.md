# Run EVOKE through Reactor Runtime

Run the public [Alaya-EVOKE world model](https://github.com/AlayaLab/Evoke) as
an interactive Reactor backend. Use this recipe to start an autoregressive
world from an image, reference video, or text prompt, apply six-axis camera
motion, stream generated video, and record the session.

The adapter and upstream implementation remain separate. This directory owns
the Reactor integration and a narrow patch that exposes EVOKE's native chunk
boundary. On first load, the adapter clones the exact tested upstream revision
into Reactor's mounted weights cache, applies that patch once, and downloads
the required post-distillation inference assets.

## Prerequisites

- The [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation) and
  a running Docker daemon.
- An NVIDIA Hopper or Blackwell GPU, NVIDIA driver, and NVIDIA Container
  Toolkit. The recipe uses FlashAttention 4 and declares one B200 for deployed
  instances; CPU inference is unsupported.
- About 90 GB of persistent storage for the source checkout, Python 3.10
  worker, post-distillation checkpoint, ViGeo weights, and download caches.
- Access to the public GitHub and Hugging Face repositories on first load.

EVOKE's source and weights are Apache-2.0. ViGeo is CC-BY-NC-4.0; review that
license before commercial use.

## Run

This directory is a `reactor` workspace. `reactor.yaml` names the model and
controls its Reactor Runtime 3.2.5 image, while `requirements.txt` lists the
adapter's serving dependencies. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

```sh
cd models/evoke
reactor build
reactor run --gpus device=0
```

`reactor run` reuses the image produced by `reactor build`, or builds it
automatically when no local image exists. It bind-mounts
`runtime.weights_path`, clones the pinned EVOKE source, creates one persistent
Python 3.10 inference worker, and downloads the pinned model snapshots on first
load. Later containers reuse all of those files. The model reports ready on
`http://localhost:8080` after its weights have loaded.

Rebuild after editing anything included in the model image:

```sh
reactor build && reactor run --gpus device=0
```

Connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) using
**Local (Direct)** and `http://localhost:8080`, or point the
[JS SDK](https://docs.reactor.inc/sdk-reference/using-the-sdk) at it
with `local: true`. A quick liveness check:

```sh
curl -s localhost:8080/health
```

## Keep large files off the system disk

`runtime.weights_path` is the persistent root for the source checkout, worker,
checkpoints, uploads, and uv, Hugging Face, Torch, Triton, and CuTe DSL caches.
It defaults to `~/.cache/reactor_registry/evoke`. Point that path at a large
disk before the first run when the home filesystem is space constrained:

```sh
export EVOKE_STORAGE_ROOT=/path/to/large-disk/reactor-evoke
mkdir -p "$EVOKE_STORAGE_ROOT" "$HOME/.cache/reactor_registry"
ln -s "$EVOKE_STORAGE_ROOT" "$HOME/.cache/reactor_registry/evoke"
```

Alternatively, edit `runtime.weights_path` to an absolute large-disk path.
`reactor run` mounts it at the same path inside the container and exports
`REACTOR_WEIGHTS_PATH`, so the adapter's default caches remain on the
persistent volume.

Model storage and container image storage are independent. Configure both on a
large disk when the system disk is space constrained.

## Runtime boundary

The original `EvokePipeline` call remains alive between requests. A narrow
patch supplies each native chunk boundary with the next prompt and camera poses
and publishes the decoded chunk while preserving the released inference state:

- rolling latent history with `history_sizes: [16, 2, 1]`;
- the persistent cross-chunk VAE decode cache;
- the external camera-indexed frame and cloud bank;
- ViGeo's streaming chunk cache with six retained frames;
- the three-level sampler with one step per level and CFG disabled.

The post-distillation checkpoint was trained with unrestricted self-attention.
Its official launcher sets `restrict_self_attn=false` and
`use_kv_cache=false`, so this recipe keeps that path. Enabling the optional
transformer KV cache would change the checkpoint's trained attention behavior.
EVOKE's latent, VAE, geometric, and ViGeo caches remain active.

EVOKE requires Python 3.10 while Reactor Runtime requires Python 3.12. The
upstream pipeline therefore runs in one persistent Python 3.10 worker. Only
chunk conditions and decoded frames cross the process boundary. Checkpoints
load once in `load()`; reset clears rollout state without starting another
worker.

## Inputs and controls

The commands cover all three upstream conditioning modes:

| Mode | Reactor command | Upstream condition | Camera |
|---|---|---|---|
| i2v | `set_image` | uploaded image and optional prompt | live six-axis c2w trajectory |
| v2v | `set_reference_video` | uploaded video, pose NPZ, and optional prompt | continues from the reference pose |
| t2v | `start_text` | optional prompt | disabled, as required upstream |

- `set_forward`, `set_strafe`, and `set_vertical` hold normalized translation
  velocities in `[-1, 1]`.
- `set_pitch`, `set_yaw`, and `set_roll` hold normalized rotation velocities in
  `[-1, 1]`.
- `set_prompt` changes the text condition at the next native chunk boundary
  without clearing visual history.
- `reset` starts a fresh rollout from the current conditioning media and
  prompt without reloading weights.

The frontend owns keyboard, pointer, touch, joystick, and gamepad mappings. The
backend integrates these six axes into the absolute OpenCV camera-to-world
matrices consumed by EVOKE.

EVOKE generates nine latent frames per native turn at 384x640 and 24 FPS.
Camera-conditioned i2v and v2v warm the persistent decoder with a real prior
and emit 36 RGB frames per command. Prompt-only t2v has no prior, so its first
turn emits 33 frames and later turns emit 36. Each turn is submitted as one
frame batch; playback adapts to measured inference throughput, and the output
queue holds one complete 36-frame chunk.

## Start from an image

[`example_images`](example_images) contains a public image suitable for testing
the upload path. Upload it through the client, then invoke `set_image` with its
upload reference and an optional prompt. Arbitrary images are accepted, but
images far outside the model's distribution may produce unstable rollouts.

When user text is empty, the adapter selects this documented, scene-neutral
condition:

> The input scene continues faithfully with balanced exposure, preserved
> highlight and shadow detail, stable brightness, consistent color, consistent
> lighting, and smooth temporal continuity.

The distilled checkpoint is CFG-free, so negative prompts have no effect at
`guidance_scale=1.0`. `set_prompt("")` restores the stability condition at the
next chunk boundary.

## Model messages

Every successful command returns `command_applied` to its requester and
broadcasts `state_update`. The complete state snapshot includes conditioning
mode and filenames, prompt, seed, completed and next chunk, and all six
camera axes. A newly connected viewer and every completed chunk receive the
same snapshot.

After 512 chunks, the adapter starts a fresh rollout from the active condition
and emits `rollout_restarted`. This bounds the preallocated pose timeline while
keeping checkpoints and the worker resident.

## Public source and model assets

`evoke.yaml` pins the source and both Hugging Face snapshots by full immutable
revision. The adapter can prepare them automatically. To create the source
checkout explicitly before first startup, clone it beneath the configured
weights path:

```sh
git clone https://github.com/AlayaLab/Evoke.git \
  "$HOME/.cache/reactor_registry/evoke/Evoke"
git -C "$HOME/.cache/reactor_registry/evoke/Evoke" checkout \
  74d268516d95c8fceadd2378f91a73f9f187042b
```

Startup verifies the checkout revision and applies `stateful_rollout.patch`
once. The required snapshots are:

- `AlayaLab/Evoke/evoke-base`;
- `AlayaLab/Evoke/evoke/stage3_post_distillation`;
- `pkqbajng/ViGeo/vigeo.pt`.

Interrupted Hugging Face downloads resume.

## Recording

`reactor.yaml` records `main_video` by default in six-second segments and
allows clips up to five minutes. EVOKE does not emit audio.

## Notes

- The distilled checkpoint was trained on v2v conditioning. Its released i2v
  and t2v paths are zero-shot, while v2v is the in-distribution mode.
- The geometric bank stays at the release recipe's default horizon because
  memory-behavior evaluations depend on it.
- FlashAttention 4 supplies the varlen attention implementation needed by
  EVOKE on Hopper and Blackwell.
- Stop `reactor run` to remove the container and release its GPU memory.
