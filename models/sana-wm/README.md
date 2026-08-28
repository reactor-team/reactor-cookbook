# Run SANA-WM through Reactor Runtime

Run NVIDIA's public [SANA-WM](https://github.com/NVlabs/SANA) streaming world
model as an image-, prompt-, and camera-controlled Reactor backend. The recipe
targets the distilled `Efficient-Large-Model/SANA-WM_streaming` release and
emits one 24-frame autoregressive video chunk per model turn.

The adapter loads an exact tested SANA revision from Reactor's persistent
weights cache and calls its streaming inference components directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  intentionally unsupported; the manifest declares one NVIDIA B200.
- About 110 GB in Reactor's weights cache for the pinned SANA-WM, Gemma, and
  Pi3X assets, plus Docker image space for CUDA and the inference dependencies.

## Run

This directory is a `reactor` workspace. `reactor.yaml` controls its Reactor
Runtime 3.2.5, CUDA 12.8, Python 3.12, system packages, and Python dependencies.
`requirements.txt` contains the model's inference dependencies. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Validate the workspace, build the image, and expose one GPU to the container:

```sh
cd models/sana-wm
reactor validate
reactor build
reactor run --gpus device=0
```

`reactor run` reuses the image produced by `reactor build` and builds it
automatically when the local image is absent. The first run clones the pinned
SANA source and downloads the pinned model snapshots beneath the CLI-mounted
weights cache. Interrupted Hugging Face downloads resume, and later runs reuse
the completed resources.

Rebuild after changing code, dependencies, or the manifest build definition:

```sh
reactor build && reactor run --gpus device=0
```

The Runtime reports ready only after the Stage-1 DiT, causal VAE, refiner,
Gemma encoders, and their CUDA tensors are loaded. It serves on port 8080 by
default; pass `--port` to change both the host mapping and Runtime port:

```sh
reactor run --gpus device=0 --port 18080
curl -s localhost:18080/health
curl -s localhost:18080/schema
```

## Large local storage

`runtime.weights_path` defaults to
`~/.cache/reactor_registry/sana-wm-streaming-reactor`. Point that path at a
large volume before the first run when the system disk is small:

```sh
export SANA_WORK=/path/to/large-volume/sana-wm
mkdir -p "$SANA_WORK" "$HOME/.cache/reactor_registry"
ln -s "$SANA_WORK" "$HOME/.cache/reactor_registry/sana-wm-streaming-reactor"
```

Skip the `ln` command when the destination already exists. Reactor's weights
mount retains the upstream source and model assets across image rebuilds.
The container engine stores image layers and build cache separately, so
configure that storage on the large volume when the system disk is small.

## Controls

- `set_image(image, prompt, intrinsics)` selects an uploaded JPEG, PNG, WebP,
  or BMP first frame. The prompt and NumPy intrinsics are optional.
- `random_image` selects another pinned public SANA-WM example with its prompt
  and calibration.
- `set_prompt(prompt)` applies new non-empty scene text by starting a fresh
  rollout from the selected first frame.
- `set_control(control, pressed)` holds or releases `forward`, `back`,
  `strafe_left`, `strafe_right`, `yaw_left`, `yaw_right`, `pitch_up`, or
  `pitch_down`. Multiple controls can remain held together.
- `release_controls` releases every held live camera control.
- `set_camera_trajectory(trajectory)` selects an uploaded NumPy camera-to-world
  trajectory shaped `(F, 4, 4)` for exact upstream-compatible replay.
- `use_interactive_controls` leaves finite trajectory playback and returns to
  held live controls on a fresh rollout.
- `reset(seed)` clears the incremental caches and optionally changes the seed.

A session starts without a selected image. `set_image` and
`random_image` start continuous generation from the selected first frame.

When uploaded intrinsics are omitted, the pinned public Pi3X model estimates
camera calibration through the same path used by SANA-WM. Arbitrary images are
accepted, but images near the model's training distribution produce the most
stable rollouts.

## Autoregressive inference

One turn runs upstream's four-step self-forcing Stage 1, one three-latent
refiner block, and one causal-VAE decode, then emits 24 RGB frames at 1280x704.
Playback adapts to measured inference throughput, and the output queue holds
one complete 24-frame chunk. The adapter constructs
`SelfForcingFlowEulerCamCtrl.sample_chunks` once per rollout and advances that
incremental runner across Reactor turns.

The upstream state remains incremental across Reactor turns:

- Stage 1 retains its two configured cached blocks.
- `RefinerChunkRunner` retains the released 11-frame sliding KV window.
- `CausalVaeStreamingDecoder` retains its per-layer causal feature cache.

The defaults in `sana_wm.yaml` preserve those released memory lengths. Prompt
changes start a fresh rollout because upstream encodes text when the caches are
initialized. The distilled release uses classifier-free guidance scale 1 with
its fixed empty negative prompt.

Each rollout is bounded at 512 chunks. Reaching the bound broadcasts a reset,
reinitializes the caches from the selected image, prompt, and seed, and
continues without reloading model weights. This bounds trajectory and latent
storage while allowing the Runtime process to remain long-lived.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `image_selected` identifies the uploaded or built-in image, effective prompt,
  and calibration source.
- `prompt_changed`, `control_changed`, `controls_released`,
  and `trajectory_selected` confirm their resulting state and
  affected chunk.
- `rollout_reset_queued` identifies the fresh rollout.
- `trajectory_exhausted` reports the end of finite trajectory playback.
- `state_update` is a complete snapshot of image, calibration, prompt, camera
  mode, held controls, seed, generation state, and chunk position.
  A joining viewer receives one immediately, and successful state changes
  broadcast another.

Message delivery remains outside the synchronous inference loop.

## Recording

`reactor.yaml` records `main_video` by default in four-second H.264 chunks and
allows clips up to five minutes. The manifest-generated image includes FFmpeg.
SANA-WM does not emit audio.

## Notes

- `sana_wm.yaml` pins the public SANA source, SANA-WM checkpoint, Gemma text
  encoder, Pi3X source and checkpoint, inference memory lengths, camera motion,
  and five built-in scenes.
- Set `SANA_WM_SOURCE_PATH` with
  `reactor run -e SANA_WM_SOURCE_PATH=/path/in/container` to reuse a clean
  checkout already visible inside the weights mount. The adapter imports that
  checkout directly.
- Diffusers 0.38 is intentional because the released refiner accesses LTX-2.3
  attention-gating fields introduced after 0.37. Inference uses the released
  checkpoint precision.
- The adapter follows this cookbook's Apache-2.0 license. SANA source, SANA-WM
  checkpoints, Gemma, LTX-2 components, and Pi3X retain their upstream licenses
  and terms.
- Ending a session releases its controls, uploaded conditioning, and
  per-rollout caches. Stop `reactor run` to remove the container and release
  its GPU memory.
