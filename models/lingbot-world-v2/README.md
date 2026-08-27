# Play LingBot-World-V2 through Reactor Runtime

Run the public [LingBot-World-V2 world model](https://github.com/Robbyant/lingbot-world-v2)
as an interactive Reactor backend. Start from an uploaded or built-in image,
change the text prompt without resetting the world, and explore with continuous
six-axis camera motion.

The adapter loads an exact tested source revision from Reactor's weights cache
and calls the upstream causal-fast inference components directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. The deployment
  manifest requests one NVIDIA B200.
- About 100 GB on a local volume for the public source and 14B `causal-fast`
  checkpoint, plus Docker image and build-cache space.

LingBot-World-V2 source and weights are released under CC BY-NC-SA 4.0 for
non-commercial use. Review the
[upstream license](https://github.com/Robbyant/lingbot-world-v2/blob/main/LICENSE.txt)
before running or redistributing the model.

## Run

This directory is a `reactor` workspace. Its `reactor.yaml` names the model,
controls its Reactor Runtime 3.2.3, CUDA and Python image, and points Runtime at
the model adapter and configuration. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Build the image, expose one GPU, and start Runtime:

```sh
cd models/lingbot-world-v2
reactor build
reactor run --gpus device=0
```

`reactor run` reuses the image and serves on `http://localhost:8080`. It builds
automatically when no local image exists. Rebuild after changing adapter code,
configuration, or dependencies. To use another host GPU and port:

```sh
reactor build && reactor run --gpus device=4 --port 18089
```

The service reports healthy only after its public assets and model weights are
ready:

```sh
curl -s localhost:8080/health
```

On first load, the adapter clones the pinned public source and downloads the
pinned Hugging Face checkpoint into the CLI-mounted `runtime.weights_path`.
Later starts verify and reuse both. The checked-in default is
`~/.cache/reactor_registry/lingbot-world-v2`; change that one manifest value to
place all model assets on a larger volume. Configure the container engine's
image and build-cache storage on that volume when the system disk is small.

`LINGBOT_WORLD_V2_PATH` and `LINGBOT_WORLD_V2_CHECKPOINT_PATH` may point to
existing copies available inside the container. A source override must be a
clean Git checkout at the configured revision; an incomplete checkpoint is
completed with a resumable Hugging Face download.

## Runtime boundary

The 14B model loads once at process startup. A session owns one causal rollout,
including its scheduler generator, causal Wan VAE state, prompt cross-attention
KV, and rolling self-attention KV. Ending the session releases rollout state
while keeping model weights resident for the next session.

Each inference turn generates one native four-latent chunk with the released
four-step sampler. Chunk one emits 13 RGB frames and later chunks emit 16.
Playback adapts to measured inference throughput, and the output queue holds
one complete 16-frame chunk so inference, WebRTC playout, and recording retain
the same complete sequence.

The adapter preserves the released 18-frame rolling self-attention window,
six-frame attention sink, clean-`x0` cache commit, and causal decoder state.
Changing the prompt replaces cross-attention KV at the next chunk boundary and
retains visual history. Attention uses the upstream PyTorch SDPA path; model
execution is eager and model loading happens once during startup.

## Controls

- `set_image(image, prompt)` starts a fresh world from an uploaded JPEG, PNG,
  WebP, or BMP. An empty prompt keeps the current prompt or uses the configured
  generic continuation prompt.
- `random_image` starts a fresh world from one of six public LingBot examples
  and its matching prompt and camera calibration.
- `set_prompt(prompt)` changes text conditioning at the next chunk boundary
  without clearing the current causal world.
- `set_camera(forward, strafe, vertical, pitch, yaw, roll)` atomically sets all
  six held axes for forthcoming chunks. Each value is in `[-1, 1]` and remains
  active until changed or released.
- `release_camera` returns all six axes to neutral.
- `reset(seed)` starts again from the selected image and prompt, optionally
  using another non-negative seed.

The frontend owns device mapping. WASD maps naturally to `forward` and
`strafe`; pointer, arrow, touch, or gamepad input can drive `yaw` and `pitch`;
additional controls can drive `vertical` and `roll`. Sending the complete
camera state in one `set_camera` call keeps diagonal movement and simultaneous
rotation on the same chunk boundary.

Prompt and camera commands accepted during inference apply to the following
chunk. An in-flight GPU chunk finishes before a reset or disconnect takes
effect. Disconnecting releases held camera motion while preserving the shared
world.

Selecting an image begins continuous playback.

## Image uploads

`set_image` declares an `UploadedFile` in Runtime's schema, allowing a
schema-driven client to render a file picker. A client reserves a session upload
slot, writes the raw bytes to its returned URL, and sends the resulting upload
reference with the command.

Uploads are limited to 25 MiB and 100 million pixels. Runtime verifies the
declared media type, actual codec, dimensions, and decodability before replacing
the current world. EXIF orientation is applied before the image is resized to
the model's 480p area. Uploaded images use the configured public 480p camera
intrinsics because they contain no calibration metadata.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `prompt_queued`, `camera_motion_changed`, `image_selected`, and
  `rollout_reset_queued` report an accepted change and the chunk where it
  takes effect.
- `chunk_completed` reports generation time, frame count, prompt, and all six
  camera axes sampled by one completed chunk.
- `rollout_limit_reached` reports that the current causal timeline is full.
- `state_update` is a complete snapshot of prompt, image, camera, seed,
  completed chunks, and the next available boundary. A joining viewer
  receives one immediately, and every successful state change broadcasts
  another.

Message delivery stays outside the synchronous inference path.

## Rollout length and recording

The public model ships a 1024-position temporal RoPE table. At four latents per
chunk, one world supports 256 chunks: about 4 minutes and 15 seconds of steady
16 FPS video after the first chunk. Reaching the limit idles generation;
`reset`, `set_image`, or `random_image` starts a fresh timeline.

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes. LingBot-World-V2 emits video without audio.
