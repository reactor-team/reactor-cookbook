# Play Zing 0.5 through Reactor Runtime

Run the public [Zing 0.5 world model](https://github.com/seedleap/zing-world-model)
as an interactive Reactor backend. Start from a text prompt, an uploaded image,
or Zing's public example image, then explore the generated world with its native
W/A/S/D movement and I/J/K/L look controls.

The adapter loads exact tested source and checkpoint revisions and calls the
released autoregressive inference components directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. The deployment
  manifest requests one NVIDIA H100 80 GB GPU; the recipe is also tested on B200.
- About 45 GB on a high-capacity local volume for the public checkpoint, source,
  generated image, and build cache.

Review the [upstream repository](https://github.com/seedleap/zing-world-model)
and [checkpoint page](https://huggingface.co/seedleap/Zing-0.5) for their current
license and use conditions before running or redistributing the model.

## Run

This directory is a `reactor` workspace. Its `reactor.yaml` declares the model,
runtime, recording, persistent weights directory, and complete CUDA-capable
image build. `requirements.txt` contains the model's Python dependencies. The
workspace uses Reactor's generated-image flow and has no handwritten
Dockerfile. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported YAML fields.

Build the image, expose one GPU, and start Runtime:

```sh
cd models/zing-0-5
reactor build
reactor run --gpus device=0
```

`reactor run` reuses the image and serves on `http://localhost:8080`. It builds
automatically when no local image exists. Rebuild after changing adapter code,
configuration, or dependencies. To use another host GPU and port:

```sh
reactor build && reactor run --gpus device=4 --port 18085
```

The service reports healthy only after the pinned source and model checkpoint
are ready:

```sh
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/schema
```

The checked-in `runtime.weights_path` is
`/opt/dlami/nvme/.cache_hf/zing-0.5`. Reactor mounts that host directory at the
same path in the container, and the adapter resolves the checkpoint, Hugging
Face cache, uploaded-image workspace, and compiled-kernel caches beneath it.
Change that manifest value when another high-capacity host volume is preferred.
Configure Docker's image and build-cache storage on that volume separately when
the system disk is small.

## Runtime boundary

Zing's weights load once at process startup. A session owns one autoregressive
world, including its random generator, prompt condition, generated latents, and
rolling attention state. Ending the session releases rollout state while
keeping model weights resident for the next session.

Each inference turn produces one native causal chunk with the released
four-step DMD sampler. A text-to-video world begins with a one-latent chunk;
later text-to-video chunks and all image-conditioned chunks advance four
latents and emit 16 new RGB frames at 1248×704. Playback adapts to measured
inference throughput, and the output queue holds one complete 16-frame chunk.

The adapter preserves Zing's released 97-position local attention window,
9-position attention sink, clean-latent cache commit, and prompt-switch path.
Changing the prompt at a chunk boundary replaces text conditioning while
retaining prior world history. Generation stays synchronous so commands and
lifecycle events take effect only between complete native chunks.

New sessions remain idle until a client calls `set_prompt`, `set_image`, or
`example_image`.

## Controls

- `set_key(key, pressed)` holds or releases `w`, `a`, `s`, `d`, `i`, `j`, `k`,
  or `l` for forthcoming chunks.
- `release_controls` returns every held movement and look control to neutral.
- `reset(seed)` starts a fresh world from the selected text or image condition
  and current prompt, optionally using another non-negative seed.

W/A/S/D control forward, left, backward, and right movement. I/J/K/L control
up, left, down, and right look direction. Keys remain held until released, and
the complete held state is sampled when the next chunk begins. A frontend can
map these commands to keyboard, pointer, touch, or gamepad controls.

An in-flight GPU chunk finishes before a control change, reset, or disconnect
takes effect. Disconnecting releases every held control while preserving the
shared world until the session ends.

## Start from text or an image

`set_prompt(prompt)` starts text-to-video generation when the session has no
world. During an active rollout, it changes the prompt for the next chunk
without discarding prior world history.

`set_image(image, prompt, seed)` starts a fresh world from an uploaded JPEG,
PNG, WebP, or BMP. The upload may be up to 25 MiB and 100 million pixels. It is
converted to RGB and resized to 1248×704. A blank prompt uses the configured
image-neutral prompt, and `-1` retains the active seed.

`example_image` is a parameterless button command that starts from Zing's
released `case0.jpg` and its matching pixel-art adventure prompt. No image is
selected automatically when a session begins.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `prompt_queued`, `image_selected`, `action_changed`, `controls_released`, and
  `rollout_reset` report an accepted change and where it takes effect.
- `chunk_completed` reports generation time, frame count, prompt, held controls,
  and retained world positions for one completed chunk.
- `state_update` is a complete snapshot of prompt, image, controls, seed,
  generation status, and completed chunks. A joining viewer receives one
  immediately, and every successful state change broadcasts another.

Message delivery stays outside the synchronous inference loop.

## Rollout length and recording

`zing.yaml` limits one continuous world to 32 chunks. Reaching the configured
bound automatically starts a fresh world from the same selected condition,
prompt, and seed. Calling `reset`, `set_image`, or an initial `set_prompt` also
starts a fresh timeline.

`reactor.yaml` records `main_video` by default. Zing 0.5 emits video without
audio.

## Notes

- `zing.yaml` pins the upstream source, checkpoint revision, output resolution,
  seed, rollout length, and released attention-window settings.
- The generated image installs the pinned upstream source and model Python
  dependencies; model weights remain outside the image in
  `runtime.weights_path`.
- Selecting the same condition, prompt, controls, and seed reproduces the same
  sampling inputs.
- Stop `reactor run` to remove its container and release GPU memory.
