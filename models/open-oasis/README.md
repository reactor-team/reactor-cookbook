# Play Open-Oasis through Reactor Runtime

Run the public [Open-Oasis](https://github.com/etched-ai/open-oasis) 500M
autoregressive Minecraft model as an interactive Reactor backend. A client can
start from an uploaded image, consecutive frames from an uploaded video, or the
built-in upstream scene; hold native keyboard and mouse controls; apply camera
motion; stream generated video; and record the session.

The adapter checks out an exact tested Open-Oasis revision in the generated
model image and calls its released sampler directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  unsupported. The recipe requests one B200, although the 500M checkpoint can
  share a sufficiently large GPU with other services.
- About 5 GB in Reactor's persistent weights cache for the model and VAE
  checkpoints, plus Docker image and build-cache space for CUDA dependencies.

## Run

This directory is a `reactor` workspace. `reactor.yaml` declares the model,
Reactor Runtime, CUDA and Python versions, system packages, Python dependencies,
recording, persistent weights directory, and the pinned upstream checkout. The
CLI generates the complete serving image from that manifest; this recipe does
not contain or require a Dockerfile. See Reactor's
[build configuration](https://docs.reactor.inc/platform/build) for the
supported YAML fields.

Validate the workspace, build the generated image, and expose one GPU to the
container:

```sh
cd models/open-oasis
reactor validate
reactor build
reactor run --gpus device=0
```

`--gpus device=4` selects host GPU 4 and presents it as device 0 inside the
container. `reactor run` reuses the image from `reactor build` and builds it
automatically when the local tag is missing. Run `reactor build` again after
changing adapter code, dependencies, or `reactor.yaml`.

The first startup downloads the pinned model and VAE checkpoints. Later starts
reuse them from the CLI-mounted weights cache. Set `HF_KEY` when the Hugging Face
environment requires authentication and forward it without placing its value on
the command line:

```sh
export HF_KEY=hf_your_read_token
reactor run --gpus device=0 -e HF_KEY
```

The default endpoint is `http://localhost:8080`. Pass `--port` to use another
port:

```sh
reactor run --gpus device=0 --port 18098 -e HF_KEY
```

Connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) using
**Local (Direct)**, or check readiness and the generated contract directly:

```sh
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/schema
```

## Controls

Generation waits for `set_image`, `set_video`, or `random_scene`; connecting
never selects a default image.

- `set_key_state(key, pressed)` holds or releases `w`, `a`, `s`, `d`, `space`,
  `shift`, `ctrl`, `e`, `escape`, `f`, `q`, or hotbar keys `1`-`9`.
- `set_mouse_button_state(button, pressed)` holds or releases `left`, `right`,
  or `middle`.
- `mouse_move(camera_x, camera_y)` adds normalized camera movement for the next
  generated frame.
- `release_controls` releases every held key and mouse button and discards
  queued camera movement.
- `set_image(image)` starts a fresh rollout from one uploaded Minecraft frame.
- `set_video(video, offset, prompt_frames)` starts from 1-32 consecutive frames
  of an uploaded video.
- `random_scene` explicitly selects the official upstream sample image.
- `reset(seed)` restarts the selected context and optionally replaces the seed.

Keyboard keys and mouse buttons remain held until released. Camera movement is
consumed after one generated frame. A press followed by a release before the
next inference boundary is retained as a one-frame pulse, so short browser input
is not lost while inference is running. Resetting or selecting new visual
context clears all controls.

## Starting context

`set_image` uses Reactor's upload protocol and accepts a decodable `image/*`
file. EXIF orientation is applied before the image is resized to Open-Oasis's
native 640x360 frame shape.

`set_video` accepts a decodable `video/*` upload and selects consecutive source
frames beginning at `offset`. The upload must contain at least
`offset + prompt_frames` frames. Supplying multiple frames preserves the model's
released video-conditioning path and gives the initial causal window motion
history that a single image cannot provide.

Open-Oasis has no text encoder or prompt-conditioned inference path. The Reactor
contract therefore exposes image and video uploads but does not invent a text
prompt input. Uploaded context is discarded when the viewer disconnects, so a
new connection waits for its own upload or an explicit `random_scene` command.

## Runtime boundary

Open-Oasis is an autoregressive video world model. Each Reactor inference turn
samples and emits exactly one new RGB frame while retaining the generated visual
history and aligned 25-dimensional VPT action history for the next turn. The
output queue holds one complete one-frame turn, keeping command-to-frame latency
bounded.

The released 500M sampler does not expose an incremental transformer KV cache.
For fidelity, each turn uses its native bounded causal window of up to 32 latent
frames and actions. The adapter preserves the upstream DDIM schedule,
stabilization step, latent scaling, noise clamp, and VAE path instead of
substituting a different cache implementation.

Model reset and per-frame sampling use Reactor's synchronous inference boundary,
matching the other cookbook world-model adapters. Image and video decoding run
outside the asynchronous command loop so upload preparation does not delay
unrelated commands. Controls are sampled at the start of each generated frame.

## Model messages

Every accepted command returns its own typed result and broadcasts a complete
`state_update` snapshot:

- `action_changed` reports held keys, held mouse buttons, and camera movement
  queued for the next frame.
- `conditioning_changed` reports the selected context source, filename or
  built-in identifier, and accepted prompt-frame count.
- `rollout_reset` reports the selected seed and retained starting context.
- `state_update` reports the complete shared controls, seed, and context
  selection. A joining viewer receives the same snapshot immediately.

Message delivery remains outside the model's blocking inference work.

## Public source and model assets

`open_oasis.yaml` pins the upstream source and Hugging Face checkpoint
revisions. The YAML-generated image installs the source under `/opt/open-oasis`.
The model and VAE checkpoints remain outside the image in
`runtime.weights_path`; Reactor mounts that path at the same absolute location
inside the container and exports it as `REACTOR_WEIGHTS_PATH`.

The checked-in configuration stores persistent assets under
`/opt/dlami/nvme/.cache_hf/reactor_registry/open-oasis/weights`. Change
`runtime.weights_path` before building when another host volume should own the
checkpoints. Docker image layers and build cache are managed by the host Docker
daemon separately; configure Docker's data root on a large volume when the
system disk is constrained.

## Recording

`reactor.yaml` records `main_video` as H.264 in four-second chunks and allows
clips up to five minutes. The model emits video without generated audio.

## Notes

- `inference.context_frames: 32` preserves the released model's default causal
  memory window.
- `inference.ddim_steps: 10` preserves the released sampling schedule.
- Selecting the same starting context and seed reproduces the same initial
  rollout conditions; later actions determine the generated trajectory.
- Ending a session releases controls and uploaded context while retaining loaded
  model weights for the next session.
- Stop `reactor run` to remove the container and release its GPU memory.
