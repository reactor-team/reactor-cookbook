# Play DIAMOND CSGO through Reactor Runtime

Run the public [DIAMOND CSGO world model](https://github.com/eloialonso/diamond/tree/csgo)
as an interactive Reactor backend. Use this recipe when a client needs to start
from an official spawn or an uploaded CSGO frame, apply native keyboard and
mouse actions, stream generated video, and record the session.

The adapter and upstream implementation remain separate. This directory owns
the Reactor integration; the Docker build fetches a pinned, unmodified DIAMOND
source snapshot into the model image.

## Prerequisites

- The [Reactor CLI](https://docs.reactor.inc/deploy/overview) and a
  running Docker daemon. On macOS:

  ```sh
  brew install reactor-team/tools/reactor-cli
  brew install --cask docker
  ```

- An NVIDIA GPU with the NVIDIA Container Toolkit for real-time local play, or
  a CPU for functional but slow inference.
- About 1.4 GB of free cache space for the pinned checkpoint and spawn bundle.

The Dockerfile pins DIAMOND source revision
`851cefb497733d27f1b85c804104638765860fca` and sets `DIAMOND_PATH` inside the
image. On the first run, the adapter downloads the pinned checkpoint,
configuration, and official spawn data into Reactor's mounted weights cache;
later containers reuse those files.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model, the
`Dockerfile` builds the image, and `requirements.txt` pins Runtime 3.1.2
alongside DIAMOND's serving dependencies. The CLI builds the image with the
runtime inside and runs it — nothing to install on your host but the CLI and
Docker.

```sh
cd models/diamond-csgo-world-model
reactor build
reactor run
```

`reactor run` reuses the image `reactor build` produced (it builds one on first
run if none exists), mounts the persistent model cache, and serves WebRTC
signaling on `http://localhost:8080`. Rebuild after editing anything baked into
the image:

```sh
reactor build && reactor run
```

`diamond.yaml` selects the `fast` profile for interactive play. Change
`profile` to `higher_quality` to use more diffusion denoising steps for cleaner
generation at substantially lower throughput.

On an NVIDIA host, expose the GPU with `reactor run --gpus all`. On Apple
Silicon, build the native Linux image before running:

```sh
reactor build --platform linux/arm64
reactor run
```

Docker containers cannot access Metal/MPS, so Apple Silicon inference is
CPU-only and substantially slower than host-native MPS.

Connect a client from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/)
(pick **Local (Direct)**), or point the
[JS SDK](https://docs.reactor.inc/sdk-reference/using-the-sdk) at it
with `local: true`. A quick liveness check:

```sh
curl -s localhost:8080/health
```

## Adapter layout

`diamond.py` is the complete model entrypoint: it owns model loading, lifecycle
hooks, commands, and inference. `diamond_types.py` defines the Reactor contract,
while `diamond_support.py` contains configuration, import, image, and tensor
helpers. These files remain independent from the upstream checkout.

## Controls

- `set_spawn_image(image)` starts from an uploaded CSGO image. The adapter
  center-crops it to the native aspect ratio, builds the four conditioning
  frames DIAMOND requires, uses neutral action history, and selects human input.
- `random_scene` starts from a random official DIAMOND spawn, including its
  recorded action trajectory for replay mode.
- `set_key_state(key, pressed)` holds or releases a native CSGO key.
- `set_mouse_button_state(button, pressed)` holds or releases fire or scope.
- `mouse_move(delta_x, delta_y)` supplies native relative movement for one step.
- `set_controller(controller)` selects human input or the recorded spawn replay.
- `set_paused(paused)` stops model inference; `step` advances once while paused.
- `reset` starts from another spawn state.

Control events return an `ActionChanged` message containing the acknowledged
controller, current held keys and mouse buttons, and any mouse delta accepted by
that command. Commands that change durable controls also broadcast a
`StateUpdate` snapshot containing controller, pause, keyboard, and mouse-button
state. A newly connected viewer receives the same snapshot without reconstructing
state from earlier events. Message delivery therefore stays outside the
synchronous inference loop.

## Start from an image

Ready-to-upload CSGO frames live in [`example_images`](example_images). Upload
one through the client, then pass its upload reference to `set_spawn_image`:

```js
const image = await uploadFile(file);
await sendCommand("set_spawn_image", { image });
```

The next inference boundary resets the environment and emits the uploaded frame
before consuming the first action. This frame is also emitted while paused,
without running an expensive model step. A single image has no motion history,
so the adapter repeats it four times; arbitrary non-CSGO images are accepted but
may produce unstable results because they are outside the model's training
distribution.

Use `random_scene` when the client should choose a dataset-backed initial
condition instead. Both scene commands return `SceneChanged` with the selected
source and filename or dataset scene identifier.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes.

## Notes

- `human` applies client input. `replay` follows the selected spawn's recorded
  action trajectory; DIAMOND's CSGO checkpoint does not contain a learned
  player policy.
- Keyboard keys and mouse buttons are held until released. `mouse_move`
  contains relative deltas that are consumed by exactly one inference step.
- Generation advances at DIAMOND's native 15 FPS with a one-frame output
  buffer, so held controls do not run ahead of the displayed world.
- The frontend owns keyboard, pointer, touch, and gamepad mappings. The backend
  exposes DIAMOND's native action semantics without prescribing a control
  layout or sensitivity.
- Checkpoint loading and GPU allocation happen once during `load()`. Each
  Reactor session owns one shared world: temporary client disconnects preserve
  it, while session end discards queued scenes and control state.
