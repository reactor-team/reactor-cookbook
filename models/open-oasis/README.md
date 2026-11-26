# Play Open Oasis through Reactor

Explore a Minecraft-style world with the
[Open Oasis 500M model](https://github.com/etched-ai/open-oasis).
Select an image or a short video, then use keyboard, mouse-button and camera
controls to guide the generated view.

## Prerequisites

- [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
  NVIDIA driver and NVIDIA Container Toolkit.
- One NVIDIA B200 GPU and storage for model weights and the serving image.
- Review the [upstream source](https://github.com/etched-ai/open-oasis), and
  accept the terms on the [model card](https://huggingface.co/Etched/oasis-500m)
  with your Hugging Face account. The repository is gated.

## Run

The YAML manifest declares the image build, Runtime and recording
configuration. The image build fetches the pinned upstream source. Build the
weights folder once, with a Hugging Face token that can read the gated
repository. It downloads the two checkpoints that `load()` reads, about
3.4 GB, at the pinned revision:

```sh
cd models/open-oasis
pip install huggingface-hub pyyaml
HF_TOKEN=<your token> python prepare_weights.py --output ./weights
```

Then build, run and deploy. `runtime.weights_path` points `reactor run` at the
same folder, and the deploy uploads it, so a new instance does not download
the checkpoints:

```sh
reactor build
reactor run --gpus device=0
reactor deploy production --weights ./weights
```

Nothing downloads at runtime (`HF_HUB_OFFLINE=1`). A missing checkpoint stops
`load()` with an error that names the command that builds the folder.

The default endpoint is `http://localhost:8080`. Use `--port` to choose
another port. Connect after the service is available. Rebuild after changing
code or dependencies.

## Inputs and controls

The session starts without a selected image.

- `set_image` selects an uploaded starting image.
- `set_video` selects up to 32 consecutive starting frames from an uploaded
  video, beginning at the requested frame offset.
- `random_scene` explicitly selects the upstream sample scene.
- `set_key_state` and `set_mouse_button_state` hold or release the supported controls.
- `mouse_move` supplies pitch/yaw movement for the next generated frame.
- `release_controls` releases held controls and pending movement.
- `reset` restarts the selected scene, optionally with another seed.

Use the generated schema for accepted key names, button names and upload
limits. Starting images and video frames are resized to 640×360.
A brief press and release is retained until a frame successfully uses it.
Camera movement returns to zero after a successful frame, with an updated
state broadcast. Failed generation does not consume pending controls.

A viewer disconnect releases controls while preserving the shared world.
Ending the session clears its selections and progress.

## Output and messages

`main_video` emits one new 640×360 RGB frame per step without audio.
Playback follows generation throughput. The manifest enables video recording.

Command replies include `action_changed`, `conditioning_changed` and
`rollout_reset`; read each reply from the awaited
command. `state_update` broadcasts controls, seed and selected input so that
all connected viewers can display the shared state.

## Sample and tests

The sample image is distributed by the pinned upstream repository under its
MIT license. Attribution and source information accompany it in
`example_images/`. It is selected only by `random_scene`.

```sh
cd models/open-oasis  # From the repository root.
PYTHONPATH=. python -m pytest tests/ -q
```

CPU tests cover shared-session disconnects, successful/failed control
consumption, bounded video decoding and the application/model contract.
