# YUME-1.5 through Reactor Runtime

Generate an interactive video world from text, an image, or a short video with
[YUME-1.5](https://github.com/stdstu12/YUME). Hold movement and camera-view
keys and update the scene description between generated chunks.

## Run

Requires the Reactor CLI, Docker, NVIDIA Container Toolkit and one available
B200 GPU. This workspace builds from `reactor.yaml` and serves on Runtime 3.6.

```sh
cd models/yume-1-5
reactor build
reactor run --gpus device=0
```

The first startup downloads the pinned checkpoint into the mounted weights
directory. Use `--weights /path/to/weights` to choose storage and `--port`
to change the HTTP port. Forward `HF_TOKEN` if authentication is required.

## Select a scene

Generation waits for an explicit scene command; no default image is selected.

- `set_image(image, prompt, seed)`: upload an image, up to 25 MiB and
  100 million pixels. It is resized to 1280×704. A blank prompt uses the
  neutral image-continuation description.
- `set_video_scene(video, prompt, seed)`: upload a video up to 500 MiB with
  at least 33 frames and provide a nonempty prompt. The first 33 frames
  initialize the scene.
- `set_text_scene(prompt, seed)`: begin from a nonempty description.

The default session seed is 42. A scene command's seed of -1 retains the
current seed. Selecting a scene replaces the current world and clears controls.

## Interact

- `set_key_state(key, pressed)` holds or releases WASD movement or
  `arrow_left`, `arrow_right`, `arrow_up`, `arrow_down` view controls.
  Compatible directions combine; opposing directions are rejected.
- `release_controls()` releases all keys.
- `set_prompt(prompt)` changes the nonempty description for the next chunk.
- `reset(seed)` restarts the selected scene with controls released.

These commands require a selected scene. A viewer disconnect releases
controls while preserving the shared scene.

## Output and completion

Each successful step delivers 29 new RGB frames on `main_video`. Playback
follows measured generation time. There is no generated audio.

The serving configuration ends each rollout after 10 chunks. This is an
explicit deployment limit, not an upstream maximum. The native visual history
is preserved for every chunk within the rollout. Operators can change
`inference.max_chunks` after validating capacity; longer rollouts require
more memory and computation.

Commands return `scene_queued`, `prompt_changed`, `action_changed` or
`rollout_reset_queued`. Shared `state_update` messages include the selected
scene, controls, completed count and `limit_reached`. Each generated chunk
emits `chunk_completed`; the final one is followed by
`rollout_limit_reached`. Reset or select a new scene to continue.

## Assets and recording

The build pins the public upstream source, and `yume.yaml` pins the
checkpoint revision. Consult the upstream repository and model card for their
usage terms. The manifest enables recording of `main_video`.

For remote browsers, arrange reachable WebRTC media transport in addition to
HTTP signalling; forwarding only the HTTP port over SSH does not forward UDP.


## Tests

From the repository root, run the CPU contract tests without model weights:

```sh
cd models/yume-1-5
PYTHONPATH=. python -m pytest tests/ -q
```
