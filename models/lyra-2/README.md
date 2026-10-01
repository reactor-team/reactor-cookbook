# Play Lyra 2.0 through Reactor

Explore an image-conditioned video world with
[Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2).
Change the scene description and control six-axis camera motion while the
world continues. This recipe uses the released four-step video model.

## Prerequisites

- [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
  NVIDIA driver and NVIDIA Container Toolkit.
- One NVIDIA B200 GPU and persistent checkpoint storage.
- The public Lyra 2.0 checkpoints, arranged under
  `source/Lyra-2/checkpoints/` in the weights directory according to the
  upstream instructions. Review the upstream source and model usage terms.
- The image's upstream source directory must allow creation of its checkpoint
  symlink at startup. With a read-only filesystem, provision that symlink to
  the selected checkpoint mount before starting the service.

## Run

The YAML manifest declares the serving-image build and Runtime configuration.

```sh
cd models/lyra-2
reactor build
reactor run --gpus device=0 --weights /path/to/lyra-weights
```

The default endpoint is `http://localhost:8080`. Select another endpoint port
with `--port`. Connect after the service is available; rebuild after code or
dependency changes.

## Inputs and controls

A new session waits for an image selection. The initial image preparation
precedes video playback.

| Command | Effect |
| --- | --- |
| `set_image(image, prompt, seed)` | Start a new world from JPEG, PNG, WebP or BMP, up to 25 MiB and 100 million pixels. The image is resized to 768×448. Blank text uses the default continuation prompt; seed -1 keeps the current seed. |
| `random_image()` | Select an upstream sample image and its paired description. |
| `set_prompt(prompt)` | Change the description for the next video segment. Requires a selected image. |
| `set_camera_motion(forward, strafe, vertical, pitch, yaw, roll)` | Set all six held camera axes together, after selecting an image. |
| `set_forward`, `set_strafe`, `set_vertical`, `set_pitch`, `set_yaw`, `set_roll` | Set an individual held axis, including before image selection. |
| `release_camera()` | Stop all camera movement. Requires a selected image. |
| `reset(seed)` | Restart from the selected image and current prompt; progress immediately returns to zero. |

Axis values range from -1 to 1; zero stops movement. Forward/strafe/vertical
control translation, and pitch/yaw/roll control rotation. Changes take effect
on the next video segment. Selecting an image or resetting clears all axes.
A viewer disconnect releases camera controls and preserves the shared world.

## Output and messages

`main_video` contains 768×448 RGB video, 80 frames per segment, without audio.
Playback follows generation throughput. `chunk_completed` reports generation
progress; delivery and playback can finish later.

Commands return `image_selected`, `prompt_queued`, `camera_changed` or
`reset_queued`. Read command replies from the awaited call.
`state_update` broadcasts the selected image, queued and most recently used
prompt, seed, completed segments and camera controls.

Ending the session clears its selections and progress. The manifest records
`main_video` by default.

## Tests

```sh
cd models/lyra-2  # From the repository root.
PYTHONPATH=. python -m pytest tests/ -q
```

CPU contract tests cover input preparation, resets, session isolation, camera
continuity, rejected steps and failed generation without advancing progress.
