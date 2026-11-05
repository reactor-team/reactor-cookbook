# SolarWM through Reactor Runtime

Generate an interactive video world from an uploaded image, a prompt and
six-axis camera motion with [SolarWM](https://github.com/Junchao-cs/SolarWM).
This recipe serves the Wan2.2 TI2V-5B Stage2 checkpoint on one B200.

## Finite camera paths

After `set_image`, upload JSON with `poses`, an array of OpenCV camera-to-world
matrices, and pass the uploaded file to `set_camera_path(trajectory=...)`.
Provide exactly `12K-3` RGB-frame poses for `K` chunks (up to 320 chunks).
The first pose anchors the relative path; output contains 9 frames in the
first chunk and 12 in each later chunk. Generation stops at the path end.
`reset` replays the selected path and `set_image` clears it. Held camera-axis
commands are unavailable while a path is selected.

## Run

Requires the Reactor CLI, Docker, NVIDIA Container Toolkit, sufficient model
storage, and access to the gated
[junchaoh-cs/SolarWM](https://huggingface.co/junchaoh-cs/SolarWM) repository.

```sh
cd models/solarwm-stage2
reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

The build installs one pinned upstream checkout and Runtime 3.10. Startup
validates that checkout and downloads missing checkpoint files to the mounted
weights directory. Use `--weights /path/to/weights` to select storage and
`--port` to change the default HTTP port 8080.

## Inputs and controls

Generation waits for an uploaded image; there is no default image.

- `set_image(image, prompt)`: select a JPEG, PNG, WebP or BMP image up to
  25 MiB and 100 million pixels. It is resized and center-cropped to 864×480
  exactly as upstream `decode_video` prepares a first frame (torch bilinear
  resize without antialiasing, bfloat16 pixels).
  A blank prompt retains the active prompt or uses the configured neutral
  description when starting the first world.
- `set_prompt(prompt)`: restart from the selected image with a new nonempty
  description.
- `set_forward`, `set_strafe`, `set_vertical`: hold camera translation.
- `set_pitch`, `set_yaw`, `set_roll`: hold camera rotation.
- `release_camera()`: return every axis to zero.
- `reset(seed)`: restart from the selected image and prompt. A seed of -1
  retains the current seed.

Camera values range from -1 to 1. Positive pitch looks upward; negative pitch
looks downward. Camera commands apply from the next chunk and remain held.
Changing the image or prompt starts a new world with neutral controls.

## Output and messages

The first chunk contains 9 RGB frames, followed by 12 frames per chunk on
`main_video`. Playback follows measured generation time. No audio is produced.

Generation follows the upstream Stage2 sampler: the seed's generator first
draws the initial noise of the whole rollout (the camera path's chunks, or the
rollout limit for held controls), then each chunk's re-noising. A camera path
therefore reproduces upstream's output for the same image, prompt, seed and
poses. Frames are rounded to uint8 as upstream does before encoding.

Command replies are `image_selected`, `prompt_queued`,
`camera_motion_changed` and `rollout_reset_queued`. Shared `state_update`
messages report the selected inputs, six held axes, completed chunks, next
chunk size and most recent generation time.

The default rollout limit is 320 chunks. At the limit,
`rollout_limit_reached` is emitted and generation waits for a reset or new
image. A viewer disconnect releases controls while keeping the shared world.
Ending the session releases the world while retaining loaded weights.

## Deployment and recording

`reactor.yaml` defines the complete image build and recording of
`main_video`. The image contains a fixed upstream source revision; the
persistent weights mount holds model assets and working data.

Remote clients require reachable WebRTC media transport as well as HTTP
signalling. An SSH HTTP-port forward alone does not carry UDP media.

Source and checkpoint revisions are pinned in `solarwm_stage2.yaml`. Consult
the upstream repository and model card for their usage terms.


## Tests

From the repository root, run the CPU contract tests without model weights:

```sh
cd models/solarwm-stage2
PYTHONPATH=. python -m pytest tests/ -q
```
