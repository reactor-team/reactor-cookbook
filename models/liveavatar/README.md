# LiveAvatar through Reactor Runtime

Generate a speaking-avatar video from an uploaded reference image and speech
audio with [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) and Reactor.
Add a scene or performance prompt, optionally supply a prepared pose sequence,
and stream the resulting video with the uploaded speech.

The recipe serves LiveAvatar's four-step Turbo profile on three NVIDIA B200
GPUs. A session waits for your inputs and an explicit `start` command.

## Prerequisites

- The [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation),
  Docker, an NVIDIA driver and NVIDIA Container Toolkit.
- Three available NVIDIA B200 GPUs.
- A high-capacity volume for the base checkpoint, LiveAvatar checkpoint,
  image layers and persistent runtime data.

## Run

This directory is a `reactor` workspace. Its `reactor.yaml` declares the model,
runtime, GPU resources and complete serving-image build. See the [build configuration
guide](https://docs.reactor.inc/deploy/platform/build).

Prepare a weights directory containing the base checkpoint in `wan2_2/` and
the LiveAvatar checkpoint in `liveavatar_lora/`, using the model assets linked
below. Replace `/path/to/liveavatar-weights` with that directory:

```sh
cd models/liveavatar
reactor build
reactor run --gpus '"device=0,1,2"' \
  --weights /path/to/liveavatar-weights
```

`reactor run` serves at `http://localhost:8080` by default and reuses the built
image and mounted weights. Rebuild after changing code, configuration or
dependencies. Adjust `--gpus` to select available devices and add `--port` to
choose a different HTTP port. Check readiness and the generated contract:

```sh
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/schema
```

Wait for `state: available` before connecting a new session. During loading,
the health endpoint can already return HTTP 200.

The container uses the prepared weights and keeps persistent runtime data
beneath the weights mount. Storage locations are chosen by the operator.

## Connect and prepare a take

Open [Reactor Sandbox](https://reactor-sandbox.vercel.app/), choose
**Local (Direct)** and enter the Runtime URL. Create a session, then use the
model commands in Controls:

1. Upload an image and submit `set_avatar_image`.
2. Upload speech audio and submit `set_audio`.
3. Optionally submit `set_prompt`, `set_pose_video` and
   `set_generation_options`.
4. Check `state_update.ready` is true and `state_update.running` is false.
5. Send `start` with an empty payload.

Image and audio may be selected in either order. Await each command reply
before starting. A newly created session has empty input selections;
uploading files and changing conditions leave generation idle until `start`.

Use a client with image and audio upload support. In Sandbox, creating the
session and sending the model's `start` command are separate actions.
For remote browsers, HTTP signalling and WebRTC media both need a reachable route.
SSH HTTP-port forwarding alone covers signalling; a TURN/TCP relay and its
forwarded port can carry the media connection.

## Inputs and controls

All input setters apply to the next `start` and require an idle take.

| Command | Input and effect |
| --- | --- |
| `set_avatar_image(image)` | Select a decodable reference image, such as PNG, JPEG or WebP, up to 25 MiB and 40 million pixels. |
| `set_audio(audio)` | Select decodable speech audio, such as WAV, FLAC or MP3, up to 100 MiB and at least 1.92 seconds long. |
| `set_pose_video(pose_video)` | Select a prepared MP4 pose sequence up to 100 MiB, or pass null to clear pose guidance. |
| `set_prompt(prompt, negative_prompt)` | Describe appearance, surroundings and performance. Both text fields accept up to 4096 characters; `negative_prompt` is compatibility text with no effect on video in this serving profile. |
| `set_generation_options(seed, max_chunks)` | Select a seed from 0 through 2147483647 and a clip limit from 1 through 10000. Omitted fields use 420 and 10000 respectively. |
| `start()` | Begin a take from the accepted inputs and reset its progress counters. Requires both image and audio. |
| `stop()` | End the take and clear queued playback, retaining inputs, options and progress. |
| `reset()` | End the take, clear selections and progress, and restore the default seed and clip limit. |

The prompt conditions visual appearance and performance; speech comes from
the uploaded audio. Prepare pose guidance before uploading it. Pose-conditioned
generation requires separate acceptance testing.

A running take rejects input changes. Send `stop`, await its reply, update
conditions, then send `start` for another take. Commands run between inference
turns, so an in-flight clip can delay a stop or reset reply.

## Runtime boundary and output

The model loads once at service startup. Each take uses the selected image,
audio and conditions throughout its generation. Automatic completion retains
the selections and model weights for another take. Ending a session clears its
uploaded selections. Stopping an active take releases its generation workers;
the next take can therefore incur model-loading latency.

Each inference turn supplies one audiovisual clip. The first clip carries
45 frames; later clips carry 48. `main_video` plays at the model's native
25 FPS. `main_audio` carries the selected speech, resampled to mono 48 kHz
and aligned with the generated clip durations. Audio length and `max_chunks`
determine when a take finishes.

## Model messages

Commands return typed, command-correlated replies:

- `input_accepted` identifies an accepted image, audio, pose, prompt or
  generation-option selection.
- `take_changed` acknowledges `start`, `stop` or `reset`.
- `state_update` is the complete snapshot of selected filenames, text,
  seed, readiness, running state, clip limit, progress and generation error.
  A joining client receives a snapshot immediately. Accepted changes,
  generated clips and automatic completion or failure broadcast updates.
- `chunk_complete` reports the one-based generated clip number and its
  frame count. Client playback can lag generation progress.
- `generation_ended` reports `complete` when a take reaches its audio or
  clip limit, or contains the generation failure text.

Rejected commands return `command_error`. Missing image or audio produces
`inputs_required`; changing inputs during generation produces
`take_running`. Read command acknowledgements from the awaited call and use
`state_update` to render the current session.

## Inference performance and verification

Three-B200 container tests observed approximately 1.55–1.56 seconds of worker
build time for a steady 48-frame clip, which represents 1.92 seconds of
playback. First-clip preparation and compilation add substantial latency.
These worker timings exclude client backpressure and allow work to overlap;
end-to-end latency also depends on media transport and client buffering.

Playback can contain waiting silence while the next clip is being generated.

[check_model.py](check_model.py) exercises the real SDK upload and command
sequence and saves received video, received PCM and messages. Its `take.mp4`
preview pairs received video with the uploaded speech on the model timeline;
use `audio.wav` to inspect the actual received audio, including waiting
silence.

## Public source and model assets

- [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar)
  supplies the pinned inference source included by the YAML build.
- [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
  supplies the base model.
- [Quark-Vision/Live-Avatar](https://huggingface.co/Quark-Vision/Live-Avatar)
  supplies the LiveAvatar checkpoint.

Review the upstream repositories and model cards for their usage terms.
Checkpoint revisions are pinned in `liveavatar_assets.py`, and the image's
source revision is pinned in `reactor.yaml`.
