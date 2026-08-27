# AlayaWorld example

Serve the public AlayaWorld distilled autoregressive world model through
Reactor Runtime. The adapter calls AlayaWorld's `FlashAlayaPipeline` directly;
the weights, text encoder, spatial memory, prompt encoding, denoiser, and VAE
remain in the Runtime process and stay resident across chunks.

AlayaWorld generates four latent frames per turn. Each turn adds 32 RGB frames,
or about 1.33 seconds of video. Prompt and six-axis camera values are sampled
at the chunk boundary. Commands received during an in-flight CUDA turn apply
to the following chunk.

A turn takes longer to compute than the video it produces, and how much longer
depends on how far the camera moves, so the adapter declares no fixed frame rate
and emits each chunk with the time it took. Frames then play at the rate they
were produced, keeping the stream populated while the next turn runs.
`buffer_size` bounds the queue at one chunk, so a camera change is answered by
the next generated turn.

A new session starts without choosing a scene for the user. Upload an image
with `set_image`, or invoke `random_image` to select one of the public
AlayaWorld examples. Either command initializes the autoregressive cache and
starts generation.

## Run with the Reactor CLI

This directory is a `reactor` workspace. The manifest names the model and its
B200 resource, and its `build` block defines the complete Python 3.12 and CUDA
12.8 image with Reactor Runtime 3.2.5. `requirements.txt` contains the model
dependencies. The host needs the
[`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
the NVIDIA Container Toolkit, and a compatible NVIDIA GPU. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Gemma is gated. Accept its Hugging Face license, export a read token, then build
the image and expose one GPU to the container:

```sh
cd models/alayaworld
export HF_TOKEN=hf_your_read_token

reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

The bare `-e HF_TOKEN` form forwards the host value without putting the token in
the Docker command line. `--gpus device=3` selects host GPU 3 and presents it as
device 0 inside the container; `--gpus all` exposes every GPU. The manifest
requests one B200 when the same workspace is deployed.

`reactor run` reuses the configured image and builds one on the first run when
no local image exists. It serves WebRTC signaling on
`http://localhost:8080` by default. Rebuild after changing code or
dependencies. A different port is applied to both the container and Runtime:

```sh
reactor build && reactor run --gpus device=0 -e HF_TOKEN --port 18080
```

Check readiness after the model finishes loading. Use the port passed to
`reactor run`; for the command above that is `18080`:

```sh
curl -s localhost:18080/health
curl -s localhost:18080/schema
```

## Play it in the browser

[`demo/`](./demo) is a small Next.js app for this model: pick a starting image,
write a prompt, and steer the camera from the keyboard while the video streams
back. With the container running, start it in a second terminal:

```sh
cd models/alayaworld/demo
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The page lists three steps —
**Connect**, choose a starting image, then drive — and reports which of them it is
waiting on, so a disabled control always says why.

The app connects to `http://localhost:8080` with no configuration, matching where
`reactor run` serves. Tell it about a different port through the environment:

```sh
cp .env.example .env
# REACTOR_LOCAL_URL=http://localhost:18080   # match `reactor run --port`
```

`W`/`A`/`S`/`D` move, `I`/`J`/`K`/`L` look, `Space`/`C` change height, and `Q`/`E`
roll. [`demo/README.md`](./demo/README.md) covers the full mapping and how its
typed client is generated from the `/schema` response above.

As an alternative, the [Reactor Sandbox](https://reactor-sandbox.vercel.app/)
reaches the same container with **Local (Direct)**.

## Public source and model assets

The CLI bind-mounts `runtime.weights_path` from `reactor.yaml` and exports that
directory to the model as `REACTOR_WEIGHTS_PATH`. The checked-in default keeps
AlayaWorld under `~/.cache/reactor_registry/alayaworld`, outside the image and
the Git checkout, so container rebuilds retain every large download.

On first load the adapter clones the pinned AlayaWorld, Depth-Anything-3, and
TAEHV revisions, downloads the merged checkpoint and Gemma text encoder, and
populates the pinned DA3 cache. Interrupted Hugging Face downloads resume on
the next start. Later runs verify and reuse the completed resources. The
playground cases used by `random_image` arrive in the AlayaWorld checkout; no
separate sample dataset is required.

The TAEHV checkout supplies the tiny decoder that `inference.bank_taehv` uses for
spatial-memory pixels. Its `taeltx2_3_wide` weights are the larger LTX-2.3
variant published on a dedicated upstream branch. `assets.taehv_source` pins
that branch revision. Setting `bank_taehv: false` skips the decoder; dropping
`assets.taehv_source` also skips the download.

`source.path` in `alayaworld.yaml` is relative to the mounted weights root.
Every other relative model, configuration, and playground path is then resolved
from that source checkout. To place the cache elsewhere, change only
`runtime.weights_path` in `reactor.yaml`; the CLI creates and mounts that path.

Model loading reports preparation progress in the container log. A failed
download keeps health unavailable and names the public repository; gated model
errors point to Hugging Face authentication. An existing source checkout must
match the configured immutable revision; startup reports any mismatch.

The adapter imports DA3's inference modules directly from its pinned source
checkout.

AlayaWorld and its weights use the LTX-2 Community License for academic and
non-commercial use. Gemma and Depth-Anything-3 retain their own terms.

## Controls

- `set_image` accepts uploaded JPEG, PNG, WebP, or BMP bytes plus an optional
  prompt, then resets from that image.
- `random_image` selects a different built-in example when possible and applies
  its matching prompt.
- `set_forward`, `set_strafe`, and `set_vertical` control local Z, X, and Y
  translation. Positive values move forward, right, and up.
- `set_pitch`, `set_yaw`, and `set_roll` control local X, Y, and Z rotation.
  Positive values look up, turn right, and roll clockwise.
- Every camera command accepts `-1.0` to `1.0` and returns a
  `CameraMotionChanged` message containing the complete six-axis state and the
  one-based chunk expected to consume it.
- `set_prompt` changes the text condition for the next chunk and returns a
  `PromptQueued` confirmation with the expected one-based chunk number.
- `reset` rebuilds the autoregressive and spatial-memory state from the initial
  selected image. Its optional non-negative `seed` selects the next reproducible
  rollout, and `RolloutResetQueued` reports the seed and replaced chunk count.

Prompt and camera commands require a selected image. Prompts are stripped and
must contain non-whitespace text. `ImageSelected` reports the effective prompt
alongside the uploaded or built-in filename, so every state-changing control has
a typed confirmation in the model-message timeline. Successful controls also
broadcast a `StateUpdate` snapshot with the complete prompt, image, rollout,
and six-axis camera state. A newly connected viewer receives the same snapshot
immediately.

The frontend owns keyboard, pointer, joystick, sensitivity, and layout mapping.
Key down sends `-1` or `1`; key up sends zero. [`demo/`](./demo) binds all six
axes to twelve keys — W/S and A/D translate, I/K and J/L rotate, Space/C and Q/E
cover height and roll — which suits a model that samples one velocity per chunk
better than mouse deltas would. The same axes are equally reachable from touch
controls, twin joysticks, or a six-degree-of-freedom controller. The backend
normalizes simultaneous translation and rotation axes before integrating them
into the pixel-rate camera-to-world matrices consumed by AlayaWorld.

## Image uploads

The `set_image` command declares an `UploadedFile` parameter in Runtime's
schema. A Reactor client reserves a session upload slot, writes the raw bytes to
its returned URL, and sends the resulting upload reference with the command.
This lets a schema-driven frontend render a real file picker while the upload
channel carries the binary data.

[`example_images/`](./example_images) contains two licensed upstream still
images ready for manual upload. These convenience copies are independent of
`random_image`, which reads the same pinned playground cases from the source
checkout prepared at startup.

Uploads are limited to 25 MiB and 100 million decoded pixels. EXIF orientation
is applied before the image is resized and center-cropped to 960x544. The
configured `inputs.upload_template` supplies camera calibration and a trajectory
horizon; its pixels are not used. Uploaded bytes remain session-scoped and are
released when the session ends.

`inputs.random_images` in `alayaworld.yaml` lists the known image triplets used
by `random_image`. The current configuration exposes the two public playground
cases shipped with the pinned AlayaWorld source.

## Long-running sessions

`stream.max_chunks_per_rollout` defaults to 512. After chunk 512, the adapter
broadcasts `RolloutResetQueued`, releases camera motion, and rebuilds the
autoregressive state from the selected image, active prompt, and seed without
reloading model weights. Chunk numbering then starts again at 1. This bounds the
dense camera trajectory while allowing the Reactor session to continue.

`inference.attention_backend` chooses which attention implementation serves the
model. `flash_attention_4` is the default and needs a Hopper or Blackwell GPU;
`pytorch` serves anything older, and `upstream` leaves AlayaWorld's own selection
alone. The adapter binds the callable on the loaded attention modules. Masked
blocks use the PyTorch implementation that builds the banded sliding-window
mask.

`inference.bank_taehv` decodes spatial-memory pixels — the depth and warp
sources the model builds its memory from — with a tiny decoder. Display frames
continue through the full VAE. The memory decode is lossy. The checkpoint comes
from the pinned `assets.taehv_source` checkout; its filename selects the
decoder architecture, so renaming it silently changes the model.

The adapter uses the default `torch.compile` mode and Inductor-compiled kernels,
which support Runtime's off-loop chunk generation.

During load, the adapter generates `inference.warmup_chunks` throwaway turns
from a built-in scene to compile those kernels and the attention backend. Health
stays unavailable until warmup finishes. Set the value to `0` to compile on
first use; the caches live in the container, so a restart compiles again.

Autoregressive history remains a 16-latent sliding window. The adapter keeps
only the latest generated latent outside that window and bounds the spatial
bank at 320 frames: 160 dense recent frames plus 160 keyframes sampled across
the older trajectory. This keeps GPU memory and retrieval work bounded while
preserving sparse long-range viewpoints within the active rollout. Very old
revisits therefore have less spatial detail than recent ones, and the automatic
rollout reset deliberately trades unbounded history for predictable resources.

## Streaming decode

The public VAE is non-causal. The adapter decodes each new chunk with six
latents of left context, which keeps memory bounded. Live-edge frames differ
from a future-aware offline decode. Recording captures the same interactive
frames sent over `main_video` and requires `ffmpeg` on `PATH`.
