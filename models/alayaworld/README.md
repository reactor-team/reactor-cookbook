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

The adapter uses Reactor Runtime 3.10's native step loop. The application resolves
uploads and plans camera poses; `alayaworld_model.py` owns the native engine and
rollout cache. A new world first initializes its cache and reports its exact
camera origin. Subsequent steps generate one native chunk and publish its media
and state. Runtime measures each generation step for playout pacing.

This directory is a `reactor` workspace. The manifest names the model and its
B200 resource, and its `build` block defines the complete Python 3.12 and CUDA
12.8 image with Reactor Runtime 3.10.0. `requirements.txt` contains the model
dependencies. The host needs the
[`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
the NVIDIA Container Toolkit, and a compatible NVIDIA GPU. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

The image build fetches the pinned AlayaWorld and Depth-Anything-3 sources.
Build the weights folder once. It holds the merged AlayaWorld checkpoint, the
Gemma 3 text encoder and the Depth-Anything-3 checkpoint at their pinned
revisions, about 65 GB. Gemma is gated: accept its terms on the
[model card](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
with your Hugging Face account, then pass a token that can read it:

```sh
cd models/alayaworld
pip install huggingface-hub pyyaml
HF_TOKEN=<your token> python prepare_weights.py --output ./weights
```

Then build the image and expose one GPU to the container. `runtime.weights_path`
points `reactor run` at the same folder, and the deploy uploads it, so a new
instance does not download anything:

```sh
reactor build
reactor run --gpus device=0
reactor deploy production --weights ./weights
```

`--gpus device=3` selects host GPU 3 and presents it as
device 0 inside the container; `--gpus all` exposes every GPU. The manifest
requests one B200 when the same workspace is deployed.

`reactor run` reuses the configured image and builds one on the first run when
no local image exists. It serves WebRTC signaling on
`http://localhost:8080` by default. Rebuild after changing code or
dependencies. A different port is applied to both the container and Runtime:

```sh
reactor build && reactor run --gpus device=0 --port 18080
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

`reactor build` clones AlayaWorld and Depth-Anything-3 into `/opt` at the
revisions that `alayaworld.yaml` names under `source` and `da3_source`. Startup
checks that each checkout is at its revision and asks for a rebuild when one is
missing. The playground cases used by `random_image` arrive in the AlayaWorld
checkout; no separate sample dataset is required. The adapter imports DA3's
inference modules directly from its pinned source checkout.

`prepare_weights.py` writes one folder per `assets` entry in
`alayaworld.yaml`, each holding the files of one Hugging Face repository at its
pinned revision:

| Folder | Repository and revision | Size |
| --- | --- | --- |
| `AlayaWorld/` | `AlayaLab/AlayaWorld@f9699009a00f93214d8d2942d6546f3fa5398bea` | 33.6 GB |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | `google/gemma-3-12b-it-qat-q4_0-unquantized@68f7ee4fbd59087436ada77ed2d62f373fdd4482` (gated) | 24.4 GB |
| `DA3NESTED-GIANT-LARGE-1.1/` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1@b2359bdf726fb44ef62acca04d629dcf158053e7` | 6.8 GB |

The sources are `AlayaLab/AlayaWorld@1abbe2e196603ef9f8eeedfddd427b4d37d57bc1`
and `ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
on GitHub.

A second run downloads only the missing files. The model reads the folder
read-only and nothing downloads at runtime (`HF_HUB_OFFLINE=1`). A missing file
stops `load()` with an error that names it and the command that builds the
folder.

AlayaWorld and its weights use the LTX-2 Community License for academic and
non-commercial use. Gemma and Depth-Anything-3 retain their own terms.

## Controls

`set_camera_path(trajectory)` accepts a JSON upload after image selection.
Its `poses` array contains `[32 * chunks + 8, 4, 4]` camera-to-world matrices
at 24 FPS: the first pose anchors the image, the next poses describe the generated
frames, and seven final poses provide look-ahead. The reply reports the chunk
and output frame counts. Each chunk emits 32 new frames; generation stops at
the path end. `reset` replays the path. Selecting another image restores
held camera controls. Uploads are limited to 8 MiB.

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
checkout in the image.

Uploads are limited to 25 MiB and 100 million decoded pixels. EXIF orientation
is applied, then the image is resized and center-cropped to 960x544 by
AlayaWorld's own case-image fit, so an upload matches the same image passed to
the official `inference.run`. The configured `inputs.upload_template` supplies
camera calibration, the history camera prefix and a trajectory horizon; its
pixels are not used. The spatial memory is built when the first chunk's camera
poses arrive, so its depth is aligned to the requested path. Uploaded bytes remain session-scoped and are
released when the session ends.

`inputs.random_images` in `alayaworld.yaml` lists the known image triplets used
by `random_image`. The current configuration exposes the two public playground
cases shipped with the pinned AlayaWorld source.

## Long-running sessions

Generation stops after 512 chunks and preserves the final world state. Use
`reset` or select another image to start a fresh rollout without reloading weights.

The recipe uses PyTorch attention and the full VAE for both display frames
and spatial-memory pixels. It does not install FlashAttention 4 or download
a tiny spatial-memory decoder.

The default `torch.compile` mode and FlexAttention match the original recipe.
Load-time warmup is disabled, so the first generation can include compilation
latency. Chunk sizes, autoregressive history and spatial-memory limits are
unchanged.

Autoregressive history remains a 16-latent sliding window. The native spatial
bank retains all accumulated frames for retrieval until an explicit reset.
Memory use grows with the rollout length. Already streamed output latents are
released independently of the spatial bank and autoregressive history.

## Streaming decode

The public VAE is non-causal. The adapter decodes each new chunk with six
latents of left context, which keeps memory bounded. Live-edge frames differ
from a future-aware offline decode. Recording captures the same interactive
frames sent over `main_video` and requires `ffmpeg` on `PATH`.
