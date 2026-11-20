# LiveAvatar on Reactor

A talking avatar for voice agents. From one portrait, [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar)
renders a face that lip-syncs, live, to the speech an agent streams to it, at 512x512 and 25 FPS on one NVIDIA B200.
The model joins a [LiveKit](https://livekit.io) room as a LiveKit avatar, so any LiveKit voice agent can speak
through it with LiveKit's avatar protocol: the agent sends speech, the avatar publishes video and that speech,
lip-synced, on the agent's behalf.

This folder is the model, a `reactor` workspace. [The example's README](../README.md) runs it with a voice agent and a
page you talk to it from.

## How a call works

```mermaid
sequenceDiagram
  participant P as Your app (server)
  participant L as LiveKit room
  participant G as Voice agent
  participant R as This model (Reactor)
  P->>L: create the room, dispatch the agent
  L->>G: the agent joins
  P->>R: start a session, starting input set_avatar(portrait)
  P->>R: join the room with the avatar's token
  R->>L: joins as the avatar, publishing on behalf of the agent
  Note over G,R: speech on lk.audio_stream; the avatar's video and voice to the user
  G-->>R: lk.clear_buffer when the user interrupts
  R-->>G: lk.playback_started, lk.playback_finished
```

Your app's server starts the avatar the same way whether the model runs on your machine (`reactor run`) or on
Reactor: start a session with `set_avatar` as its starting input, upload the portrait, and ask the session to join
the room with the avatar's token. [`../client/lib/reactor.ts`](../client/lib/reactor.ts) does both.

## The contract

| | Name | What it carries |
| --- | --- | --- |
| Input track | `agent_audio` | Speech to say, 24 kHz mono, one LiveKit byte stream (`lk.audio_stream`) per utterance |
| Output tracks | `main_video`, `main_audio` | The avatar's frames, and the speech as it plays with them (24 kHz) |
| Command | `set_avatar` | The portrait; only the session's starting input can send it |
| Command | `clear_buffer` | Stop the utterance playing and return to idle (the `lk.clear_buffer` RPC) |
| Command | `user_speaking_state` | `speaking` or `idle` (data packets on `avatar.user_speaking_state`) |
| Message | `playback_started` | The first frame of an utterance plays (the `lk.playback_started` RPC to the agent) |
| Message | `playback_finished` | An utterance played to its end or was cut, with `playback_position` and `interrupted` |

The `runtime.livekit` block in [`reactor.yaml`](reactor.yaml) maps them onto the room. `GET /schema` on a running
model lists every field.

## What you need

- Linux, one NVIDIA B200, Docker and the NVIDIA Container Toolkit, and the
  [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation).
- About 50 GB for the weights.

## 1. Prepare the weights

```sh
pip install "huggingface_hub[cli]<1"
python prepare_weights.py --output /path/to/liveavatar-weights --allow-download
```

The folder gets [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) in `wan2_2/` and the
[LiveAvatar checkpoint](https://huggingface.co/Quark-Vision/Live-Avatar) in `liveavatar_lora/`, at the revisions
pinned in [`liveavatar_assets.py`](liveavatar_assets.py). The image build clones the pinned upstream source.

## 2. Start the model

```sh
reactor build
reactor run --gpus device=0 --weights /path/to/liveavatar-weights --port 8080
```

**Check:** `curl http://localhost:8080/health` reports `"state":"available"`. Loading compiles the model and renders
`warmup_blocks` blocks, so the first session runs at full speed. The compile caches go to `LIVEAVATAR_WORK_DIR` (by
default a folder under the system's temporary directory); with them warm, a restart takes about two minutes, and the
weights can be mounted read-only.

## 3. Talk to it

[The example's README](../README.md) starts the voice agent and the page, and points the page at
`http://localhost:8080`.

## Performance

A block is 12 frames, 0.48 s of video: the model keeps real time while a block renders in less than that. On one
B200 at 512x512, measured offline with [`bench.py`](bench.py) over 360 blocks of speech:

| Profile | Per block (median / max) | Real-time factor |
| --- | --- | --- |
| `fp8: true` (the default) | 455 / 461 ms | 0.95 |
| `fp8: false` | 542 / 546 ms | 1.13, slower than real time |

`fp8` runs the transformer blocks' projections on FP8 tensor cores with per-tensor scales, upstream's `--fp8`
scheme; the frames change slightly. Without it, one B200 does not keep real time.

```sh
python bench.py --weights /path/to/liveavatar-weights --source /path/to/LiveAvatar \
  --portrait face.png --audio speech.wav --blocks 40 --compile --fp8 --clip out.mp4
```

## Configuration

[`config.yml`](config.yml) sets the default portrait, the text prompt, the seed, the output size, `fp8`, the
warm-up, and how speech is paced into blocks. Each key is documented in place.

## Notes on the code

- **The application half is model-agnostic.** [`avatar_app.py`](avatar_app.py) is the whole LiveKit avatar
  protocol: utterances out of `agent_audio`, a playout clock that pairs each block's frames with its audio and
  dates the playback reports, interrupts, and the commands. It drives any model that implements
  [`avatar_contract.py`](avatar_contract.py): render a block of video from a block of speech, and reset.
  [`liveavatar_app.py`](liveavatar_app.py) only builds LiveAvatar's model half. Another avatar model fits the same
  application by implementing that contract.
- **The model streams one block per call.** Upstream renders a whole file in 48-frame clips of four causal blocks,
  each denoising step keeping its own key/value cache. [`liveavatar_engine.py`](liveavatar_engine.py) runs that
  loop one block at a time from audio that arrives during the session: the per-step caches roll over one clip,
  the clip noise is drawn every fourth block as upstream draws it, and the wav2vec features of each block read the
  audio before it as context.
- **The streamed-block forward is rewritten for speed, not changed.** [`liveavatar_fast.py`](liveavatar_fast.py)
  computes upstream's forward for this one case with the same weights; it differs from upstream's forward by as
  much as swapping the attention kernel does. It keeps each step's cache in one buffer, broadcasts the modulation
  per frame, builds the rotary tables once per block, caches the prompt's cross-attention keys, and runs the
  audio encoder once per block; each step then compiles and is captured as a CUDA graph.
- **Its own loop.** The app overrides `run()`: a block renders on a worker thread while commands, the speech
  pump and the playback reports run on the event loop, so `clear_buffer` lands mid-render.
- **No listening pose.** LiveAvatar renders a block marked listening like any silent one.

## Tests

From this folder, without a GPU or the weights:

```sh
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

## Public source and model assets

- [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar): the inference source, pinned in
  `reactor.yaml`.
- [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B): the base model.
- [Quark-Vision/Live-Avatar](https://huggingface.co/Quark-Vision/Live-Avatar): the LiveAvatar checkpoint.

Review the upstream repositories and model cards for their usage terms.
