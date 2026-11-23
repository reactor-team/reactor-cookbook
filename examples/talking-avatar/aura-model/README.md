# AURA on Reactor

A talking avatar for voice agents. From one portrait, [AURA](https://github.com/kyutai-labs/aura) renders a face that
lip-syncs, live, to the speech an agent streams to it, at 576x576 and 16 FPS on one NVIDIA B200. The model joins a
[LiveKit](https://livekit.io) room as a LiveKit avatar, so any LiveKit voice agent can speak through it with LiveKit's
avatar protocol: the agent sends speech, the avatar publishes video and that speech, lip-synced, on the agent's behalf.

This folder is the model, a `reactor` workspace, and the example's default. [The example's README](../README.md) runs
it with a voice agent and a page you talk to it from. [`../liveavatar-model/`](../liveavatar-model) is the alternative
model behind the same contract.

**License.** AURA's code is under the
[Snap Inc. Non-Commercial License](https://github.com/kyutai-labs/aura/blob/main/LICENSE.txt), and the AURA-1.3B
weights under [CC BY-NC 4.0](https://huggingface.co/kyutai/AURA-1.3B): for research and other non-commercial use only.
This folder holds none of that code; the image installs it from AURA's repository.

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
| Message | `avatar_failed` | The model failed during the session, with the reason; no more frames follow in it |

The `runtime.livekit` block in [`reactor.yaml`](reactor.yaml) maps them onto the room. `GET /schema` on a running
model lists every field.

## What you need

- Linux, one NVIDIA B200 (the model holds about 36 GB), Docker and the NVIDIA Container Toolkit, and the
  [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation).
- About 18 GB for the weights.

## 1. Prepare the weights

```sh
pip install "huggingface_hub[cli]<1"
python prepare_weights.py --output /path/to/aura-weights --allow-download
```

The folder gets the [AURA-1.3B](https://huggingface.co/kyutai/AURA-1.3B) checkpoint in `aura/`, the VAE and umT5 text
encoder of [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) in `wan2_1/`, and Kyutai's
[Mimi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) speech codec in `mimi/`, at the revisions pinned in
[`aura_assets.py`](aura_assets.py). The model reads only this folder, never the network.

## 2. Start the model

```sh
reactor build
reactor run --gpus device=0 --weights /path/to/aura-weights --port 8080
```

**Check:** `curl http://localhost:8080/health` reports `"state":"available"`. Loading compiles the transformer and
renders `warmup_blocks` chunks, so the first session runs at full speed. The compile caches go to `AURA_WORK_DIR` (by
default a folder under the system's temporary directory); with them warm, a restart takes under a minute, and the
weights can be mounted read-only.

## 3. Talk to it

[The example's README](../README.md) starts the voice agent and the page, and points the page at
`http://localhost:8080`.

## Performance

A chunk is 16 frames, 1.0 s of video: the model keeps real time while a chunk renders in less than that. On one B200,
measured offline with [`bench.py`](bench.py) over 24 chunks of speech, with `compile: true`, a chunk renders in
606 ms median and 609 ms at most: a real-time factor of 0.61.

```sh
python bench.py --weights /path/to/aura-weights --portrait portrait.jpg --audio speech.wav --chunks 30
```

## Configuration

[`config.yml`](config.yml) sets the default portrait, the text prompt, the seed, the audio context, compilation, the
warm-up, and how speech is paced into chunks. Each key is documented in place.

## Notes on the code

- **The application half is shared.** [`avatar_app.py`](avatar_app.py) and [`avatar_contract.py`](avatar_contract.py)
  are the same files as in [`../liveavatar-model/`](../liveavatar-model), and a test keeps them identical; the
  application's behaviour is tested there. [`aura_app.py`](aura_app.py) only builds AURA's model half.
- **AURA's code runs as it is.** [`aura_model.py`](aura_model.py) builds AURA's reference pipeline from the weights
  folder, as AURA's own `Aura("aura-1.3b")` builds it from the Hugging Face hub, and runs AURA's chunk loop one chunk
  per call: the rolling key/value cache, the four denoising steps, and the streaming VAE decode are AURA's.
- **Speech that arrives during the session.** AURA reads the Mimi features of a whole file. Here each chunk's features
  come from Mimi run on the chunk's speech plus up to 4 s before it (`audio_context_s`), sampled at the positions AURA
  samples a whole file at (`feature_positions`).
- **Failures stop the session, not the runtime.** A portrait that cannot be decoded is refused at `set_avatar`, so the
  call does not start. An error from the model during a call stops that call's video and sends `avatar_failed`; the
  runtime keeps serving.

## Tests

From this folder, without a GPU or the weights:

```sh
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

## Public source and model assets

- [kyutai-labs/aura](https://github.com/kyutai-labs/aura): the inference source, pinned in `requirements.txt`. Snap Inc.
  Non-Commercial License.
- [kyutai/AURA-1.3B](https://huggingface.co/kyutai/AURA-1.3B): the AURA checkpoint. CC BY-NC 4.0.
- [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B): the VAE and text encoder. Apache 2.0.
- [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16): the Mimi speech codec.

Review the upstream repositories and model cards for their usage terms.
