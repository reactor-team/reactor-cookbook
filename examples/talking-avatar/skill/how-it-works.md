# How the talking avatar works

The example ships two avatar models behind one application. **AURA** (`aura-model/`) is the one the example deploys.
**LiveAvatar** (`liveavatar-model/`) is the alternative. Both folders hold the same application half,
`avatar_app.py`: it speaks LiveKit's avatar protocol, cuts the agent's speech into blocks, and keeps the picture and
the sound together. Each model half only renders a block of video from a block of speech. So everything under "The
application" holds for both models, and the model sections hold for one each.

## Contents

- AURA (the default)
  - The model
  - One chunk at a time
  - How the face follows the voice
  - Keeping real time
- The application (both models)
  - How the picture and the sound stay together
  - Interrupts and playback reports
  - When the model fails
  - Where the latency goes
- LiveAvatar (the alternative)
  - The model
  - One block at a time
  - How the face follows the voice
  - Quiet between utterances
  - Keeping real time on one B200

## AURA (the default)

### The model

[AURA-1.3B](https://huggingface.co/kyutai/AURA-1.3B) is a 1.3-billion-parameter video diffusion transformer on
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), conditioned on speech and distilled by its authors
into a causal, four-step model: it generates video forward in time, one chunk at a time, each chunk in four denoising
steps. It renders 576x576 at 16 FPS. Per chunk it reads:

- the chunk's **speech**, as features from [Mimi](https://huggingface.co/kyutai/mimi), Kyutai's speech codec;
- a **portrait**, encoded once per clip by the VAE;
- a **text prompt**, encoded once at load by umT5 (`prompt` in `aura-model/config.yml`);
- the **chunks before it**, through one rolling key/value cache.

The code is AURA's own, installed from its repository at a pinned commit; `aura-model/aura_model.py` builds it from the
weights folder and feeds it one chunk per call.

### One chunk at a time

A chunk is four latent frames, which the VAE decodes to sixteen video frames: 1.0 s at 16 FPS. The first chunk of a
clip decodes to thirteen.

The key/value cache keeps the first chunk of the clip as an identity anchor that is never evicted, plus the latest
chunks in a rolling window; the portrait's tokens join the keys at every step. Past the length AURA was trained on,
the rotary positions stop growing, so the geometry stays the one it saw in training however long the call runs. The
streaming VAE decoder keeps its own causal cache, so consecutive chunks join without a seam.

### How the face follows the voice

For each chunk the model half (`aura_model.py`):

1. runs Mimi on the chunk's 1.0 s of speech plus up to 4 s of earlier speech (`audio_context_s`), so the features at
   the chunk's start have context, as they have when AURA encodes a whole file;
2. takes Mimi's 12.5-per-second features at the position of each of the chunk's sixteen video frames, the way AURA
   samples them from a whole file (`feature_positions`): one feature vector per video frame.

Inside the transformer, an audio encoder turns those per-frame features into audio tokens, and at 11 of the 30
layers each latent frame's video tokens attend to the audio tokens of that same latent frame (four video frames). That
pairing moves the mouth with the sound: the frame shown at time *t* was generated from the speech at time *t*.

### Keeping real time

A chunk is 1.0 s of video, so the model keeps real time while it renders a chunk in less than a second. On one B200 a
chunk takes about 0.61 s with `compile: true`, and the model holds about 36 GB. AURA's own code runs as it is: each
transformer block is compiled, nothing else changes.

## The application (both models)

### How the picture and the sound stay together

The application half (`avatar_app.py`) never lets the two drift:

- **Speech is cut into utterances.** The agent sends each utterance as one LiveKit byte stream on `lk.audio_stream`;
  the runtime cuts it into 10 ms frames tagged with the stream's id and marks its end. A block carries speech from one
  utterance only: a full block as soon as one is in, and the utterance's last, shorter part once its end arrives,
  padded with silence.
- **Each block is emitted with the exact audio that drove it.** A block's frames go out together with its samples in
  one output, and the runtime plays video and audio from that output in lockstep at the model's frame rate. When a
  block comes back shorter (the first block of a clip), the speech it did not show is given back to the utterance and
  plays with the next block.
- **A playout clock mirrors the runtime's pacer.** Every emitted block is placed on it, starting when the block before it
  ends. The clock says how far playout runs ahead of now, which paces the idle blocks and caps how far ahead speech
  renders (`speech_lead_s`), and it dates the playback reports.
- **Speech does not wait behind idle video.** When an utterance's first block is ready and only idle frames are queued
  ahead of it, those are dropped (the picture holds its last frame for the cut), so the reply starts at once
  (`cut_idle_for_speech`).

### Interrupts and playback reports

- `lk.clear_buffer` from the agent (the user talked over the avatar) lands at once, even while a block renders: the
  queued media is flushed with the last frame held, the speech still queued is dropped, and any block rendered for the
  cut speech is dropped when it lands. The next block is idle.
- `lk.playback_finished` goes back to the agent with `playback_position` (seconds of the utterance that actually
  played, read off the playout clock) and `interrupted`. The agent uses it to know what the user heard, so its
  conversation history matches what was said.
- `lk.playback_started` goes back when the first frame of an utterance plays, not when its first byte arrived.
- `user_speaking_state` (data packets on `avatar.user_speaking_state`) marks idle blocks as listening.

### When the model fails

- **A call that cannot run does not start.** `set_avatar` refuses a portrait that is not a decodable image, and the
  page's server reports the refusal.
- **A model error stops the call, not the model.** When the model raises during a call, that call renders nothing
  more: the last frame holds, an `avatar_failed` message says why, and the model is reset when the call ends. The
  runtime stays up and serves the next call.
- **A block carries the portrait whenever the model needs it.** Each portrait gets an id unique for the runtime's
  life, and the application tracks the one the model has bound, clearing it on every reset.

### Where the latency goes

From the agent's first byte of an utterance to the avatar's first frame:

- **waiting for a block of speech:** 1.0 s of audio for AURA, 0.48 s for LiveAvatar. Speech from a realtime model
  usually arrives faster than it plays, so this is shorter in wall time;
- **rendering the block:** about 0.61 s for AURA, 0.45 s for LiveAvatar;
- **the network:** the room, between the agent, the model and the browser.

A block's length is fixed by each model's training.

## LiveAvatar (the alternative)

### The model

LiveAvatar is [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B), a 14-billion-parameter video diffusion
transformer conditioned on speech, distilled by the LiveAvatar authors into a causal, four-step model. It renders
512x512 at 25 FPS. Per block it reads:

- the block's **speech**, as wav2vec2 features;
- a **portrait**, encoded once per clip by the VAE into reference tokens;
- a **text prompt**, encoded once by T5 (`prompt` in `liveavatar-model/config.yml`);
- the **blocks before it**, through key/value caches.

### One block at a time

A block is three latent frames, which the VAE decodes to twelve video frames: 0.48 s at 25 FPS. Upstream groups four
blocks into a 48-frame clip and draws the clip's noise at its start.

The property that makes streaming possible is upstream's "timestep forcing": every denoising step keeps its own
key/value cache of past blocks, at that step's noise level. Block *k* at step *i* needs only block *k-1*'s step-*i*
entries, never a finished frame. So a block can be rendered as soon as its own speech is in.

`liveavatar-model/liveavatar_engine.py` runs that loop one block per call:

- the four per-step caches roll over one clip (4 blocks, 1.92 s), plus a reference cache filled once per portrait;
- after the first clip, upstream re-anchors the reference on the first frame it generated rather than on the encoded
  portrait, which keeps the identity stable;
- the clip noise is drawn every fourth block from a seeded generator, as upstream does per clip;
- the streaming VAE decoder keeps its own causal cache, so consecutive blocks join without a seam. The first block of a
  clip decodes to nine frames rather than twelve.

### How the face follows the voice

The agent's speech arrives at 24 kHz. For each block the model half (`liveavatar_model.py`):

1. resamples the block's 0.48 s of speech to 16 kHz, carrying a few samples from the block before so there is no edge;
2. runs wav2vec2 on that block plus 4 s of earlier speech (`audio_context_s`), so the features at the block's start
   have context (`_audio_features` in `liveavatar_engine.py`);
3. takes the wav2vec2 features (all 25 layers, 50 per second) at the position of each of the block's twelve video
   frames, the way upstream samples them from a whole file: one feature vector per video frame.

Inside the transformer, a small audio encoder turns those per-frame features into audio tokens, and at twelve of the
forty layers each latent frame's video tokens cross-attend to the audio tokens of that same latent frame (four video
frames).

### Quiet between utterances

LiveAvatar was trained and evaluated on continuous speech; its paper loops the audio of long-video tests to avoid
silence gaps. Left alone, its face keeps moving in silence as if it were still talking. Measured as the mean change
between consecutive frames over three seeds and two portraits, a silent block moved about two thirds as much as a
speech block, where upstream's own offline pipeline, run on the same audio, moves about half as much.

Two settings in `liveavatar-model/config.yml` bring it below upstream:

- **`listening_prompt`:** silent and listening blocks are conditioned on a text that asks for closed lips and a still
  head, rather than on the speaking prompt, which asks for lively expressions and head movement. Speech blocks keep
  the speaking prompt, and the switch shows no jump in or out of silence. Each prompt's cross-attention keys and values
  are computed once at load and copied into the buffers the captured steps read, so switching costs nothing per block.
- **`audio_context_s: 4`:** the audio encoder reads 4 s of earlier speech with each block. Upstream encodes a whole
  file, so silence is encoded next to the speech around it; with 1 s, a block a second into a silence is encoded as
  bare digital silence, and the face moves more. Of 1, 2, 4, 6 and 8 s, 4 s calmed silence the most.

With both, a silent block moves under half as much as a speech block, and speech moves as before. Two other levers
made silence move more and are not used: rendering silent blocks on a schedule that stays nearer to noise, and feeding
the audio encoder zero features in place of silence (the model's "no audio" mode is a lively, talking face).

### Keeping real time on one B200

A block is 0.48 s of video, so the model keeps real time while it renders a block in less than 480 ms. On one B200 at
512x512 it takes about 455 ms with `fp8: true` and about 540 ms without (`liveavatar-model/README.md` has the
measurements; `bench.py` reproduces them).

`liveavatar_fast.py` computes upstream's forward for exactly this case with the same weights; its output differs from
upstream's by as much as swapping the attention kernel does. What it removes is overhead around the arithmetic:

- each step's keys and values live in one buffer, the reference tokens first and the rolling window after them, so a
  layer attends to a slice instead of concatenating a 12k-token window with the reference every call;
- the six per-layer modulation vectors broadcast per latent frame instead of being repeated to every token in float32;
- the rotary tables are built once per block, the prompt's cross-attention keys once per session, and the audio
  encoder runs once per block rather than at every step;
- nothing reads a value back to the CPU inside a step, so each layer compiles, and each step is captured as a CUDA
  graph and replayed.

`fp8` runs the transformer blocks' projections on FP8 tensor cores with per-tensor scales (upstream's `--fp8` scheme);
it changes the pixels slightly. Matrix multiplies plus attention alone are about 90 ms per step in bfloat16, so four
steps and the decode do not fit in 480 ms without it.
