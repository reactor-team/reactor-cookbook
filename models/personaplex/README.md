# PersonaPlex example

Hold a real-time spoken conversation with
[NVIDIA PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) — a 7B
Moshi-architecture speech-to-speech model — served through Reactor Runtime. A
client sends its microphone and receives the agent's voice, plus the agent's own
words as text. The persona is a sentence you write ("You work for CitySan
Services and your name is Ayelen Lucero"), and the voice is one of the model's
packaged voice prompts.

Reach for this when you want a spoken agent that behaves like a person on a
call rather than a walkie-talkie: it can be interrupted mid-sentence, it talks
over you, and it takes a turn back quickly. That is the model's doing, not the
adapter's, and the next section explains why it comes for free.

## Runtime boundary

PersonaPlex encodes incoming speech and generates its own on the same 12.5 Hz
token clock — one step consumes 80 ms of what it hears and produces 80 ms of
what it says. Because the agent is speaking while it is still listening, there is
no turn to detect and no barge-in to implement: interruptions, overlaps, and
rapid turn-taking fall out of the model. The adapter never inspects the audio
and never decides whose turn it is.

What the adapter does own is clocks and rates:

- **Rate conversion.** Reactor's WebRTC transport carries mono 16-bit PCM at
  48 kHz in both directions; Mimi works at 24 kHz float32. The ratio is exactly
  two, so `personaplex_audio.py` decimates on the way in and interpolates on the
  way out, through one linear-phase FIR applied with retained history so the
  filter stays continuous across frame boundaries. Restarting it every frame
  would stamp a discontinuity into the audio 12.5 times a second. Fixed cost:
  0.65 ms of group delay per leg, which does not accumulate.
- **Pacing.** The microphone read is the clock. It cannot return faster than
  real time, so the loop needs no rate limiter and the agent's speech leaves at
  exactly the rate the participant's arrives. `fps` is pinned to Mimi's 12.5 Hz
  rather than left adaptive: an 80 ms frame is 80 ms of speech however long it
  took to compute, and pacing it to the measured step time would resample the
  agent's voice into a pitch shift.
- **Framing.** Inbound blocks are whatever the transport decoded — 10 ms each in
  practice — and never line up with an 80 ms frame. The remainder is carried
  between turns so no sample is dropped or duplicated at a block boundary.

The playout queue is bounded at four frames, or 320 ms. The loop is mic-paced so
it rarely fills; what the bound buys is room to absorb a step that overruns its
80 ms budget without the client hearing a gap. Deeper would only add latency to
a conversation, which is the one thing that has to stay low.

If the model cannot keep up with real time, the inbound buffer drops its oldest
blocks rather than growing a queue — right for a conversation, but silent, so the
adapter logs `inbound audio is backing up` when it happens. A session that
reports it is a session where the agent is missing parts of what was said.

### Conditioning is a separate phase

A voice prompt and a role prompt are stepped through the language model before
it can generate anything, which takes seconds rather than milliseconds. So it is
not part of the per-frame loop: the adapter enters a conditioning phase on the
first step of a session, and again whenever a client sends `set_persona`,
`set_voice`, or `restart`.

While it runs, no agent speech is produced. `StateUpdate.conversation_pending`
is true throughout and `ConversationStarted` is broadcast at the end, so a
client can show the agent as preparing rather than as broken. On the way out,
queued playout is flushed and the microphone backlog is dropped: the agent
starts listening from that moment instead of answering speech from before the
persona changed.

`StateUpdate.conversation_index` increments per conversation, and `AgentText`
carries the index it came from, so a client can discard transcript belonging to
a persona the user has already replaced.

### One conversation per session

The model streams with a batch size of one, so a session holds exactly one
conversation. Reactor routes inbound media from every connection into the same
track, so two clients both sending microphone audio would interleave into one
corrupted stream. Additional clients are fine as **listeners** — outbound audio
and text reach all of them — but only one participant should publish a
microphone.

## Run with the Reactor CLI

This directory is a `reactor` workspace, described in
[Build your own model](https://docs.reactor.inc/deploy/overview). The host needs
the [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation),
Docker, the NVIDIA Container Toolkit, and a Blackwell NVIDIA GPU — the pinned
PyTorch is a CUDA 12.8 build, for the reason the next section gives. The 7B
checkpoint runs in bfloat16 and wants roughly 20 GB of VRAM.

The model repository is **gated**: accept its licence on Hugging Face and
forward a read token without putting its value on the Docker command line.

```sh
cd models/personaplex
export HF_TOKEN=hf_your_read_token
reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container; `--gpus all` exposes every GPU. `reactor run` reuses the built image
and serves on `http://localhost:8080`. Rebuild after changing code or
dependencies. To use a different port:

```sh
reactor build && reactor run --gpus device=0 -e HF_TOKEN --port 18011
```

The first start downloads roughly 20 GB and captures CUDA graphs, so it takes a
while; later starts reuse both. Once it is ready:

```sh
curl -s localhost:8080/health
curl -s localhost:8080/schema
curl -s -X POST localhost:8080/start_session \
  -H 'content-type: application/json' -d '{}'
```

A WebRTC client publishes a microphone on `mic` and subscribes to `voice`.
Connect using a client built with the
[Reactor SDK](https://docs.reactor.inc/sdk-reference/using-the-sdk).

### Why the PyTorch pin is overridden

The manifest requests one `NVIDIA_B200`. A B200 is `sm_100`, and the first
PyTorch with Blackwell support is
[2.7, with CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/) — while
PersonaPlex declares `torch >= 2.2, < 2.5`. The two are mutually exclusive, and
not softly: a cu124 build carries no `sm_100` kernel image, so it does not run
slowly on a B200, it fails at the first CUDA op.

So the pin is overridden. `requirements.txt` installs `torch==2.9.1+cu128` —
2.9 rather than 2.7 because Blackwell support had a few releases to settle, and
because `alayaworld` in this repository already serves B200 on that line.

The declared ceiling looks conservative rather than load-bearing. PersonaPlex
touches only mainstream PyTorch, and its whole inference path — Mimi streaming
encode/decode, `LMGen.step`, the `.wav` voice-prompt path, and
`step_system_prompts` — runs unmodified on torch 2.9.1, and on 2.13 with numpy
2.5 besides. The one relevant default that changed since upstream's pin is
`torch.load`'s `weights_only`, flipped to `True` in torch 2.6; the voice `.pt`
files are a dict of tensors, which the strict loader accepts.

What blocks the override is packaging, not the code: pip reads the declared
metadata and refuses the install outright with `ResolutionImpossible`. Hence the
two passes in the `Dockerfile` — the runtime, PyTorch, and everything
PersonaPlex declares resolve as one set, then the model code lands `--no-deps`
on top. `requirements-personaplex.txt` carries that reasoning next to the pin.

Two consequences worth knowing. The build loses its single-pass guarantee: a
future conflict between the runtime and PersonaPlex will surface at model load
rather than during the build, so `pip check` inside the image is the thing to
read — it should report the deliberate `torch<2.5` complaint and nothing else.
And the weights run against a PyTorch upstream has not tested them on.

**Running on an H100 instead** is the conservative alternative: set
`gpu.type: NVIDIA_H100`, change the index to `.../whl/cu124` and the pin to
`torch==2.4.1+cu124`, and the two passes collapse back into one with
`moshi-personaplex` moved into `requirements.txt`. That configuration honours
every declared bound. It is not the default only because the hardware here is
Blackwell.

Either way the model is latency-bound rather than throughput-bound — a 7B model
stepping at 12.5 Hz — and one conversation uses about 20 GB. The GPU choice
buys session density, not a faster conversation.

## Model assets

The CLI bind-mounts `runtime.weights_path` from `reactor.yaml` and exports it as
`REACTOR_WEIGHTS_PATH`. The checked-in default keeps everything under
`~/.cache/reactor_registry/personaplex`, outside both the image and this Git
checkout, so rebuilds retain it. Nothing large is baked into a layer.

On first load the adapter downloads four files at the pinned revision — the 7B
language model, the Mimi codec checkpoint, the SentencePiece tokenizer, and the
voice archive — then unpacks the voices once, guarded by a marker recording the
revision they came from. An interrupted download resumes on the next start; a
revision bump re-unpacks rather than silently mixing two sets of voices.

`assets.revision` in `personaplex.yaml` is an immutable commit rather than
`main`, so every build installs the same weights. To relocate everything, change
only `runtime.weights_path`.

## Controls

Every command returns `StateUpdate`, a complete snapshot of the conversation's
settings, and a joining client receives the same message — so clients render
their controls from one payload instead of reconstructing state from partial
updates.

Two settings are applied live, on the next step:

- `set_audio_temperature` — sampling temperature for the agent's speech, 0 to 2.
- `set_text_temperature` — sampling temperature for the agent's text, 0 to 2.

Two condition the model, so they start a fresh conversation:

- `set_persona` — the role prompt: who the agent is, its background, and the
  scenario. The adapter wraps it in the `<system>` tags the model was trained to
  read. An empty persona leaves the agent unconditioned by any role.
- `set_voice` — the voice prompt name. Names are read off the unpacked archive
  rather than hard-coded, so `StateUpdate.voices` is the list this deployment
  actually carries and an unknown name is rejected with `unknown_voice` naming
  the valid ones. Upstream ships natural (`NATF*`, `NATM*`) and varied
  (`VARF*`, `VARM*`) sets.

And one restarts without changing anything:

- `restart` — a fresh conversation under the current persona and voice,
  discarding everything said so far.

The model sends back:

- `AgentText` — one fragment of the agent's own speech as text, alongside every
  frame it generates. Fragments carry their leading spaces, so concatenating
  them in arrival order reproduces the transcript with no separator. Text
  arrives slightly ahead of the audio that renders it.
- `ConversationStarted` — conditioning is done and the floor is open.
- `StateUpdate` — the snapshot described above.

Client-settable defaults live on `PersonaPlexState` in `personaplex_types.py`,
not in `personaplex.yaml`, so the published schema and the running model cannot
disagree about them. `personaplex.yaml` holds only what a client cannot choose:
where the assets live, the device, the top-k cutoffs, and the seed.

## Notes on the code

- **No recording.** Runtime's recorder muxes around a video track and disables
  itself for a model that emits none, so `reactor.yaml` carries no `recording:`
  block. Enabling it on this audio-only model would log a warning and record
  nothing.
- **The persona field is moderated; the voice name is not.** A persona is
  free text a client authored, so it carries Runtime's default moderation
  eligibility. A voice name is an identifier chosen from a published list, so it
  opts out — moderating `NATF0` would spend a check on nothing.
- **PersonaPlex's dependency list is restated by hand, plus two it omits.**
  Installing it `--no-deps` means pip never reads its metadata, so
  `requirements.txt` carries that list at the bounds upstream declares. Two more
  are there that upstream does not declare at all: `pyloudnorm`, imported inside
  `normalize_audio` on the `.wav` voice-prompt path, and `accelerate`, inside
  the `cpu_offload` branch of `get_moshi_lm`. Both sit behind a specific path,
  so without them the failure is a `ModuleNotFoundError` mid-session rather than
  at build. They are what make a reference-audio `.wav` and the
  `cpu_offload: true` knob actually work.
- **Eight codebooks are decoded, not sixteen.** `get_moshi_lm` builds the
  language model with `dep_q = 16`: eight codebooks for the agent's own speech
  and eight more for its prediction of the participant's stream. Only the
  agent's half is audio to send, and its width is exactly the number of
  codebooks `get_mimi` configures, so the adapter reads it from Mimi rather than
  halving `dep_q` or hard-coding `1:9` the way upstream does.
- **One Mimi, not two.** Upstream's `server.py` and `offline.py` both construct
  a second `MimiModel` and run `encode`/`decode` on it in lockstep with the
  first, discarding every result; it never reaches the language model or the
  output. This adapter omits it, which halves Mimi's cost per step on the path
  that has to stay under 80 ms.
- **`BOS` and `EOS` are kept out of the transcript.** Upstream filters only
  `EPAD` and `PAD`, so its client also receives raw `<s>` and `</s>` pieces.
  Those are structural markers rather than speech, so they are dropped here.
  This affects the transcript only, never the audio.
- **`<system>` closes with `<system>`.** Not a typo carried over by accident:
  upstream wraps the role prompt in `<system> … <system>`, and the tokenizer was
  trained that way, so a well-formed closing tag would not match what the model
  expects.
- **Once-per-session work runs in `@session_started`, not `@connected`.** A
  session closed server-side is torn down without a per-client disconnect, so
  anything counted across connections would not return to its starting value for
  the next session.

## Licences

The adapter follows this repository's Apache-2.0 licence. PersonaPlex's code is
MIT; its weights are governed by the
[NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/personaplex-7b-v1),
which you accept on Hugging Face before the first download.
