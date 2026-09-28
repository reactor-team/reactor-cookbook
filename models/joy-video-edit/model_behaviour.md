# JoyAI-Video-Edit: model behaviour

For client developers who drive the model without reading its code. The wire
contract (every command, field, and message, with descriptions) is also what
`python -m reactor_runtime.schema --path .` renders; this page adds the state
machine, the ordering, and working sequences.

## What it does

Real-time, instruction-guided video editing
([JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit),
[paper](https://arxiv.org/abs/2608.03974)). Given a live `camera` stream and a
natural-language instruction, it edits frames as they arrive, chunk by chunk,
without future frames and without a known video length. It supports style
transfer, local object edits, adding, removing or replacing a subject,
background changes, and edits guided by a reference image.

## Tracks

| Direction | Name | Description |
| --- | --- | --- |
| Inbound | `camera` | The video to edit: a webcam, a screen, or a video played in real time. Any size; frames are resized to 1248x720. |
| Outbound | `main_video` | The edited video at 1248x720, chunk by chunk. |

## States

```
connect  ->  session_state
   |
   v
+--------------------------------------------------------------+
| WAITING                                                      |
| Accepted: set_prompt, set_reference_image, set_seed, start,  |
|           reset                                              |
| Nothing on main_video.                                       |
+------------------------------+-------------------------------+
                               |  start
                               |  -> generation_started (reply), session_state
                               v
+--------------------------------------------------------------+
| GENERATING (a run)                                           |
| Accepted: set_prompt, set_seed (both apply from the next     |
|           start), reset                                      |
| Refused with command_error: start, set_reference_image       |
| Once camera has frames: chunk_complete, then that chunk's    |
| frames on main_video, for every chunk                        |
+------------------------------+-------------------------------+
                               |  reset
                               |  -> generation_reset, generation_complete, session_state
                               |  last client disconnects
                               |  -> generation_complete, session_state
                               v
                            WAITING
```

A run is one `start`. It fixes the instruction, the reference image, and the
seed until it ends; to change them, `reset`, set the new values, and `start`
again. `reset` keeps the instruction, reference image, and seed, so a bare
`reset` then `start` restarts the same edit.

## Commands (client to model)

| Command | Parameters | Accepted | Answer |
| --- | --- | --- | --- |
| `set_prompt` | `prompt: str`, up to 2000 characters; `""` clears it | any time | replies `prompt_accepted`; broadcasts `session_state` |
| `set_reference_image` | `reference_image`: an uploaded PNG or JPEG | while waiting | replies `reference_image_accepted`; broadcasts `session_state`. While generating: `command_error` to the sender |
| `set_seed` | `seed: int`, 0 or more (default 42) | any time | nothing; `start` reads it |
| `start` | none | while waiting | replies `generation_started`; broadcasts `session_state`. While generating: `command_error` to the sender |
| `reset` | none | any time | while generating: broadcasts `generation_reset`, `generation_complete`, then `session_state`. While waiting: `session_state` only |

The prompt and the reference image are free-form client content and are
marked for moderation; a deployment that moderates screens them.

The reference image is fitted, aspect kept, into the nearest of a square,
4:3, or 3:4 frame and padded with grey; it is never cropped. It stays loaded
across `reset` until another upload replaces it.

## Messages (model to client)

A reply goes to the client that sent the command, both as the awaited result
of `sendCommand` and as a `message` event on that client. Everything else is
broadcast to every connected client.

| Message | Sent | Fields |
| --- | --- | --- |
| `session_state` | on connect; after `set_prompt`, `set_reference_image`, `start`, `reset`; when a run ends | `started`, `prompt`, `has_reference_image`, `seed` |
| `prompt_accepted` | reply to `set_prompt` | `prompt` |
| `reference_image_accepted` | reply to `set_reference_image` | `width`, `height` of the upload as decoded |
| `generation_started` | reply to `start` | `prompt`, `has_reference_image`, `seed` of the run |
| `chunk_complete` | before each chunk's frames go out on `main_video` | `chunk_index` (0-based, per run), `frames_emitted`, `elapsed_ms` |
| `generation_reset` | `reset` while generating | `reason` (`client requested`) |
| `generation_complete` | when a run ends | `total_chunks`, `total_frames` of the run |
| `command_error` | to the sender of a refused command | `command`, `reason` |

## Timing

- **Chunks.** The first chunk of a run takes 1 camera frame and emits 1
  frame; every later chunk takes 8 and emits 8. Every 600 camera frames the
  model refreshes its context, and the chunk after that carries 1 frame again.
- **Speed.** On one NVIDIA B200 a steady 8-frame chunk takes about 0.2 s,
  about 40 edited frames per second. The model warms up every chunk shape
  when it loads, so the first chunk of a session is as fast as the rest.
- **Playout.** `main_video` plays each chunk at `frames_emitted` over
  `elapsed_ms`, so a slower chunk plays slower rather than freezing.
- **Latest input first.** Each chunk takes the newest camera frames and drops
  older queued ones, so the output trails the input by about one chunk. A
  camera faster than the model loses frames; one slower than the model waits
  for them.
- **Failures.** A chunk that fails is retried once on a fresh context of the
  same run, with no message. A second failure in a row ends the session with
  an error.

## Usage

With the JavaScript SDK (`@reactor-team/js-sdk` 3.x), after `connect()`:

### Edit from an instruction

```js
reactor.on("trackReceived", (name, track) => {
  if (name === "main_video") video.srcObject = new MediaStream([track]);
});
reactor.on("message", (message) => {
  if (message.type === "session_state") render(message.data);
  if (message.type === "chunk_complete") showLatency(message.data.elapsed_ms);
});

const [camera] = (await navigator.mediaDevices.getUserMedia({ video: true })).getVideoTracks();
await reactor.publishTrack("camera", camera);

await reactor.sendCommand("set_prompt", { prompt: "Turn the video into a watercolor painting" });
// -> { type: "prompt_accepted", ... }; session_state is broadcast
const started = await reactor.sendCommand("start");
// -> { type: "generation_started", data: { prompt, has_reference_image, seed } }
// chunk_complete per chunk, and main_video starts playing
```

### Edit with a reference image

Upload before `start`, or after `reset`:

```js
await reactor.sendCommand("set_prompt", { prompt: "Put the hat from the reference image on the person" });
const reference = await reactor.uploadFile(file);
await reactor.sendCommand("set_reference_image", { reference_image: reference });
// -> { type: "reference_image_accepted", data: { width, height } }
await reactor.sendCommand("start");
```

### Change the edit

```js
await reactor.sendCommand("reset");
// broadcast: generation_reset, generation_complete, session_state { started: false }
await reactor.sendCommand("set_prompt", { prompt: "Make everything sepia" });
await reactor.sendCommand("start");
```

A `set_prompt` during a run is stored and answered, but the running edit keeps
its instruction until the next `start`.

## Connections

One client per session is the intended use. Replies go to the sender and the
rest is broadcast, so a second client sees the state but cannot tell which
client started the run. When the last client disconnects, the run ends; a
client that reconnects in the same session finds the model waiting, with its
instruction, reference image, and seed kept.
