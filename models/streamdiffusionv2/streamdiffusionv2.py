"""StreamDiffusionV2, the application half.

Restyles a live video from a text prompt, chunk by chunk. The client sends its video on the
`video_in` track (a webcam, or a file played in the page), sets a prompt, a seed and a starting
noise strength, and sends `start`; each step rewrites the next 4 input frames (5 on the first)
at 480x832, and the output trails the input by one chunk (the source's "stream batch").

| Input | Modality | Role | Kind | Command |
|---|---|---|---|---|
| video to rewrite | video | the video to rewrite | live control (by nature) | `video_in` track |
| prompt | text | the style / scene description | live control | `set_prompt` |
| seed | number | randomness | session setup | `set_seed` |
| noise strength | number | how far the output may depart from the input, 0..1 | session setup | `set_noise_scale` |

Every command answers with `state_update`; `chunk_complete` precedes every chunk's frames.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import cast

import numpy as np
from transformers import AutoTokenizer
from streamdiffusionv2_assets import prepare_assets, read_config
from streamdiffusionv2_model import StreamDiffusionV2Model, StreamInput, StreamResult
from streamdiffusionv2_types import (
    CAPABILITIES,
    PROMPT_TOKEN_BUDGET,
    SEED_MAX,
    TRAINED_FPS,
    ChunkComplete,
    GenerationComplete,
    GenerationStarted,
    StateUpdate,
    StreamDiffusionV2Media,
    StreamDiffusionV2Output,
    StreamDiffusionV2State,
)

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputBuffer,
    InputField,
    ReactorApp,
    ReadMode,
    StepOutcome,
    TrackPayload,
    connected,
    event,
    get_logger,
    session_ended,
    session_started,
)

logger = get_logger(__name__)

FIRST_CHUNK = StreamDiffusionV2Model.FIRST_CHUNK_FRAMES
CHUNK = StreamDiffusionV2Model.CHUNK_FRAMES
# Frames kept waiting beyond the next chunk before the oldest are dropped, so a client sending
# faster than the model rewrites never builds up latency.
MAX_BACKLOG = 2 * CHUNK


class StreamDiffusionV2(ReactorApp):
    """StreamDiffusionV2 1.3B: restyle a live video from a text prompt, 4 frames at a time."""

    media: StreamDiffusionV2Media
    state: StreamDiffusionV2State
    fps = TRAINED_FPS

    def load(self, config_path: Path | None) -> None:
        """Fetch the pinned source and public weights, then load the model half once and warm it up."""
        config = read_config(config_path)
        prepare_assets(config)
        self.engine = StreamDiffusionV2Model()
        self.engine.load(
            config_path=str(config.config_path),
            source_path=str(config.source_path),
            weights_root=str(config.weights_root),
        )
        # The umT5 tokenizer the text encoder uses counts the prompt budget, special tokens included.
        self.tokenizer = AutoTokenizer.from_pretrained(str(config.wan.target / "google" / "umt5-xxl"))

    # -- the step ---------------------------------------------------------------

    async def process_input(self) -> StreamInput:
        """Refuse unless running with a full chunk of input; otherwise hand the model its chunk."""
        s = self.state
        self._drain_video()
        if s._phase != "running":
            raise ApplicationError(s._phase)
        pending = cast(deque, s._pending)
        need = CHUNK if s._started else FIRST_CHUNK
        if len(pending) < need:
            raise ApplicationError("waiting_for_video")
        taken = [pending.popleft() for _ in range(need)]
        # WebRTC rescales an inbound track as bandwidth moves, so one chunk can mix frame sizes.
        # Same-size frames go as one array; mixed sizes go as a list, and the model resizes each.
        frames = np.stack(taken) if len({f.shape for f in taken}) == 1 else taken
        return StreamInput(
            frames=frames, prompt=cast(str, s._prompt), seed=cast(int, s._seed),
            noise_scale=cast(float, s._noise_scale), run_id=s._run_id,
        )

    def generate(self, input: StreamInput) -> StreamResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> StreamDiffusionV2Output | None:
        """Report the chunk, then send its frames."""
        if outcome.error is not None:
            # process_input only builds chunks of the size the model takes, so any error here is a
            # bug or a GPU failure, which a reset does not repair: the session ends with it.
            raise outcome.error
        r: StreamResult = outcome.result
        s = self.state
        if r.run_id != s._run_id:
            return None  # a chunk of a run that reset replaced
        first = r.index == 0
        prompt_changed = not first and r.prompt != s._applied_prompt
        if prompt_changed:
            logger.info("prompt applied", chunk=r.index)
        s._started = True
        s._chunk_index = r.index
        s._applied_prompt, s._applied_seed, s._applied_noise_scale = r.prompt, s._seed, r.noise_scale
        if prompt_changed:
            await self.send(self._snapshot())
        if first:
            await self.send(GenerationStarted())
            await self.send(self._snapshot())
        n = 0 if r.frames is None else int(r.frames.shape[0])
        await self.send(
            ChunkComplete(
                index=r.index, frames_out=n,
                generate_seconds=round(float(outcome.elapsed), 4), noise_scale=round(r.noise_scale, 4),
            )
        )
        if r.frames is None:
            return None
        return StreamDiffusionV2Output(main_video=TrackPayload(r.frames, metadata=[{"chunk": r.index}] * n))

    # -- commands ---------------------------------------------------------------

    @event(
        name="set_prompt",
        description=(
            "Set the text the model restyles the video toward. Live control: valid at any time; sent "
            "before `start` it is the starting prompt, sent while running it applies at the next chunk "
            "(about 0.25 s) without restarting. Leading and trailing spaces are stripped; an empty prompt "
            f"is a value. Answers with `state_update`. Fails with `prompt_too_long` over {PROMPT_TOKEN_BUDGET} umT5 tokens."
        ),
    )
    def set_prompt(
        self, prompt: str = InputField(description="The prompt, any length up to 512 tokens.", moderate=True)
    ) -> StateUpdate:
        text = prompt.strip()
        count = len(self.tokenizer(text, add_special_tokens=True)["input_ids"])
        if count > PROMPT_TOKEN_BUDGET:
            raise CommandError("prompt_too_long", f"{count} tokens; the limit is {PROMPT_TOKEN_BUDGET}.")
        self.state._prompt = text
        return self._snapshot()

    @event(
        name="set_seed",
        description=(
            "Set the run's seed. Session setup: valid while idle; sent while running it fails with "
            f"`locked_while_running`. 0..{SEED_MAX}, or -1 for a random seed drawn now. Answers with "
            "`state_update`. Fails with `invalid_seed` outside that range."
        ),
    )
    def set_seed(self, seed: int = InputField(description="The seed, or -1 for random.")) -> StateUpdate:
        self._locked("seed")
        if seed == -1:
            seed = random.randint(0, SEED_MAX)
        if not 0 <= seed <= SEED_MAX:
            raise CommandError("invalid_seed", f"seed must be 0..{SEED_MAX} or -1.")
        self.state._seed = seed
        return self._snapshot()

    @event(
        name="set_noise_scale",
        description=(
            "Set the starting noise strength, 0.0..1.0: how far the output may depart from the input "
            "video (the source's example uses 0.8). The model adapts it at every chunk to how much the "
            "input moves, as the source does. Session setup: valid while idle; sent while running it "
            "fails with `locked_while_running`. Answers with `state_update`. Fails with "
            "`invalid_noise_scale` outside 0..1."
        ),
    )
    def set_noise_scale(
        self, noise_scale: float = InputField(description="Starting noise strength, 0.0..1.0.")
    ) -> StateUpdate:
        self._locked("noise_scale")
        if not 0.0 <= noise_scale <= 1.0:
            raise CommandError("invalid_noise_scale", "noise_scale must be between 0 and 1.")
        self.state._noise_scale = float(noise_scale)
        return self._snapshot()

    @event(
        name="start",
        description=(
            "Start rewriting the video: idle -> running. The first chunk runs once 5 frames have "
            "arrived on `video_in`. Answers with `state_update`; `generation_started` follows with the "
            "first chunk. Fails with `missing_prompt`, `missing_seed` or `missing_noise_scale` when that "
            "input was never sent, and `already_running` when not idle."
        ),
    )
    def start(self) -> StateUpdate:
        s = self.state
        if s._phase != "idle":
            raise CommandError("already_running", f"the model is {s._phase}.")
        if s._prompt is None:
            raise CommandError("missing_prompt", "set_prompt before start.")
        if s._seed is None:
            raise CommandError("missing_seed", "set_seed before start.")
        if s._noise_scale is None:
            raise CommandError("missing_noise_scale", "set_noise_scale before start.")
        s._run_id += 1
        s._started = False
        s._chunk_index = -1
        cast(deque, s._pending).clear()
        s._last_capture_us = None
        s._phase = "running"
        logger.info("run started", run_id=s._run_id, seed=s._seed, noise_scale=s._noise_scale)
        return self._snapshot()

    @event(
        name="pause",
        description=(
            "Hold on the last frame; input frames arriving meanwhile are dropped. Answers with "
            "`state_update`. Fails with `not_running` unless running."
        ),
    )
    def pause(self) -> StateUpdate:
        if self.state._phase != "running":
            raise CommandError("not_running", f"the model is {self.state._phase}.")
        self.state._phase = "paused"
        return self._snapshot()

    @event(
        name="resume",
        description=(
            "Continue the run from where it paused; a prompt sent while paused applies at the next "
            "chunk. Answers with `state_update`. Fails with `not_paused` unless paused."
        ),
    )
    def resume(self) -> StateUpdate:
        s = self.state
        if s._phase != "paused":
            raise CommandError("not_paused", f"the model is {s._phase}.")
        cast(deque, s._pending).clear()
        s._last_capture_us = None
        s._phase = "running"
        return self._snapshot()

    @event(
        name="reset",
        description=(
            "End the run and clear every input (prompt, seed, noise strength); allowed in any state. "
            "Emits `generation_complete` when a run was going. To run the same conditions again, set "
            "them again. Answers with `state_update`."
        ),
    )
    async def reset(self) -> StateUpdate:
        was_running = self.state._phase != "idle"
        self.engine.reset()
        self._clear()
        self.output.flush(hold=True)
        if was_running:
            await self.send(GenerationComplete())
        return self._snapshot()

    # -- lifecycle --------------------------------------------------------------

    @session_started
    def on_session_started(self) -> None:
        self.engine.reset()
        self._clear()

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        await client.send(self._snapshot())

    @session_ended
    def on_session_ended(self) -> None:
        self.engine.reset()
        self.output.flush()

    # -- helpers ----------------------------------------------------------------

    def _clear(self) -> None:
        """Forget every input and the run, keeping the run counter so old chunks stay recognisable."""
        s = self.state
        s._phase = "idle"
        s._prompt = s._seed = s._noise_scale = None
        s._started = False
        s._applied_prompt = s._applied_seed = s._applied_noise_scale = None
        s._pending = deque()
        s._last_capture_us = None
        s._chunk_index = -1

    def _locked(self, what: str) -> None:
        if self.state._phase != "idle":
            raise CommandError("locked_while_running", f"{what} is session setup; reset first to change it.")

    def _drain_video(self) -> None:
        """Move arrived frames into the pending queue, resampled to 16 fps by capture time."""
        s = self.state
        pending = cast(deque, s._pending)
        buf = cast(InputBuffer, self.media.video_in)
        while (got := buf.try_read(1, mode=ReadMode.FIFO)) is not None:
            frame = got[0]
            if frame.data.ndim != 3 or frame.data.shape[2] != 3:
                continue
            stamp = frame.capture_time_us
            if stamp is not None and s._last_capture_us is not None:
                # Keep a frame once 1/16 s has passed since the last kept one (5 ms slack for jitter).
                if stamp - s._last_capture_us < 1_000_000 / TRAINED_FPS - 5_000:
                    continue
            if stamp is not None:
                s._last_capture_us = stamp
            pending.append(frame.data)
        need = CHUNK if s._started else FIRST_CHUNK
        while len(pending) > need + MAX_BACKLOG:
            pending.popleft()

    def _snapshot(self) -> StateUpdate:
        s = self.state
        return StateUpdate(
            state=s._phase,
            settings={
                "prompt": {"sent": s._prompt, "applied": s._applied_prompt},
                "seed": {"sent": s._seed, "applied": s._applied_seed},
                "noise_scale": {"sent": s._noise_scale, "applied": s._applied_noise_scale},
            },
            capabilities=CAPABILITIES,
            frames_waiting=len(s._pending or ()),
            chunk_index=s._chunk_index,
        )
