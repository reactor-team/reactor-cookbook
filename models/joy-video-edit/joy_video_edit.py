# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""JoyAI-Video-Edit, the application half.

The :class:`ReactorApp` the runtime drives. It declares the client contract
(``joy_video_edit_types.py``: the state, the tracks, the messages), owns
``process_input`` and ``process_output``, and holds the model half from
``joy_video_edit_model.py`` under ``self.engine``. The two files meet on two
dataclasses: the app builds a :class:`JoyVideoEditInput` from the client's
state and the ``camera`` track, and reads a :class:`JoyVideoEditResult` back.
It never reads an attribute of the model; what it needs to know about a step
rides on the result.

Real-time instruction-guided video editing: the client sets an edit
instruction (and optionally a reference image), calls ``start``, and streams
``camera``; edited frames stream back on ``main_video`` chunk by chunk. A run
is one ``start``; ``reset`` ends it, as does the last client leaving.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from joy_video_edit_model import (
    CHUNK_FRAMES,
    ChunkFailed,
    JoyVideoEditInput,
    JoyVideoEditModel,
    JoyVideoEditResult,
    RunConditioning,
)
from joy_video_edit_types import (
    ChunkComplete,
    CommandError,
    GenerationComplete,
    GenerationReset,
    GenerationStarted,
    JoyVideoEditMedia,
    JoyVideoEditOutput,
    JoyVideoEditState,
    PromptAccepted,
    ReferenceImageAccepted,
    SessionState,
)
from PIL import Image as PILImage
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    InputField,
    ReactorApp,
    ReadMode,
    StepOutcome,
    UploadedFile,
    connected,
    disconnected,
    event,
    get_logger,
    get_weights_path,
    session_ended,
)

logger = get_logger(__name__)

# Input accounting is logged every this many chunks, and when a run ends.
_STATS_EVERY_CHUNKS = 24


@dataclass
class _InputStats:
    """Camera frames that arrived, were consumed by chunks, were dropped, and were emitted.

    Rates are taken over the span from the run's first frame read to its last chunk
    emitted, so idle time before and after is excluded.
    """

    t0: float | None = None
    t1: float | None = None
    rx0: int = 0
    rx1: int = 0
    consumed: int = 0
    dropped: int = 0
    emitted: int = 0


class JoyVideoEdit(ReactorApp):
    """Real-time instruction-guided video editor.

    Stream video on `camera`, set an edit instruction with `set_prompt` (and
    optionally a reference image with `set_reference_image`), then call `start`:
    the edited video streams back on `main_video` at 1248x720, chunk by chunk.
    One client per session.
    """

    media: JoyVideoEditMedia
    state: JoyVideoEditState

    # Output buffering: one chunk of frames between the model and the wire.  Each
    # extra buffered frame is added output latency; start-up jitter is removed by
    # the load-time warm-up, not by queue depth.
    buffer_size = CHUNK_FRAMES

    def load(self, config_path: Path | None) -> None:
        """Read the application's own knobs, then construct and load the model half."""
        config = _read_config(config_path)
        # How each chunk reads the camera: latest (newest frames, older backlog dropped) or
        # fifo (arrival order, no loss while a chunk computes, backlog beyond
        # camera_backlog_chunks chunks dropped).
        self._camera_read = str(config.get("camera_read", "fifo")).lower()
        if self._camera_read not in ("fifo", "latest"):
            raise ValueError(f"camera_read must be fifo or latest, got {self._camera_read!r}")
        self._camera_backlog_frames = max(1, int(config.get("camera_backlog_chunks", 3))) * CHUNK_FRAMES
        self._input_stats = _InputStats()
        self._failures_in_a_row = 0

        self.engine = JoyVideoEditModel()
        self.engine.load(config_path, get_weights_path())

    # ---------------------------------------------------------------------------
    # The step
    # ---------------------------------------------------------------------------

    async def process_input(self) -> JoyVideoEditInput:
        """Refuse until `start` and until the camera has the frames the model asked for."""
        if not self.state._started:
            raise ApplicationError("not started")
        if self.state._conditioning is None:
            # The run's conditioning is fixed from here to the run's end.
            self.state._conditioning = RunConditioning(
                prompt=self.state.prompt,
                reference=self.state._reference,
                seed=int(self.state.seed),
            )
        frames = self._read_camera(self.state._frames_wanted)
        if frames is None:
            raise ApplicationError(f"waiting for {self.state._frames_wanted} camera frames")
        # The conditioning rides on the step input only until the model reports that it
        # opened this run; every other step carries the id alone.
        new_run = self.state._run_id != self.state._applied_run_id
        return JoyVideoEditInput(
            frames=frames,
            run_id=self.state._run_id,
            conditioning=self.state._conditioning if new_run else None,
        )

    def generate(self, input: JoyVideoEditInput) -> JoyVideoEditResult:
        """One step of the model. The model half does the work."""
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> JoyVideoEditOutput | None:
        """Report the chunk and hand its frames to `main_video`; recover a chunk that failed once."""
        if isinstance(outcome.error, ChunkFailed):
            return self._recover(outcome.error)
        if outcome.error is not None:
            # Any other model error is a bug in how the step was built, or a failure a
            # fresh session does not repair: end the session loudly.
            raise outcome.error
        self._failures_in_a_row = 0
        result: JoyVideoEditResult = outcome.result
        self.state._applied_run_id = result.run_id
        self.state._frames_wanted = result.frames_wanted

        emitted = int(result.frames.shape[0])
        await self.send(ChunkComplete(
            chunk_index=self.state._chunk_index,
            frames_emitted=emitted,
            elapsed_ms=round(outcome.elapsed * 1000.0, 1),
        ))
        self.state._chunk_index += 1
        self.state._total_frames += emitted

        stats = self._input_stats
        stats.emitted += emitted
        stats.t1, stats.rx1 = time.perf_counter(), self.media.camera.total_received
        if self.state._chunk_index % _STATS_EVERY_CHUNKS == 0:
            self._log_input_stats()
        return JoyVideoEditOutput(main_video=result.frames)

    def _recover(self, error: ChunkFailed) -> None:
        """Replace the run's session after a failed chunk, keeping the run's conditioning."""
        self._failures_in_a_row += 1
        if self._failures_in_a_row > 1:
            # The chunk failed again on a fresh session: the fault is not in the session.
            raise error
        logger.error(
            "chunk failed; starting a new session", chunk_index=self.state._chunk_index, error=repr(error)
        )
        self.engine.reset()
        self.state._applied_run_id = None
        self.state._frames_wanted = 1
        return None

    def _read_camera(self, n: int) -> list[np.ndarray] | None:
        """Take the next chunk's `n` frames off `camera`, or None while fewer are queued.

        latest: the newest `n`, older backlog dropped.  fifo: the oldest `n`, in arrival
        order, so no frame is lost while a chunk computes; a backlog of more than
        camera_backlog_chunks chunks beyond them is dropped (oldest first) to bound
        latency.  The first read of a run always takes the newest frames.
        """
        camera = self.media.camera
        available = camera.available
        if available < n:
            return None
        stats = self._input_stats
        if stats.t0 is None:
            stats.t0 = time.perf_counter()
            stats.rx0 = camera.total_received - available
        if self._camera_read == "latest" or stats.consumed == 0:
            frames = camera.try_read(n, mode=ReadMode.LATEST)
            stats.dropped += available - n
        else:
            frames = camera.try_read(n, mode=ReadMode.FIFO)
            excess = camera.available - self._camera_backlog_frames
            if excess > 0 and camera.try_read(excess, mode=ReadMode.FIFO) is not None:
                stats.dropped += excess
                logger.info(
                    "camera backlog trimmed",
                    frames_behind=excess + self._camera_backlog_frames,
                    dropped=excess,
                )
        if frames is None:
            return None
        stats.consumed += n
        # (H, W, 3) uint8 RGB arrays, as the session takes them: no PIL round trip.
        return [frame.data for frame in frames]

    def _log_input_stats(self) -> None:
        stats = self._input_stats
        if stats.t0 is None or stats.t1 is None:
            return
        dt = max(1e-6, stats.t1 - stats.t0)
        arrived = stats.rx1 - stats.rx0
        logger.info(
            "camera input",
            seconds=round(dt, 1),
            arrived=arrived,
            arrived_fps=round(arrived / dt, 2),
            consumed=stats.consumed,
            consumed_fps=round(stats.consumed / dt, 2),
            dropped=stats.dropped,
            chunks=self.state._chunk_index,
            chunks_per_s=round(self.state._chunk_index / dt, 2),
            emitted=stats.emitted,
            emitted_fps=round(stats.emitted / dt, 2),
            camera_read=self._camera_read,
        )

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    @connected
    async def on_connect(self) -> None:
        logger.info("on_connect")
        await self._send_state()

    @disconnected
    async def on_disconnect(self) -> None:
        """The last client leaving ends the run, as a new `start` would be needed to see it."""
        logger.info("on_disconnect")
        if self.connected.is_set() or not self.state._started:
            return
        await self._end_run()
        await self._send_state()

    @session_ended
    def on_session_ended(self) -> None:
        """A session end is the application's cue to reset the model."""
        self.engine.reset()

    # ---------------------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------------------

    @event(name="set_prompt", description=(
        "Set the edit instruction, in natural language (for example 'Turn the video into "
        "watercolor style'). Accepted any time. `start` reads it and it stays fixed for that "
        "run, so a change while generating applies from the next `start`. Emits "
        "`prompt_accepted` and broadcasts `session_state`."
    ))
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=2000,
            description=(
                "Edit instruction, up to 2000 characters. Replaces the stored instruction; "
                "an empty string clears it."
            ),
            moderate=True,
        ),
    ) -> PromptAccepted:
        logger.info("set_prompt", prompt=prompt[:80])
        self.state.prompt = prompt
        await self._send_state()
        return PromptAccepted(prompt=prompt)

    @event(name="set_reference_image", description=(
        "Provide an optional reference image whose appearance guides the edit (for example "
        "the object to add or the look to match). Accepted only while not generating: send "
        "it before `start`, or after `reset`. It stays loaded for later runs until another "
        "upload replaces it. Emits `reference_image_accepted` and broadcasts "
        "`session_state`, or `command_error` while generating."
    ))
    async def set_reference_image(
        self,
        reference_image: UploadedFile = InputField(
            default=None,
            description=(
                "Reference image uploaded via the Reactor file-upload protocol. PNG or JPEG. "
                "It is fitted, aspect kept, into the nearest of a square, 4:3, or 3:4 frame "
                "and padded with grey, never cropped."
            ),
            moderate=True,
        ),
        client: ClientInfo = None,
    ) -> ReferenceImageAccepted:
        logger.info("set_reference_image", name=reference_image.name, bytes=reference_image.size)
        if self.state._started:
            if client is not None:
                await client.send(CommandError(
                    command="set_reference_image",
                    reason="Generation is active; send set_reference_image before start.",
                ))
            return
        pil = PILImage.open(io.BytesIO(reference_image.data)).convert("RGB")
        self.state.reference_image = reference_image
        self.state._reference = np.asarray(pil)
        await self._send_state()
        return ReferenceImageAccepted(width=pil.width, height=pil.height)

    @event(name="start", description=(
        "Begin a run: edit the `camera` track and stream the result on `main_video`. "
        "Accepted when not already generating. Fixes the current instruction, reference "
        "image, and seed for the run. Frames flow once `camera` delivers video, with a "
        "`chunk_complete` per chunk. Emits `generation_started` and broadcasts "
        "`session_state`, or `command_error` if already generating."
    ))
    async def start(self, client: ClientInfo = None) -> GenerationStarted:
        logger.info("start")
        if self.state._started:
            if client is not None:
                await client.send(CommandError(command="start", reason="Already generating."))
            return
        # A new run id: the model opens a new run on the next step that has camera frames.
        self.state._started = True
        self.state._run_id += 1
        self.state._conditioning = None
        self.state._frames_wanted = 1
        self.state._chunk_index = 0
        self.state._total_frames = 0
        self._input_stats = _InputStats()
        self._failures_in_a_row = 0
        await self._send_state()
        return GenerationStarted(
            prompt=self.state.prompt,
            has_reference_image=self.state._reference is not None,
            seed=self.state.seed,
        )

    @event(name="reset", description=(
        "Stop generating and return to waiting. Accepted any time. The instruction, "
        "reference image, and seed are kept, so the next `start` begins a new run with them. "
        "While generating, emits `generation_reset` and then `generation_complete`; always "
        "emits `session_state`."
    ))
    async def reset(self) -> None:
        logger.info("reset")
        if self.state._started:
            await self.send(GenerationReset(reason="client requested"))
            await self._end_run()
        await self._send_state()

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    async def _end_run(self) -> None:
        """End the live run: release the model's session, report the run's totals, and wait."""
        self.engine.reset()
        self._log_input_stats()
        logger.info("run ended", chunks=self.state._chunk_index, frames=self.state._total_frames)
        self.state._started = False
        self.state._conditioning = None
        self.state._applied_run_id = None
        self.state._frames_wanted = 1
        await self.send(GenerationComplete(
            total_chunks=self.state._chunk_index,
            total_frames=self.state._total_frames,
        ))

    async def _send_state(self) -> None:
        await self.send(SessionState(
            started=self.state._started,
            prompt=self.state.prompt,
            has_reference_image=self.state._reference is not None,
            seed=self.state.seed,
        ))


def _read_config(config_path: Path | None) -> dict[str, Any]:
    """Parse the application's knobs out of ``config.yml``; no path means the defaults."""
    if config_path is None:
        return {}
    document = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    return document
