"""StreamForce, the application half.

Image-to-video with a force you steer while it plays. The client sets a first frame, a prompt
and a force, sends `start`, and the model streams 480x832 video, 12 frames per chunk (9 in the
first). The force and the prompt are live controls: a change applies from the next chunk the
model starts, which reaches the screen 3 chunks later. The first frame and the seed are session
setup. A run ends on its own after 4089 frames with `generation_complete`.

Every command answers with `state_update`; `chunk_complete` precedes every chunk's frames.
"""

from __future__ import annotations

import asyncio
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from streamforce_assets import prepare_assets, read_config
from streamforce_model import Force, StreamForceInput, StreamForceModel, StreamForceResult
from streamforce_types import (
    CAPABILITIES,
    PROMPT_TOKEN_BUDGET,
    ChunkComplete,
    GenerationComplete,
    GenerationStarted,
    StateUpdate,
    StreamForceOutput,
    StreamForceState,
    force_dict,
)

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    connected,
    event,
    session_ended,
    session_started,
)

MAX_UPLOAD_PIXELS = 4096 * 4096
UNDECODABLE = (OSError, ValueError, Image.DecompressionBombError)


class StreamForce(ReactorApp):
    """StreamForce: image-to-video under a point or wind force you change while it plays."""

    state: StreamForceState

    def load(self, config_path: Path | None) -> None:
        """Fetch the public assets, check the checkpoint, and load the model half once."""
        config = read_config(config_path)
        prepare_assets(config)
        self.engine = StreamForceModel()
        self.engine.load(
            config_path=str(config.config_path),
            source_path=str(config.source_path),
            wan_root=str(config.wan_root),
            checkpoint_path=str(config.checkpoint_path),
        )
        self.frames_per_session = self.engine.frames_per_session

    # -- the step -------------------------------------------------------------

    async def process_input(self) -> StreamForceInput:
        """Refuse unless a run is going; the first frame rides only on a run's first step."""
        s = self.state
        if s._phase != "running":
            raise ApplicationError(s._phase)
        new_run = s._run_id != s._applied_run_id
        return StreamForceInput(
            run_id=s._run_id,
            first_frame=s._first_frame if new_run else None,
            seed=s._run_seed,
            prompt=s._prompt,
            force=s._force,
        )

    def generate(self, input: StreamForceInput) -> StreamForceResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> StreamForceOutput | None:
        """Report what the chunk used, then send its frames; end the run at the maximum length."""
        if outcome.error is not None:
            # NotStarted cannot arrive: process_input sends the first frame on a run's first
            # step and start refuses without one. Anything else is a bug or a GPU failure,
            # which a reset does not repair, so the session ends with the error.
            raise outcome.error
        r: StreamForceResult = outcome.result
        s = self.state
        if r.run_id != s._run_id:
            return None  # a step of a run that reset or a new start replaced
        new_run = s._applied_run_id != r.run_id
        changed = new_run or r.prompt != s._applied_prompt or r.force != s._applied_force
        if new_run or r.prompt != s._applied_prompt:
            s._prompt_from_chunk = r.window
        if new_run or r.force != s._applied_force:
            s._force_from_chunk = r.window
        s._applied_run_id = r.run_id
        s._applied_seed, s._applied_prompt, s._applied_force = r.seed, r.prompt, r.force
        if changed:
            await self.send(self._snapshot())
        if r.frames is None:
            return None
        s._last_chunk = r.block
        s._frames_in_run += r.frames.shape[0]
        await self.send(ChunkComplete(index=r.block, frames=int(r.frames.shape[0])))
        output = StreamForceOutput(
            main_video=TrackPayload(r.frames, metadata=[{"chunk": r.block}] * r.frames.shape[0])
        )
        if r.finished:
            s._phase = "idle"
            s._applied_run_id = None
            self.engine.reset()
            await self.send(GenerationComplete(reason="maximum length reached", frames=s._frames_in_run))
            await self.send(self._snapshot())
        return output

    # -- lifecycle --------------------------------------------------------------

    @event(
        name="start",
        description=(
            "Start a run: idle to running. Needs a first frame, a prompt and a force. Answers with "
            "`state_update` and emits `generation_started`. Fails with `missing_first_frame`, "
            "`missing_prompt` or `missing_force` when one was never set, and `already_running` "
            "when a run is running or paused."
        ),
    )
    async def start(self) -> StateUpdate:
        s = self.state
        if s._phase != "idle":
            raise CommandError("already_running", "A run is already going; reset it first.")
        if s._first_frame is None:
            raise CommandError("missing_first_frame", "Set a first frame with set_first_frame before start.")
        if s._prompt is None:
            raise CommandError("missing_prompt", "Set a prompt with set_prompt before start (an empty prompt is fine).")
        if s._force is None:
            raise CommandError("missing_force", "Set a force with set_force before start.")
        s._run_id += 1
        s._phase = "running"
        s._last_chunk = -1
        s._frames_in_run = 0
        s._run_seed = s._seed if s._seed >= 0 else random.randrange(2**31)
        await self.send(GenerationStarted(seed=s._run_seed))
        return self._snapshot()

    @event(
        name="pause",
        description=(
            "Hold on the last frame: running to paused. Inputs sent while paused apply on resume. "
            "Answers with `state_update`. Fails with `not_running` when nothing is running."
        ),
    )
    def pause(self) -> StateUpdate:
        if self.state._phase != "running":
            raise CommandError("not_running", "Nothing is running.")
        self.state._phase = "paused"
        return self._snapshot()

    @event(
        name="resume",
        description=(
            "Continue from the last frame: paused to running. Answers with `state_update`. Fails "
            "with `not_paused` when the run is not paused."
        ),
    )
    def resume(self) -> StateUpdate:
        if self.state._phase != "paused":
            raise CommandError("not_paused", "The run is not paused.")
        self.state._phase = "running"
        return self._snapshot()

    @event(
        name="reset",
        description=(
            "End the run and clear every input (first frame, prompt, force, seed); valid in any "
            "state. To run the same conditions again, set them again. Answers with `state_update`."
        ),
    )
    def reset(self) -> StateUpdate:
        self._clear()
        self.output.flush()
        return self._snapshot()

    # -- inputs -----------------------------------------------------------------

    @event(
        name="set_first_frame",
        description=(
            "Session setup: the picture the video starts on. PNG, JPEG or WebP of any size; it is "
            "resized to cover 832x480 and centre-cropped when the run starts. Valid while idle. "
            "Answers with `state_update`. Fails with `locked_while_running` during a run, "
            "`unsupported_media` when the upload is not an image, and `undecodable_image` when it "
            "cannot be decoded."
        ),
    )
    async def set_first_frame(
        self,
        image: UploadedFile = InputField(description="The first frame (PNG, JPEG or WebP).", moderate=True),
    ) -> StateUpdate:
        self._locked("first frame")
        if not (image.mime_type or "").startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} is not an image.")
        try:
            rgb = await asyncio.to_thread(_decode, image.data)
        except UNDECODABLE as exc:
            raise CommandError("undecodable_image", "The file is not a decodable image.") from exc
        self.state._first_frame = rgb
        return self._snapshot()

    @event(
        name="set_prompt",
        description=(
            "Live control: the scene description. Sent before `start` it is the starting prompt; "
            "sent while running it applies from the next chunk the model starts, and the change "
            "shows over the following 4 chunks. Leading and trailing spaces are stripped; an empty "
            "prompt is a valid value. Answers with `state_update`. Fails with `prompt_too_long` "
            f"over {PROMPT_TOKEN_BUDGET} tokens."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(description="Scene description, in English.", moderate=True),
    ) -> StateUpdate:
        text = prompt.strip()
        count = await asyncio.to_thread(self.engine.prompt_tokens, text)
        if count > PROMPT_TOKEN_BUDGET:
            raise CommandError(
                "prompt_too_long", f"The prompt is {count} tokens; the model reads at most {PROMPT_TOKEN_BUDGET}."
            )
        self.state._prompt = text
        return self._snapshot()

    @event(
        name="set_force",
        description=(
            "Live control: the force acting on the scene. `point` pushes at (`x`, `y`); `wind` is "
            "a field over the whole frame and ignores `x` and `y`. Sent before `start` it is the "
            "starting force; sent while running it applies from the next chunk the model starts "
            "and reaches the screen 3 chunks later. Held until changed. Answers with "
            "`state_update`. Fails with `unsupported_value` for another mode or a value out of range."
        ),
    )
    def set_force(
        self,
        mode: str = InputField(description="`point` or `wind`."),
        strength: float = InputField(description="0..1 of the strongest force in training; 0 is no push."),
        angle: float = InputField(
            description="Direction in degrees, counter-clockwise from +x: 0 right, 90 up, 180 left, 270 down."
        ),
        x: float = InputField(default=0.5, description="Point only: horizontal position, 0..1 over the output frame from the left."),
        y: float = InputField(default=0.5, description="Point only: vertical position, 0..1 over the output frame from the top."),
    ) -> StateUpdate:
        if mode not in ("point", "wind"):
            raise CommandError("unsupported_value", f"mode must be point or wind, not {mode!r}.")
        if not (math.isfinite(strength) and 0.0 <= strength <= 1.0):
            raise CommandError("unsupported_value", "strength must be in 0..1.")
        if not math.isfinite(angle):
            raise CommandError("unsupported_value", "angle must be a number of degrees.")
        if mode == "point":
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise CommandError("unsupported_value", "x and y must be in 0..1.")
            self.state._force = Force("point", float(strength), float(angle), float(x), float(y))
        else:
            self.state._force = Force("wind", float(strength), float(angle))
        return self._snapshot()

    @event(
        name="set_seed",
        description=(
            "Session setup: the seed of the run's randomness; -1 draws a random one at `start` "
            "(reported in `generation_started`). Valid while idle. Answers with `state_update`. "
            "Fails with `locked_while_running` during a run and `unsupported_value` out of range."
        ),
    )
    def set_seed(self, seed: int = InputField(description="Integer 0..2^32-1, or -1 for random.")) -> StateUpdate:
        self._locked("seed")
        if seed < -1 or seed >= 2**32:
            raise CommandError("unsupported_value", "seed must be -1 or an integer in 0..2^32-1.")
        self.state._seed = int(seed)
        return self._snapshot()

    # -- hooks ------------------------------------------------------------------

    @session_started
    def on_session_started(self) -> None:
        self.engine.reset()

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        await client.send(self._snapshot())

    @session_ended
    def on_session_ended(self) -> None:
        self._clear()
        self.output.flush()

    # -- helpers ----------------------------------------------------------------

    def _locked(self, what: str) -> None:
        if self.state._phase != "idle":
            raise CommandError("locked_while_running", f"The {what} is session setup; reset to change it.")

    def _clear(self) -> None:
        s = self.state
        self.engine.reset()
        s._phase = "idle"
        s._first_frame = None
        s._prompt = None
        s._force = None
        s._seed = -1
        s._applied_run_id = None
        s._applied_seed = s._applied_prompt = s._applied_force = None
        s._prompt_from_chunk = s._force_from_chunk = None
        s._last_chunk = -1
        s._frames_in_run = 0

    def _snapshot(self) -> StateUpdate:
        s = self.state
        return StateUpdate(
            state=s._phase,
            settings={
                "first_frame": {"sent": s._first_frame is not None, "applied": s._applied_run_id is not None},
                "prompt": {"sent": s._prompt, "applied": s._applied_prompt, "from_chunk": s._prompt_from_chunk},
                "force": {
                    "sent": force_dict(s._force),
                    "applied": force_dict(s._applied_force),
                    "from_chunk": s._force_from_chunk,
                },
                "seed": {"sent": s._seed, "applied": s._applied_seed},
            },
            capabilities=CAPABILITIES,
            last_chunk=s._last_chunk,
            frames_per_session=self.frames_per_session,
        )


def _decode(data: bytes) -> np.ndarray:
    """Decode an upload to an RGB uint8 array, honouring EXIF orientation."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if image.width * image.height > MAX_UPLOAD_PIXELS:
        raise ValueError("image too large")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
