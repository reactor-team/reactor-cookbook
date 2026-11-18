"""MovieRoll application policy; GPU state lives in the isolated model half."""

import asyncio
from pathlib import Path
from typing import Literal

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from movieroll_assets import prepare
from movieroll_images import ImageError, decode_image
from movieroll_model import (
    MAX_PROMPTS,
    Anchor,
    HistoryFullError,
    MovieRollInput,
    MovieRollModel,
    MovieRollResult,
)
from movieroll_types import MovieRollOutput, MovieRollState, StateUpdate


class MovieRoll(ReactorApp):
    """Stream MovieRoll-Distilled from text or an image, with live event prompts."""

    state: MovieRollState
    buffer_size = 16

    def load(self, config_path: Path | None) -> None:
        settings = prepare(
            config_path or Path(__file__).with_name("movieroll.yaml"),
            get_weights_path(),
        )
        self.engine = DistributedRunner(
            MovieRollModel,
            world_size=settings.gpu_count,
            init_process_group=settings.gpu_count > 1,
            load_kwargs={"settings": settings},
            call_timeout=120,
            start_timeout=300,
        )
        self.engine.start()

    async def process_input(self) -> MovieRollInput:
        s = self.state
        if s._anchor is None or s.paused:
            raise ApplicationError("Press Start or resume generation")
        if not s.prompt.strip():
            raise ApplicationError("Enter a scene description")
        return MovieRollInput(
            world_id=s._world_id,
            prompt=s.prompt,
            anchor=s._anchor if s._applied_world_id != s._world_id else None,
        )

    def generate(self, input: MovieRollInput) -> MovieRollResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> MovieRollOutput | None:
        if isinstance(outcome.error, HistoryFullError):
            if outcome.error.world_id is not None and (
                outcome.error.world_id != self.state._world_id
                or self.state._anchor is None
            ):
                return None
            self.engine.reset()
            self.state._anchor = None
            self.state._applied_world_id = None
            self.state.paused = False
            self.state._end_reason = "history_full"
            self.output.flush()
            await self.send(self._status())
            return None
        if outcome.error is not None:
            # A partial native forward can mutate KV, RNG, and decoder state.
            # Unknown failures end the service; a reset is not proven recovery.
            raise outcome.error
        result: MovieRollResult = outcome.result
        s = self.state
        if result.world_id != s._world_id or s._anchor is None:
            return None
        s._applied_world_id = result.world_id
        s._completed = result.chunk_index
        s._frames += len(result.frames)
        s._last_seconds = outcome.elapsed
        await self.send(self._status())
        return MovieRollOutput(main_video=result.frames)

    def _status(self) -> StateUpdate:
        s = self.state
        return StateUpdate(
            prompt=s.prompt,
            active=s._anchor is not None,
            paused=s.paused,
            has_image=s._image is not None,
            mode=s._anchor.mode if s._anchor else None,
            seed=s._anchor.seed if s._anchor else None,
            world_id=s._world_id,
            completed_chunks=s._completed,
            output_frames=s._frames,
            last_chunk_seconds=s._last_seconds,
            prompt_updates_remaining=MAX_PROMPTS - s._prompts_used
            if s._anchor
            else MAX_PROMPTS - 1,
            end_reason=s._end_reason,
        )

    async def _broadcast(self) -> StateUpdate:
        status = self._status()
        await self.send(status)
        return status

    def _fresh(self, seed: int, mode: Literal["t2v", "i2v"]) -> None:
        s = self.state
        s._world_id += 1
        s._applied_world_id = None
        s._anchor = Anchor(
            seed=seed, mode=mode, image=s._image if mode == "i2v" else None
        )
        s._completed = s._frames = 0
        s._last_seconds = None
        s._prompts_used = 1
        s._end_reason = None
        s.paused = False
        self.output.flush()

    @event(
        name="set_image",
        description="Stage an initial PNG, JPEG, or WebP image for a new I2V video. Does not start or replace the current video. Fails with `unsupported_image`, `image_size`, or `invalid_image` for an unusable upload.",
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description="Initial image, at most 20 MiB and 32 million pixels; orientation-corrected, resized and center-cropped to 864 by 480.",
        ),
    ) -> StateUpdate:
        try:
            decoded = await asyncio.to_thread(decode_image, image.data, image.mime_type)
        except ImageError as exc:
            raise CommandError(exc.code, str(exc)) from exc
        self.state._image = decoded
        return await self._broadcast()

    @event(
        name="clear_image",
        description="Remove the staged image. A running video keeps its original anchor.",
    )
    async def clear_image(self) -> StateUpdate:
        self.state._image = None
        return await self._broadcast()

    @event(
        name="start",
        description="Start a fresh continuous video using the released 5-step T2V or I2V checkpoint. I2V uses the staged image. Fails with `invalid_prompt` for empty text or `image_required` when I2V has no image.",
    )
    async def start(
        self,
        prompt: str = InputField(
            default="",
            moderate=True,
            max_length=8192,
            description="Full scene caption; empty uses the staged caption.",
        ),
        seed: int = InputField(
            default=0, ge=0, le=4294967295, description="New video's random seed."
        ),
        mode: str = InputField(
            default="t2v",
            choices=["t2v", "i2v"],
            description="Use the text or image distilled checkpoint.",
        ),
    ) -> StateUpdate:
        prompt = prompt.strip() or self.state.prompt.strip()
        if not prompt:
            raise CommandError("invalid_prompt", "Enter a scene description")
        if mode == "i2v" and self.state._image is None:
            raise CommandError("image_required", "Upload an initial image first")
        self.state.prompt = prompt
        self._fresh(seed, mode)
        return await self._broadcast()

    @event(
        name="set_prompt",
        description="Replace the full scene caption while the same video continues. Frames already generated keep their previous caption; changes blend through several chunks. Keeps pause state. Fails with `invalid_prompt` for empty text or `prompt_limit` after 20 accepted captions in one video.",
    )
    async def set_prompt(
        self, prompt: str = MovieRollState._public_fields["prompt"]
    ) -> StateUpdate:
        prompt = prompt.strip()
        if not prompt:
            raise CommandError("invalid_prompt", "Enter a scene description")
        s = self.state
        if s._anchor is not None and prompt != s.prompt:
            if s._prompts_used >= MAX_PROMPTS:
                raise CommandError(
                    "prompt_limit",
                    "Start a new video to use more than 20 event captions",
                )
            s._prompts_used += 1
        s.prompt = prompt
        return await self._broadcast()

    @event(
        name="set_paused",
        description="Hold or resume generation between chunks. Buffered frames may still play. Fails with `no_video` before Start or after Stop.",
    )
    async def set_paused(
        self, paused: bool = MovieRollState._public_fields["paused"]
    ) -> StateUpdate:
        if self.state._anchor is None:
            raise CommandError("no_video", "Start a video first")
        self.state.paused = paused
        return await self._broadcast()

    @event(
        name="reset",
        description="Restart from the current video's original image and accepted caption. Omitted seed retains the current seed. Fails with `no_video` before Start or after Stop.",
    )
    async def reset(
        self,
        seed: int | None = InputField(
            default=None, ge=0, le=4294967295, description="Optional replacement seed."
        ),
    ) -> StateUpdate:
        anchor = self.state._anchor
        if anchor is None:
            raise CommandError("no_video", "Start a video first")
        # A newly staged upload belongs to the next Start; Reset keeps this run's image.
        self._fresh(anchor.seed if seed is None else seed, anchor.mode)
        self.state._anchor = Anchor(
            seed=self.state._anchor.seed, mode=anchor.mode, image=anchor.image
        )
        return await self._broadcast()

    @event(
        name="stop",
        description="End the video and clear playback and temporary history. Keep the caption and staged image for Start.",
    )
    async def stop_video(self) -> StateUpdate:
        s = self.state
        s._anchor = None
        s._applied_world_id = None
        s.paused = False
        s._completed = s._frames = 0
        s._last_seconds = None
        s._prompts_used = 0
        s._end_reason = "stopped"
        self.engine.reset()
        self.output.flush()
        return await self._broadcast()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @event(
        name="get_state",
        description="Read the current video's state without changing it or broadcasting to other viewers.",
    )
    def get_state(self) -> StateUpdate:
        # SDK 3.4.0 can miss the connected push when joining a paused session.
        # The browser requests this once after ready; it does not poll.
        return self._status()

    @session_ended
    def on_session_ended(self) -> None:
        if self.engine.healthy:
            self.engine.reset()
        self.output.flush()
