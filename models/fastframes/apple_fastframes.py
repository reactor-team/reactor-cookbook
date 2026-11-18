"""Stream FastFrames clips with editable text conditioning."""

from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from apple_fastframes_assets import prepare
from apple_fastframes_model import (
    FOLLOWUP_FRAMES,
    FastFramesInput,
    FastFramesModel,
    FastFramesResult,
    Variant,
)
from apple_fastframes_types import (
    ClipCommandAccepted,
    FastFramesOutput,
    FastFramesState,
    StateUpdate,
)


class FastFramesApp(ReactorApp):
    """Generate silent video from text with prompt updates during a clip."""

    state: FastFramesState
    fps = 16
    buffer_size = FOLLOWUP_FRAMES
    _variant: Variant = "ff-s"

    def load(self, config_path: Path | None) -> None:
        settings = prepare(
            config_path or Path(__file__).with_suffix(".yaml"), get_weights_path()
        )
        self._variant = settings.variant
        self.engine = DistributedRunner(
            FastFramesModel,
            world_size=1,
            load_kwargs={"settings": settings},
        )
        self.engine.start()

    async def process_input(self) -> FastFramesInput:
        state = self.state
        if state._phase != "generating":
            raise ApplicationError("FastFrames is waiting for Start or Resume")
        return FastFramesInput(
            world_id=state._world_id,
            prompt=state._prompt,
            prompt_revision=state._prompt_revision,
            seed=state._seed if state._applied_id != state._world_id else None,
        )

    def generate(self, input: FastFramesInput) -> FastFramesResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> FastFramesOutput | None:
        if outcome.error is not None:
            # A failed GPU call can leave partially written cache entries.
            raise outcome.error
        result: FastFramesResult = outcome.result
        state = self.state
        if result.world_id != state._world_id or state._phase not in (
            "generating",
            "paused",
        ):
            return None
        state._applied_id = result.world_id
        state._applied_prompt_revision = result.prompt_revision
        state._chunks = result.chunk
        state._frames += len(result.frames)
        state._last_chunk_seconds = outcome.elapsed
        if result.complete:
            state._phase = "complete"
        await self.send(self._status())
        return FastFramesOutput(main_video=result.frames)

    def _status(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            variant=self._variant,
            world_id=state._world_id,
            phase=state._phase,
            prompt=state._prompt,
            prompt_pending=(
                state._phase in ("generating", "paused")
                and state._prompt_revision != state._applied_prompt_revision
            ),
            completed_chunks=state._chunks,
            output_frames=state._frames,
            last_chunk_seconds=state._last_chunk_seconds,
        )

    async def _broadcast(self) -> ClipCommandAccepted:
        status = self._status()
        await self.send(status)
        return ClipCommandAccepted(world_id=status.world_id)

    @event(
        name="start",
        description="Start a fresh clip. Fails with `invalid_prompt` for blank text.",
    )
    async def start_video(
        self,
        prompt: str = InputField(
            moderate=True, max_length=4096, description="Scene to generate."
        ),
        seed: int = InputField(
            default=0, ge=0, le=4294967295, description="Clip seed."
        ),
    ) -> ClipCommandAccepted:
        if not prompt.strip():
            raise CommandError("invalid_prompt", "Enter a scene description")
        state = self.state
        state._world_id += 1
        state._applied_id = None
        state._phase = "generating"
        state._seed = seed
        state._prompt = prompt
        state._prompt_revision = 0
        state._applied_prompt_revision = None
        state._chunks = 0
        state._frames = 0
        state._last_chunk_seconds = None
        self.output.flush()
        return await self._broadcast()

    @event(
        name="update_prompt",
        description=(
            "Replace the caption used by upcoming chunks while keeping the current clip. "
            "Fails with `invalid_prompt` for blank text or `no_active_clip` if the clip ended."
        ),
    )
    async def update_prompt(
        self,
        prompt: str = InputField(
            moderate=True, max_length=4096, description="Full new scene caption."
        ),
    ) -> ClipCommandAccepted:
        if not prompt.strip():
            raise CommandError("invalid_prompt", "Enter a scene description")
        state = self.state
        if state._phase not in ("generating", "paused"):
            raise CommandError(
                "no_active_clip", "Start a clip before editing its prompt"
            )
        if prompt != state._prompt:
            state._prompt = prompt
            state._prompt_revision += 1
        return await self._broadcast()

    @event(
        name="set_paused",
        description=(
            "Pause or resume generation. One in-flight chunk can finish. "
            "Fails with `no_active_clip` if the clip ended."
        ),
    )
    async def set_paused(self, paused: bool) -> ClipCommandAccepted:
        if self.state._phase not in ("generating", "paused"):
            raise CommandError(
                "no_active_clip", "There is no active clip to pause or resume"
            )
        self.state._phase = "paused" if paused else "generating"
        return await self._broadcast()

    @event(
        name="stop",
        description="Stop the clip, clear its progress and discard queued video.",
    )
    async def stop_video(self) -> ClipCommandAccepted:
        state = self.state
        state._world_id += 1
        state._applied_id = None
        state._phase = "stopped"
        state._applied_prompt_revision = None
        state._chunks = 0
        state._frames = 0
        state._last_chunk_seconds = None
        self.output.flush()
        self.engine.reset()
        return await self._broadcast()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        if self.engine.healthy:
            self.engine.reset()
        self.output.flush()
