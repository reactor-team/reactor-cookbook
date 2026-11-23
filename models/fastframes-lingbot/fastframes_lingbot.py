"""Stream an interactive FastFrames-LingBot world on Reactor."""

from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    disconnected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from fastframes_lingbot_assets import prepare
from fastframes_lingbot_images import prepare_image
from fastframes_lingbot_model import (
    CHUNK_FRAMES,
    CHUNKS,
    Anchor,
    CameraAction,
    FastFramesLingBotModel,
    LingBotInput,
    LingBotResult,
)
from fastframes_lingbot_types import (
    CameraAccepted,
    PromptAccepted,
    StateUpdate,
    WorldOutput,
    WorldStarted,
    WorldState,
    WorldStopped,
)


class FastFramesLingBotApp(ReactorApp):
    """Explore an uploaded scene using the native camera actions."""

    state: WorldState
    fps = 16
    buffer_size = CHUNK_FRAMES

    def load(self, config_path: Path | None) -> None:
        settings = prepare(config_path or Path(__file__).with_suffix(".yaml"), get_weights_path())
        self.engine = DistributedRunner(
            FastFramesLingBotModel, world_size=1, load_kwargs={"settings": settings}, call_timeout=120.0
        )
        self.engine.start()

    async def process_input(self) -> LingBotInput:
        state = self.state
        if state._phase != "active":
            raise ApplicationError("Start a world before exploring")
        anchor = state._anchor
        action = "idle" if anchor is not None else state._pending or state._held or "idle"
        if anchor is None:
            state._pending = None
        return LingBotInput(state._world_id, anchor, state._prompt, state._prompt_revision, action)

    def generate(self, input: LingBotInput) -> LingBotResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> WorldOutput | None:
        if outcome.error is not None:
            raise outcome.error
        result: LingBotResult = outcome.result
        state = self.state
        if result.world_id != state._world_id or state._phase != "active":
            return None
        state._anchor = None
        state._applied_revision = result.prompt_revision
        state._last_action = result.action
        state._chunks = result.chunk
        state._frames += len(result.frames)
        state._last_chunk_seconds = outcome.elapsed
        if result.chunk == CHUNKS:
            state._phase = "complete"
            state._owner = None
            state._pending = state._held = None
        await self.send(self._status())
        return WorldOutput(main_video=result.frames)

    def _status(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            world_id=state._world_id,
            phase=state._phase,
            queued_action=state._pending,
            held_action=state._held,
            last_action=state._last_action,
            prompt_pending=state._phase == "active" and state._prompt_revision != state._applied_revision,
            completed_chunks=state._chunks,
            max_chunks=CHUNKS,
            playback_fps=self.fps,
            output_frames=state._frames,
            last_chunk_seconds=state._last_chunk_seconds,
        )

    def _require_controller(self, client: ClientInfo, world_id: int) -> None:
        state = self.state
        if state._world_id != world_id:
            raise CommandError("stale_world", "This control belongs to a different world")
        if state._phase != "active":
            raise CommandError("no_active_world", "Start a world before exploring")
        if state._owner != client.id:
            raise CommandError("controller_required", "Only the viewer who started this world can control it")

    @event(
        name="start_world",
        description=(
            "Start with an uploaded image. Fails with `world_in_use` if another viewer is exploring, "
            "`invalid_image` if the image is unusable or `invalid_prompt` for blank text."
        ),
    )
    async def start_world(
        self,
        client: ClientInfo,
        image: UploadedFile = InputField(moderate=True, description="JPEG, PNG or WebP up to 8 MiB"),
        prompt: str = InputField(moderate=True, max_length=1600, description="Scene description"),
        seed: int = InputField(default=0, ge=0, le=4294967295, description="World seed"),
    ) -> WorldStarted:
        state = self.state
        if state._owner is not None and state._owner != client.id:
            raise CommandError("world_in_use", "Another viewer is exploring this world")
        if not prompt.strip():
            raise CommandError("invalid_prompt", "Enter a scene description")
        try:
            anchor = Anchor(prepare_image(image.data), seed)
        except ValueError as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        state._world_id += 1
        state._owner = client.id
        state._phase = "active"
        state._anchor = anchor
        state._prompt = prompt.strip()
        state._prompt_revision = 0
        state._applied_revision = None
        state._pending = state._held = state._last_action = None
        state._chunks = state._frames = 0
        state._last_chunk_seconds = None
        self.output.flush()
        await self.send(self._status())
        return WorldStarted(world_id=state._world_id)

    @event(
        name="move_camera",
        description=(
            "Queue one native camera action or repeat it until released. The latest action replaces "
            "queued input. Fails with `stale_world`, `no_active_world` or `controller_required`."
        ),
    )
    async def move_camera(
        self,
        client: ClientInfo,
        world_id: int = InputField(ge=1, description="ID of the world to move in"),
        action: CameraAction = InputField(description="Action to apply for one block"),
        hold: bool = InputField(default=False, description="Repeat the action until release"),
    ) -> CameraAccepted:
        self._require_controller(client, world_id)
        self.state._held = action if hold else None
        self.state._pending = action
        await self.send(self._status())
        return CameraAccepted(world_id=world_id, action=action)

    @event(
        name="release_camera",
        description=(
            "End camera repetition and continue at the current pose. "
            "Fails with `stale_world`, `no_active_world` or `controller_required`."
        ),
    )
    async def release_camera(
        self,
        client: ClientInfo,
        world_id: int = InputField(ge=1, description="ID of the world to release"),
        discard_pending: bool = InputField(default=False, description="Also discard a queued tap"),
    ) -> CameraAccepted:
        self._require_controller(client, world_id)
        self.state._held = None
        if discard_pending:
            self.state._pending = None
        await self.send(self._status())
        return CameraAccepted(world_id=world_id, action=None)

    @event(
        name="update_prompt",
        description=(
            "Use a new description on the next generated block in the current world. Fails with "
            "`stale_world`, `no_active_world`, `controller_required` or `invalid_prompt` for blank text."
        ),
    )
    async def update_prompt(
        self,
        client: ClientInfo,
        world_id: int = InputField(ge=1, description="ID of the world to update"),
        prompt: str = InputField(moderate=True, max_length=1600, description="New scene description"),
    ) -> PromptAccepted:
        self._require_controller(client, world_id)
        prompt = prompt.strip()
        if not prompt:
            raise CommandError("invalid_prompt", "Enter a scene description")
        if prompt != self.state._prompt:
            self.state._prompt = prompt
            self.state._prompt_revision += 1
        await self.send(self._status())
        return PromptAccepted(world_id=world_id, revision=self.state._prompt_revision)

    @event(
        name="stop_world",
        description="Stop the active world. Fails with `stale_world`, `no_active_world` or `controller_required`.",
    )
    async def stop_world(
        self, client: ClientInfo, world_id: int = InputField(ge=1, description="ID of the world to stop")
    ) -> WorldStopped:
        self._require_controller(client, world_id)
        await self._stop()
        return WorldStopped(world_id=self.state._world_id)

    async def _stop(self) -> None:
        state = self.state
        state._world_id += 1
        state._phase = "stopped"
        state._owner = state._anchor = None
        state._prompt = ""
        state._pending = state._held = state._last_action = None
        state._chunks = state._frames = 0
        state._last_chunk_seconds = None
        self.output.flush()
        self.engine.reset()
        await self.send(self._status())

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @disconnected
    async def on_disconnected(self, client: ClientInfo) -> None:
        if client.id == self.state._owner:
            await self._stop()

    @session_ended
    def on_session_ended(self) -> None:
        self.output.flush()
        if self.engine.healthy:
            self.engine.reset()
