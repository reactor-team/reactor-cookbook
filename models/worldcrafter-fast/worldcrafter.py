"""Reactor application for image-conditioned, camera-controlled WorldCrafter."""
import asyncio
from pathlib import Path

from reactor_runtime import (
    ApplicationError, ClientInfo, CommandError, InputField, ReactorApp,
    StepOutcome, TrackPayload, UploadedFile, connected, disconnected, event,
    session_ended, get_weights_path,
)
from reactor_runtime.log import get_logger
from upstream_backend import InferenceWorkerError
from worldcrafter_assets import prepare_runtime, read_settings
from worldcrafter_images import validate_image
from worldcrafter_camera import ACTIONS, FRAMES, CameraPlanner, load_camera_path, validate_motion
from worldcrafter_model import WorldCrafterInput, WorldCrafterModel, WorldCrafterResult
from worldcrafter_types import (
    CameraPathSelected, ChunkGenerated, MotionQueued, PlaybackChanged, RolloutLimitReached,
    StateUpdate, WorldCrafterOutput, WorldCrafterState, WorldStarted,
)

logger = get_logger(__name__)


class WorldCrafter(ReactorApp):
    state: WorldCrafterState
    buffer_size = 2 * FRAMES

    def __init__(self):
        super().__init__()
        self._engine = None
        self._settings = None
        self._max_chunks = 50

    def load(self, config_path: Path | None):
        settings = read_settings(config_path, weights_path=get_weights_path())
        prepare_runtime(settings)
        self._settings = settings
        self._max_chunks = settings.max_chunks
        self._start_engine()

    def _start_engine(self):
        """Start the inference worker; it reports ready once loaded and warmed up."""
        self._engine = WorldCrafterModel()
        self._engine.load(self._settings)

    @session_ended
    def on_session_ended(self):
        """Release the live world and keep the weights; replace a worker that failed."""
        if self._engine.healthy:
            try:
                self._engine.reset()
            except InferenceWorkerError:
                logger.exception("WorldCrafter inference worker failed to reset")
        if not self._engine.healthy:
            logger.error("WorldCrafter inference worker is unhealthy; restarting it")
            self._engine.close()
            self._start_engine()
        self.output.flush()

    @connected
    async def on_connected(self, client: ClientInfo):
        await client.send(self._snapshot())

    @disconnected
    async def on_disconnected(self):
        await self.send(self._snapshot())

    @event(name="start_world", description="Start a fresh world from an image and scene prompt; generates one stationary chunk. Returns `world_started` and broadcasts `state_update`. Fails with `invalid_image` or `prompt_required`.")
    async def start_world(
        self,
        image: UploadedFile = InputField(moderate=True, description="PNG, JPEG or WebP anchor image, at most 20 MiB."),
        prompt: str = InputField(default="", max_length=4096, moderate=True, description="Scene description, including first- or third-person perspective."),
        seed: int = InputField(default=0, ge=0, le=2147483647, description="Reproducible generation seed."),
        gravity_align: bool = InputField(default=True, description="Align yaw and horizontal orbit to image gravity when calibration succeeds."),
    ) -> WorldStarted:
        if not prompt.strip():
            raise CommandError("prompt_required", "Provide a scene description.")
        try:
            validated = await asyncio.to_thread(validate_image, image.data)
        except ValueError as exc:
            raise CommandError("invalid_image", "Use a valid PNG, JPEG or WebP, at least 64 pixels per side and at most 20 megapixels.") from exc
        self.state._image, self.state._prompt = validated, prompt.strip()
        self.state._seed, self.state._gravity = seed, gravity_align
        self.state._camera_path = None
        self._fresh_world()
        await self.send(self._snapshot())
        return WorldStarted(world_id=self.state._world_id, seed=self.state._seed)

    @event(name="move", description="Queue camera motion for 1–8 chunks. Translation amount is 0.1–5 units; rotation is 0.1–45 degrees per chunk. Returns `motion_queued` and broadcasts `state_update`. Fails with `image_required`, `invalid_motion`, `queue_full`, or `rollout_limit`.")
    async def move(
        self,
        action: str = InputField(choices=list(ACTIONS), description="Camera motion direction."),
        amount: float = InputField(default=1.0, ge=0.1, le=45.0, description="Distance or rotation degrees per chunk."),
        chunks: int = InputField(default=1, ge=1, le=8, description="Number of consecutive chunks."),
    ) -> MotionQueued:
        self._require_image()
        if self.state._camera_path is not None:
            raise CommandError("camera_path_active", "Start a world to return to camera motion commands.")
        try:
            validate_motion(action, amount, chunks)
        except ValueError as exc:
            raise CommandError("invalid_motion", str(exc)) from exc
        if len(self.state._queue) + chunks > 16:
            raise CommandError("queue_full", "At most 16 camera chunks may be queued.")
        in_progress = int(self.state._inflight == self.state._world_id)
        if self.state._completed + in_progress + len(self.state._queue) + chunks > self._max_chunks:
            raise CommandError("rollout_limit", "Reset or start a new world before adding more chunks.")
        self.state._queue.extend([(action, amount)] * chunks)
        await self.send(self._snapshot())
        return MotionQueued(action=action, chunks=chunks)

    @event(name="set_camera_path", description="Upload a finite camera-to-world path and restart the selected image, prompt and seed along it. Returns `camera_path_selected` and broadcasts `state_update`. Generation stops at the path end; `reset` replays it and `start_world` restores `move`. Fails with `image_required` or `invalid_camera_path`.")
    async def set_camera_path(
        self,
        trajectory: UploadedFile = InputField(moderate=True, description="JSON {\"poses\": [33 * chunks, 4, 4]}, at most 16 MiB: global right/down/forward c2w, one pose per output frame, pose zero at the input image. Gravity alignment follows start_world."),
    ) -> CameraPathSelected:
        self._require_image()
        try:
            path = await asyncio.to_thread(load_camera_path, trajectory.data, self._max_chunks)
        except ValueError as exc:
            raise CommandError("invalid_camera_path", str(exc)) from exc
        self.state._camera_path = path
        self._fresh_world()
        await self.send(self._snapshot())
        return CameraPathSelected(world_id=self.state._world_id, chunks=path.chunks,
                                  frames=path.chunks * FRAMES)

    @event(name="pause", description="Pause before the next chunk; an in-flight chunk may finish. Returns `playback_changed` and broadcasts `state_update`.")
    async def pause(self) -> PlaybackChanged:
        self.state._paused = True
        await self.send(self._snapshot())
        return PlaybackChanged(paused=True)

    @event(name="resume", description="Resume queued camera motion. Returns `playback_changed` and broadcasts `state_update`. Fails with `image_required`.")
    async def resume(self) -> PlaybackChanged:
        self._require_image()
        self.state._paused = False
        await self.send(self._snapshot())
        return PlaybackChanged(paused=False)

    @event(name="reset", description="Restart from the selected image, prompt and seed, discarding exploration history. Returns `world_started` and broadcasts `state_update`. Fails with `image_required`.")
    async def reset(self) -> WorldStarted:
        self._require_image()
        self._fresh_world()
        await self.send(self._snapshot())
        return WorldStarted(world_id=self.state._world_id, seed=self.state._seed)

    async def process_input(self) -> WorldCrafterInput:
        if self.state._image is None or self.state._paused or not self.state._queue:
            if self.state._camera_path is not None and not self.state._queue:
                raise ApplicationError("Camera path is complete. Reset or upload another path.")
            raise ApplicationError("Waiting for an image or camera motion.")
        action, amount = self.state._queue.popleft()
        world_id = self.state._world_id
        self.state._inflight = world_id
        local = world = path = None
        if action == "path":
            path = self.state._camera_path.poses[:FRAMES * (amount + 1)]
        else:
            local, world = self.state._planner.plan(action, amount)
        return WorldCrafterInput(
            world_id=world_id,
            image=self.state._image if world_id != self.state._applied_world_id else None,
            prompt=self.state._prompt, seed=self.state._seed,
            gravity_align=self.state._gravity, local_poses=local, world_poses=world,
            path_poses=path,
        )

    def generate(self, input: WorldCrafterInput) -> WorldCrafterResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> WorldCrafterOutput | None:
        self.state._inflight = None
        if outcome.error is not None:
            if not isinstance(outcome.error, InferenceWorkerError):
                raise outcome.error
            # Escaping process_output ends the Runtime's model loop. Recover
            # an expected worker failure here, under the Runtime's step lock.
            # A replacement cannot continue the lost world's scene memory.
            logger.error("WorldCrafter inference worker failed; rebuilding once",
                         error=str(outcome.error))
            self._engine.close()
            self._fresh_world()
            self.state._paused = True
            await self.send(self._snapshot())
            await asyncio.to_thread(self._start_engine)
            return None
        result: WorldCrafterResult = outcome.result
        if result.world_id != self.state._world_id:
            return None
        if self.state._applied_world_id != result.world_id and self.state._camera_path is None:
            self.state._planner.align_after_anchor(result.world_up)
        self.state._applied_world_id = result.world_id
        self.state._completed = result.chunk_index + 1
        await self.send(ChunkGenerated(world_id=result.world_id, chunk_index=result.chunk_index,
                                      frames=len(result.frames), generation_seconds=outcome.elapsed,
                                      camera_end=result.camera_end, gravity_aligned=result.gravity_aligned,
                                      gpu_count=result.gpu_count))
        if self.state._completed >= self._max_chunks:
            self.state._queue.clear()
            await self.send(RolloutLimitReached(completed_chunks=self.state._completed))
        await self.send(self._snapshot())
        metadata = [dict(world_id=result.world_id, chunk_index=result.chunk_index, frame_index=i)
                    for i in range(len(result.frames))]
        return WorldCrafterOutput(main_video=TrackPayload(result.frames, metadata=metadata))

    def _fresh_world(self):
        self.output.flush()
        self.state._world_id += 1
        self.state._planner = CameraPlanner()
        self.state._completed = 0
        self.state._paused = False
        self.state._queue.clear()
        if self.state._camera_path is None:
            self.state._queue.append(("static", 0.0))
        else:
            self.state._queue.extend(("path", index) for index in range(self.state._camera_path.chunks))

    def _require_image(self):
        if self.state._image is None:
            raise CommandError("image_required", "Start a world with an image first.")

    def _snapshot(self):
        return StateUpdate(world_id=self.state._world_id, image_selected=self.state._image is not None,
                           paused=self.state._paused, completed_chunks=self.state._completed,
                           queued_chunks=len(self.state._queue), max_chunks=self._max_chunks,
                           camera_path_chunks=None if self.state._camera_path is None
                           else self.state._camera_path.chunks)
