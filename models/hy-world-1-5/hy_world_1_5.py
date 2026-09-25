"""Serve HY-World 1.5 distilled autoregressive inference through Reactor."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from hy_world_1_5_assets import (
    ExampleImage,
    HYWorld15Config,
    load_examples,
    prepare_runtime_assets,
    read_config,
)
from hy_world_1_5_camera import CameraControl, NativeCameraPlanner
from hy_world_1_5_images import (
    FRAMES_PER_CHUNK,
    validate_uploaded_image,
)
from hy_world_1_5_model import (
    HYWorld15Anchor,
    HYWorld15Input,
    HYWorld15Model,
    HYWorld15Result,
)
from hy_world_1_5_types import (
    CameraMotionChanged,
    ChunkCompleted,
    HYWorld15Output,
    HYWorld15State,
    ImageSelected,
    PromptQueued,
    RolloutLimitReached,
    RolloutResetQueued,
    StateUpdate,
)
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
    session_started,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_DEFAULT_UPLOAD_PROMPT = "Continue the world shown in the reference image."


class HYWorld15(ReactorApp):
    """Generate an image-, prompt-, and camera-controllable HY-World 1.5 world."""

    state: HYWorld15State
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: HYWorld15Config | None = None
        self._engine = HYWorld15Model()
        self._planner: NativeCameraPlanner | None = None
        self._examples: tuple[ExampleImage, ...] = ()
        self._selected_input: Path | UploadedFile | None = None
        self._image_source: Literal["uploaded", "built_in"] | None = None
        self._image_name: str | None = None
        self._seed = 1
        self._chunk_index = 0
        self._active_prompt: str | None = None
        self._generating = False

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load the distilled model once.

        Args:
            config_path: Path to ``hy_world_1_5.yaml`` from the runtime spec.
        """
        config = read_config(config_path, get_weights_path())
        prepare_runtime_assets(config)
        self._engine.load(config)
        self._config = config
        self._planner = NativeCameraPlanner()
        self._examples = load_examples(config)
        self._seed = config.seed
        logger.info(
            "HY-World 1.5 model ready",
            examples=len(self._examples),
            max_chunks=config.max_chunks,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty shared world for continuous generation."""
        config = self._require_config()
        self._selected_input = None
        self._image_source = None
        self._image_name = None
        self._seed = config.seed
        self._chunk_index = 0
        self._active_prompt = None
        self._generating = False
        self.state.prompt = ""
        self.state._world_id = 0
        self.state._applied_world_id = None
        self.state._limit_reached = False
        self._release_camera()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera motion when its viewer leaves."""
        self._release_camera()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        """Release causal session state while retaining loaded model weights."""
        self._engine.reset()
        self._selected_input = None
        self._image_source = None
        self._image_name = None
        self._chunk_index = 0
        self._active_prompt = None
        self._generating = False
        self.state._applied_world_id = None
        self.state._limit_reached = False
        self._release_camera()

    @event(
        name="set_prompt",
        description=(
            "Queue a non-empty scene prompt without restarting the current world. Requires a "
            "selected image. The model rebuilds its text condition at the next causal chunk "
            "boundary while preserving latent history and geometric memory. Emits "
            "`prompt_queued` and broadcasts `state_update` on success, or `command_error` when "
            "the prompt is empty or no image is selected."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Scene description, up to 4096 characters and non-empty after trimming. The "
                "next generated chunk samples it; earlier latent memory remains intact."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and report the first affected chunk."""
        self._require_image()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "HY-World 1.5 requires a prompt.")
        self.state.prompt = normalized
        message = PromptQueued(
            prompt=normalized,
            applies_to_chunk=self._next_control_chunk(),
        )
        await self._send_state_update()
        return message

    @event(
        name="set_camera",
        description=(
            "Atomically set all four camera controls trained by HY-World 1.5. Requires an active "
            "world before its rollout limit. Values are sampled together at the next chunk and "
            "held for later chunks. Emits `camera_motion_changed` and broadcasts `state_update` "
            "on success, or `command_error` when no image is selected or the limit is reached."
        ),
    )
    async def set_camera(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Backward (-1) to forward (1) movement; zero is neutral.",
        ),
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Left (-1) to right (1) movement; zero is neutral.",
        ),
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Look down (-1) to up (1); zero is neutral.",
        ),
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Turn left (-1) to right (1); zero is neutral.",
        ),
    ) -> CameraMotionChanged:
        """Queue one complete native camera state and report its boundary."""
        self._require_image()
        self._require_available_rollout()
        self.state._forward = forward
        self.state._strafe = strafe
        self.state._pitch = pitch
        self.state._yaw = yaw
        message = self._camera_message()
        await self._send_state_update()
        return message

    @event(
        name="release_camera",
        description=(
            "Return every held camera control to neutral. Requires a selected image and applies "
            "at the next chunk boundary. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def release_camera(self) -> CameraMotionChanged:
        """Release all native camera axes and report their boundary."""
        self._require_image()
        self._release_camera()
        message = self._camera_message()
        await self._send_state_update()
        return message

    @event(
        name="reset",
        description=(
            "Queue a fresh world from the selected image and current prompt. Requires a selected "
            "image. Clears progress, geometric memory, KV cache, and camera motion while "
            "resuming continuous generation. Emits `rollout_reset_queued` and broadcasts `state_update` "
            "on success, or `command_error` when no image is selected."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Seed for the fresh world. Use -1 to retain the active seed; a non-negative "
                "value replaces it when reset begins."
            ),
        ),
    ) -> RolloutResetQueued:
        """Queue a fresh world and report the state it replaces."""
        self._require_image()
        if seed >= 0:
            self._seed = seed
        replaced = self._chunk_index
        self._request_restart()
        message = RolloutResetQueued(seed=self._seed, replaced_chunks=replaced)
        await self._send_state_update()
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded reference image and queue a fresh continuously generated world. "
            "The selected image queues an initial 13-frame chunk. Emits `image_selected` and "
            "broadcasts `state_update` on "
            "success, or `command_error` for an invalid upload."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "Reference image uploaded through Reactor. JPEG, PNG, WebP, or BMP up to 25 MiB "
                "and 100 million pixels; EXIF orientation is applied before center cropping."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional prompt for the fresh world. An empty value preserves the current prompt "
                "or uses a generic continuation prompt when none exists."
            ),
        ),
    ) -> ImageSelected:
        """Validate an uploaded image and queue its visible initial chunk."""
        validate_uploaded_image(image)
        self._selected_input = image
        self._image_source = "uploaded"
        self._image_name = image.name
        self.state.prompt = (
            prompt.strip() or self.state.prompt.strip() or _DEFAULT_UPLOAD_PROMPT
        )
        self._request_restart()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
        )
        await self._send_state_update()
        return message

    @event(
        name="random_image",
        description=(
            "Select an official built-in image and its matching prompt, begin continuous "
            "generation, and queue the initial 13-frame chunk. Emits `image_selected` and broadcasts "
            "`state_update` on success, or `command_error` when no example is available."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a different official example when one is available."""
        if not self._examples:
            raise CommandError("image_unavailable", "No built-in images are available.")
        current = (
            self._selected_input if isinstance(self._selected_input, Path) else None
        )
        candidates = [example for example in self._examples if example.path != current]
        selected = secrets.choice(candidates or list(self._examples))
        self._selected_input = selected.path
        self._image_source = "built_in"
        self._image_name = selected.path.name
        self.state.prompt = selected.prompt
        self._request_restart()
        message = ImageSelected(
            source="built_in",
            filename=selected.path.name,
            prompt=selected.prompt,
        )
        await self._send_state_update()
        return message

    async def process_input(self) -> HYWorld15Input:
        """Snapshot one native chunk's conditioning and controls."""
        if self._selected_input is None:
            raise ApplicationError("Select an image before generating.")
        if self.state._limit_reached:
            raise ApplicationError("Reset the world after reaching the rollout limit.")
        prompt = self.state.prompt.strip()
        if not prompt:
            raise ApplicationError("HY-World 1.5 requires a non-empty prompt.")
        anchor = None
        if self.state._world_id != self.state._applied_world_id:
            image = self._selected_input
            anchor = HYWorld15Anchor(
                image=image.data if isinstance(image, UploadedFile) else image,
                seed=self._seed,
            )
        camera = self._require_planner().plan(
            CameraControl(
                forward=self.state._forward,
                strafe=self.state._strafe,
                pitch=self.state._pitch,
                yaw=self.state._yaw,
            )
        )
        snapshot = HYWorld15Input(
            world_id=self.state._world_id,
            anchor=anchor,
            prompt=prompt,
            viewmats=camera.viewmats,
            intrinsics=camera.intrinsics,
            actions=camera.actions,
        )
        self._generating = True
        return snapshot

    def generate(self, input: HYWorld15Input) -> HYWorld15Result:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> HYWorld15Output:
        """Publish completed frames and their sampled controls."""
        self._generating = False
        if outcome.error is not None:
            # A backend failure is not repaired by discarding geometric memory.
            raise outcome.error
        result: HYWorld15Result = outcome.result
        self.state._applied_world_id = result.world_id
        self._chunk_index = result.chunk_index
        self._active_prompt = result.prompt
        await self.send(
            ChunkCompleted(
                chunk=result.chunk_index,
                frames=int(result.frames.shape[0]),
                prompt=result.prompt,
                generation_seconds=outcome.elapsed,
                forward=self.state._forward,
                strafe=self.state._strafe,
                pitch=self.state._pitch,
                yaw=self.state._yaw,
            )
        )
        config = self._require_config()
        if result.complete:
            self.state._limit_reached = True
            self._release_camera()
            await self.send(
                RolloutLimitReached(
                    completed_chunks=self._chunk_index,
                    max_chunks=config.max_chunks,
                )
            )
        await self.send(self._state_update())
        return HYWorld15Output(main_video=result.frames)

    def _request_restart(self) -> None:
        """Queue a fresh causal world and clear prior playout, camera, and progress."""
        self.output.flush()
        self._release_camera()
        self.state._world_id += 1
        self._require_planner().reset()
        self.state._limit_reached = False
        self._chunk_index = 0
        self._active_prompt = None

    def _release_camera(self) -> None:
        """Return every trained camera control to neutral."""
        self.state._forward = 0.0
        self.state._strafe = 0.0
        self.state._pitch = 0.0
        self.state._yaw = 0.0

    def _next_control_chunk(self) -> int:
        """Return the next one-based boundary that can sample a command."""
        if self.state._world_id != self.state._applied_world_id:
            return 1
        return self._chunk_index + 1 + int(self._generating)

    def _camera_message(self) -> CameraMotionChanged:
        """Build the complete atomic camera event result."""
        return CameraMotionChanged(
            forward=self.state._forward,
            strafe=self.state._strafe,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
            applies_to_chunk=self._next_control_chunk(),
        )

    def _state_update(self) -> StateUpdate:
        """Build one complete client-facing shared world snapshot."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        next_chunk: int | None
        if self._selected_input is None or self.state._limit_reached:
            next_chunk = None
        else:
            next_chunk = self._next_control_chunk()
        return StateUpdate(
            image_source=self._image_source,
            image_name=self._image_name,
            prompt=self.state.prompt.strip() or None,
            active_prompt=self._active_prompt,
            seed=self._seed,
            reset_queued=(
                self._selected_input is not None
                and self.state._world_id != self.state._applied_world_id
            ),
            generating=self._generating,
            completed_chunks=self._chunk_index,
            next_chunk=next_chunk,
            next_chunk_frames=(
                None if next_chunk is None else 13 if next_chunk == 1 else 16
            ),
            max_chunks=max_chunks,
            limit_reached=self.state._limit_reached,
            forward=self.state._forward,
            strafe=self.state._strafe,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
        )

    async def _send_state_update(self) -> None:
        """Broadcast the complete state after a mutation or boundary."""
        await self.send(self._state_update())

    def _require_image(self) -> None:
        """Reject commands that need an initialized image-conditioned world."""
        if self._selected_input is None:
            raise CommandError("image_required", "Select an image before this command.")

    def _require_available_rollout(self) -> None:
        """Reject generation controls after the configured world limit."""
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset HY-World 1.5 before requesting another chunk.",
            )

    def _require_config(self) -> HYWorld15Config:
        """Return the loaded adapter configuration."""
        if self._config is None:
            raise RuntimeError("HY-World 1.5 was not loaded")
        return self._config

    def _require_planner(self) -> NativeCameraPlanner:
        """Return the initialized native camera planner."""
        if self._planner is None:
            raise RuntimeError("HY-World 1.5 camera planner was not loaded")
        return self._planner
