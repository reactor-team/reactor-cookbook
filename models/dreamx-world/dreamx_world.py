"""Serve stateful DreamX-World autoregressive inference through Reactor Runtime."""

from __future__ import annotations

import secrets
from pathlib import Path

from dreamx_assets import prepare_runtime_assets, read_config
from dreamx_camera import FRAMES_PER_CHUNK, DreamXCameraController
from dreamx_images import validate_uploaded_image
from dreamx_types import (
    ActionChanged,
    ChunkGenerated,
    DreamXWorldOutput,
    DreamXWorldState,
    ImageSelected,
    ImageSource,
    PromptQueued,
    RolloutResetQueued,
    StateUpdate,
)
from dreamx_world_model import (
    DreamXAnchor,
    DreamXConfig,
    DreamXInput,
    DreamXModel,
    DreamXResult,
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
    session_ended,
    session_started,
)

_CAMERA_KEYS = ["w", "a", "s", "d", "i", "j", "k", "l"]


class DreamXWorld(ReactorApp):
    """Generate an image-, prompt-, and keyboard-controllable DreamX world."""

    state: DreamXWorldState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: DreamXConfig | None = None
        self._engine = DreamXModel()
        self._camera: DreamXCameraController | None = None
        self._scene_prompts: dict[Path, str] = {}
        self._selected_input: Path | UploadedFile | None = None
        self._image_source: ImageSource | None = None
        self._seed = 0
        self._chunk_index = 0
        self._active_prompt = ""
        self._generating = False

    def load(self, config_path: Path | None) -> None:
        """Prepare pinned public assets and load DreamX-World once on the GPU.

        Args:
            config_path: Path to ``dreamx_world.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        scene_prompts = prepare_runtime_assets(config)
        self._config = config
        self._scene_prompts = scene_prompts
        self._camera = DreamXCameraController(config.motion_speed)
        self._engine.load(config)
        self._seed = config.seed

    @session_started
    def on_session_started(self) -> None:
        """Wait for an uploaded or randomly selected image before generating."""
        config = self._require_config()
        self.state.prompt = ""
        self._clear_controls()
        self.state._applied_world_id = None
        self._selected_input = None
        self._image_source = None
        self._seed = config.seed
        self._chunk_index = 0
        self._active_prompt = ""
        self._generating = False

    @session_ended
    def on_session_ended(self) -> None:
        """Release image and causal rollout state when the shared session ends."""
        try:
            self._engine.reset()
        finally:
            self._clear_controls()
            self.state._applied_world_id = None
            self._selected_input = None
            self._image_source = None
            self._chunk_index = 0
            self._active_prompt = ""
            self._generating = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera keys when a viewer leaves the live session."""
        self._clear_controls()
        await self.send(self._state_update())

    @event(
        name="set_image",
        description=(
            "Select an uploaded image and queue a fresh world. Valid at any time; the image "
            "applies to chunk 1, clears rollout progress, releases held keys, and starts "
            "continuous generation. Emits "
            "`image_selected`, `state_update`, and then `chunk_generated` on success, or "
            "`command_error` when the upload is invalid."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "Starting image sent through the Reactor upload protocol. JPEG, PNG, WebP, or "
                "BMP; at most 25 MiB and 100 million pixels. DreamX resizes it to 1280x704."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene or event prompt for the fresh world's first chunk. A blank "
                "value uses the active prompt or the configured upload default."
            ),
        ),
    ) -> ImageSelected:
        """Validate an upload, queue a fresh rollout, and identify its effective input."""
        validate_uploaded_image(image)
        config = self._require_config()
        effective_prompt = (
            prompt.strip() or self.state.prompt.strip() or config.default_upload_prompt
        )
        if not effective_prompt:
            raise CommandError(
                "prompt_required", "DreamX-World requires a non-empty prompt."
            )
        self._select_image(image, "uploaded", effective_prompt)
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=effective_prompt,
            applies_to_chunk=1,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="random_image",
        description=(
            "Select one configured upstream demo image and queue a fresh world. Valid at any "
            "time; the paired upstream prompt applies to chunk 1, progress is cleared, held "
            "keys are released, and generation starts continuously. Emits `image_selected`, "
            "`state_update`, and then "
            "`chunk_generated`; configured images are validated while the model loads."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Choose a built-in scene and identify its paired upstream prompt."""
        config = self._require_config()
        image = secrets.choice(config.random_images).resolve()
        prompt = self._scene_prompts[image]
        self._select_image(image, "built_in", prompt)
        message = ImageSelected(
            source="built_in",
            filename=image.name,
            prompt=prompt,
            applies_to_chunk=1,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_prompt",
        description=(
            "Queue a new scene or event prompt without clearing causal visual history. Valid "
            "after an image is selected; the text refreshes cross-attention at the next native "
            "chunk boundary while preserving the rolling visual KV cache. Emits "
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
                "Non-empty scene or event text, trimmed and applied at the next native chunk "
                "boundary without restarting the world."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and confirm the first chunk expected to consume it."""
        self._require_selected_image()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "DreamX-World requires a non-empty prompt."
            )
        self.state.prompt = normalized
        message = PromptQueued(prompt=normalized, applies_to_chunk=self._next_chunk())
        await self.send(self._state_update())
        return message

    @event(
        name="set_key_state",
        description=(
            "Hold or release one native DreamX camera key for subsequent chunks. Valid after "
            "an image is selected. Emits `action_changed` and `state_update` after changing "
            "the complete held-key set. Unsupported values are rejected before state changes."
        ),
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=_CAMERA_KEYS,
            description=(
                "Native DreamX camera key to hold or release: w/s move forward/backward, a/d "
                "strafe left/right, i/k tilt up/down, and j/l pan left/right. When accepted, "
                "the key persists across chunks until another command releases it."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true to hold `key` for subsequent chunks or false to release it. The new "
                "state is sampled at the next native chunk boundary."
            ),
        ),
    ) -> ActionChanged:
        """Update one held camera key and report the complete native input state."""
        self._require_selected_image()
        if pressed:
            self.state._pressed_keys = self.state._pressed_keys.union((key,))
        else:
            self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        await self.send(self._state_update())
        return ActionChanged(
            pressed_keys=[
                candidate
                for candidate in _CAMERA_KEYS
                if candidate in self.state._pressed_keys
            ],
        )

    @event(
        name="reset",
        description=(
            "Restart from the selected image and queued prompt. Valid after image selection; "
            "the reset applies before the next native chunk, clears every autoregressive and "
            "VAE cache, releases held keys, and resumes continuous generation. Emits "
            "`rollout_reset_queued` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Seed for the fresh rollout from 0 to 2147483647. Use -1 to retain the active "
                "seed; a non-negative value becomes active when reset begins."
            ),
        ),
    ) -> RolloutResetQueued:
        """Queue a cache reset and report the rollout being replaced."""
        self._require_selected_image()
        if seed >= 0:
            self._seed = seed
        completed = self._chunk_index
        self._request_reset()
        message = RolloutResetQueued(
            trigger="manual",
            seed=self._seed,
            completed_chunks=completed,
            applies_to_chunk=1,
        )
        await self.send(self._state_update())
        return message

    async def process_input(self) -> DreamXInput:
        """Snapshot one native step after an image is selected."""
        if self._selected_input is None:
            raise ApplicationError("Select an image before generating.")
        prompt = self.state.prompt.strip()
        if not prompt:
            raise ApplicationError("DreamX-World requires a non-empty prompt.")
        if self._camera is None:
            raise RuntimeError("DreamX-World was not loaded")
        new_world = self.state._world_id != self.state._applied_world_id
        anchor = None
        if new_world:
            self._camera.reset()
            image = self._selected_input
            anchor = DreamXAnchor(
                image.data if isinstance(image, UploadedFile) else image, self._seed
            )
        camera = self._camera.plan_chunk(
            self.state._pressed_keys, first_chunk=new_world
        )
        snapshot = DreamXInput(
            world_id=self.state._world_id,
            anchor=anchor,
            prompt=prompt,
            poses=camera.poses,
            reference_pose=camera.reference_pose,
        )
        self._generating = True
        return snapshot

    def generate(self, input: DreamXInput) -> DreamXResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> DreamXWorldOutput:
        """Publish a completed chunk and queue bounded rollout resets."""
        self._generating = False
        if outcome.error is not None:
            # Contract violations and GPU failures cannot be repaired by a silent reset.
            raise outcome.error
        result: DreamXResult = outcome.result
        self.state._applied_world_id = result.world_id
        self._chunk_index = result.chunk_index
        self._active_prompt = result.prompt
        await self.send(
            ChunkGenerated(
                chunk=self._chunk_index,
                frames=int(result.frames.shape[0]),
                prompt=result.prompt,
                pressed_keys=[
                    key for key in _CAMERA_KEYS if key in self.state._pressed_keys
                ],
                inference_seconds=round(outcome.elapsed, 3),
            )
        )
        if result.complete:
            self._clear_controls()
            self.state._world_id += 1
            await self.send(
                RolloutResetQueued(
                    trigger="automatic_chunk_limit",
                    seed=self._seed,
                    completed_chunks=self._chunk_index,
                    applies_to_chunk=1,
                )
            )
        await self.send(self._state_update())
        return DreamXWorldOutput(main_video=result.frames)

    def _select_image(
        self,
        image: Path | UploadedFile,
        source: ImageSource,
        prompt: str,
    ) -> None:
        """Queue one selected image and prompt as a fresh continuous rollout."""
        self._selected_input = image
        self._image_source = source
        self.state.prompt = prompt
        self._request_reset()
        self._chunk_index = 0
        self._active_prompt = ""

    def _request_reset(self) -> None:
        """Queue a fresh backend rollout and clear controls consumed by the old one."""
        self.output.flush()
        self._clear_controls()
        self.state._world_id += 1

    def _clear_controls(self) -> None:
        """Release every native DreamX camera key."""
        self.state._pressed_keys = frozenset()

    def _next_chunk(self) -> int:
        """Return the one-based chunk expected to consume newly queued controls."""
        if self.state._world_id != self.state._applied_world_id:
            return 1
        return self._chunk_index + 1 + int(self._generating)

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of the shared world state."""
        config = self._config
        selected = self._selected_input
        image_name = selected.name if selected is not None else None
        return StateUpdate(
            image_source=self._image_source,
            image_name=image_name,
            prompt=self.state.prompt.strip() or None,
            active_prompt=self._active_prompt or None,
            pressed_keys=[
                key for key in _CAMERA_KEYS if key in self.state._pressed_keys
            ],
            seed=self._seed,
            reset_queued=(
                selected is not None
                and self.state._world_id != self.state._applied_world_id
            ),
            generating=self._generating,
            completed_chunks=self._chunk_index,
            next_chunk=None if selected is None else self._next_chunk(),
            max_chunks=config.max_chunks_per_rollout if config is not None else 0,
        )

    def _require_config(self) -> DreamXConfig:
        """Return loaded configuration or fail before serving partial state."""
        if self._config is None:
            raise RuntimeError("DreamX-World was not loaded")
        return self._config

    def _require_selected_image(self) -> None:
        """Reject commands whose result cannot apply before image selection."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an image before controlling DreamX-World."
            )
