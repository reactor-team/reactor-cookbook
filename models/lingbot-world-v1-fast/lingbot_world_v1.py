"""The application half of LingBot-World v1 Fast: what a client sees and steers.

This is the ``ReactorApp`` the runtime drives one step at a time. It declares
the client contract (``lingbot_world_v1_types.py``), decides in
``process_input()`` whether a chunk can be generated and what the model gets,
forwards to the model half in a one-line ``generate()``, and writes every
client-visible effect of a step in ``process_output()``. The model half lives
in ``lingbot_world_v1_model.py`` and is reached only through
``LingbotV1Input`` and ``LingbotV1Result``.
"""

from __future__ import annotations

import random
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
    session_ended,
    session_started,
)
from reactor_runtime.distributed import DistributedRunner
from reactor_runtime.log import get_logger

from lingbot_world_v1_backend import WorkerSettings
from lingbot_world_v1_camera import CameraMotionPlanner, MotionConfig
from lingbot_world_v1_config import (
    LingBotConfig,
    Sample,
    prepare_runtime,
    read_config,
)
from lingbot_world_v1_images import (
    normalize_output_frames,
    upload_suffix,
    validate_uploaded_image,
)
from lingbot_world_v1_model import (
    AnchorImage,
    LingbotV1Input,
    LingbotV1Model,
    LingbotV1Result,
)
from lingbot_world_v1_types import (
    CameraMotionChanged,
    ImageSelected,
    LingBotWorldOutput,
    LingBotWorldState,
    PromptQueued,
    RolloutLimitReached,
    RolloutResetQueued,
    StateUpdate,
)

logger = get_logger(__name__)

FRAMES_PER_CHUNK = 12


class LingBotWorldV1(ReactorApp):
    """Generate an image-, prompt-, and camera-controllable LingBot world."""

    state: LingBotWorldState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: LingBotConfig | None = None
        self._engine: DistributedRunner | None = None
        self._planner: CameraMotionPlanner | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._selected_intrinsics: Path | None = None
        self._default_prompt = ""
        self._seed = 0
        self._chunk_index = 0
        self._last_chunk_seconds: float | None = None

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load LingBot-World-Fast once.

        Args:
            config_path: Path to ``lingbot_world_v1.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        prepare_runtime(config)
        self._config = config
        self._default_prompt = config.samples[0].prompt
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            MotionConfig(
                translation_units_per_latent=config.translation_units_per_latent,
                rotation_degrees_per_latent=config.rotation_degrees_per_latent,
            )
        )
        self._engine = DistributedRunner(
            LingbotV1Model,
            world_size=config.world_size,
            call_timeout=180.0,
            load_kwargs={
                "settings": WorkerSettings(
                    source_path=config.source_path,
                    checkpoint_dir=config.checkpoint.path,
                    runtime_root=config.runtime_root,
                    max_chunks=config.max_chunks,
                    context_latents=config.context_latents,
                    max_area=config.max_area,
                    shift=config.shift,
                )
            },
        )
        self._engine.start()
        logger.info(
            "LingBot-World v1 Fast ready",
            source_revision=config.source_revision,
            fast_revision=config.fast_checkpoint.revision,
            context_latents=config.context_latents,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty shared world before its first image selection."""
        config = self._require_config()
        self.state.prompt = ""
        self._selected_input = None
        self._selected_intrinsics = None
        self._seed = config.seed
        self._clear_controls()
        self._planner_reset()
        self._chunk_index = 0
        self._last_chunk_seconds = None

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera axes when their viewer disconnects."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        """Release causal caches while retaining loaded weights."""
        try:
            if self._engine is not None:
                self._engine.reset()
        finally:
            self._clear_controls()
            self._selected_input = None
            self._selected_intrinsics = None
            self._chunk_index = 0
            self._last_chunk_seconds = None

    @event(
        name="set_image",
        description=(
            "Replace the anchor image and start a fresh world with continuous `main_video` "
            "generation. Valid at any time; progress and camera axes reset, and the upload is "
            "decoded before the command succeeds. Emits `image_selected` and broadcasts "
            "`state_update` on success, or `command_error` for invalid image bytes, media "
            "type, dimensions, or an empty prompt."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "Anchor image delivered through Reactor's upload protocol. JPEG, PNG, WebP, or "
                "BMP; non-empty, at most 25 MiB and 100 million pixels."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene prompt up to 4096 characters. A non-empty value conditions the "
                "fresh world; an empty value preserves the active prompt."
            ),
        ),
    ) -> ImageSelected:
        """Select an upload and begin a fresh continuous rollout."""
        validate_uploaded_image(image)
        normalized = prompt.strip() or self.state.prompt.strip() or self._default_prompt
        if not normalized:
            raise CommandError("prompt_required", "LingBot-World requires a prompt.")
        self._selected_input = image
        if self._selected_intrinsics is None:
            self._selected_intrinsics = self._require_config().samples[0].intrinsics
        self.state.prompt = normalized
        self._request_restart()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=normalized,
            seed=self._seed,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="random_image",
        description=(
            "Select one public LingBot demo image and its matching prompt and calibration, then "
            "start a fresh world with continuous `main_video` generation. Valid at any time; "
            "progress resets. Emits `image_selected` and broadcasts `state_update` on success."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a built-in sample and begin a fresh continuous rollout."""
        sample = random.choice(self._require_config().samples)
        self._select_sample(sample)
        self._request_restart()
        message = ImageSelected(
            source="built_in",
            filename=sample.image.name,
            prompt=self.state.prompt,
            seed=self._seed,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_prompt",
        description=(
            "Set text conditioning for the next generated chunk without clearing visual self-KV "
            "history. Valid after an anchor is selected and before the rollout limit. Emits "
            "`prompt_queued` and broadcasts `state_update` on success, or `command_error` when "
            "the prompt is empty, no image is selected, or a fresh rollout is required."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene description up to 4096 characters. It replaces the active "
                "cross-attention context at the next chunk boundary while self-KV history stays."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt change and report its first affected chunk."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "LingBot-World requires a prompt.")
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an anchor image before setting a prompt."
            )
        self._require_available_rollout()
        self.state.prompt = normalized
        message = PromptQueued(
            prompt=normalized, applies_to_chunk=self._next_control_chunk()
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_forward",
        description=(
            "Set backward-to-forward camera translation for the next chunk. Valid before the "
            "rollout limit; the value is held for later chunks. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` until a fresh rollout is "
            "started after the limit."
        ),
    )
    async def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized backward (-1) to forward (1) translation. Zero stops this axis; the "
                "value is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> CameraMotionChanged:
        """Set forward motion and report the complete held camera state."""
        return await self._set_axis("forward", forward)

    @event(
        name="set_strafe",
        description=(
            "Set left-to-right camera translation for the next chunk. Valid before the rollout "
            "limit; the value is held for later chunks. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` until a fresh rollout is "
            "started after the limit."
        ),
    )
    async def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized left (-1) to right (1) translation. Zero stops this axis; the value "
                "is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> CameraMotionChanged:
        """Set strafe motion and report the complete held camera state."""
        return await self._set_axis("strafe", strafe)

    @event(
        name="set_vertical",
        description=(
            "Set down-to-up camera translation for the next chunk. Valid before the rollout "
            "limit; the value is held for later chunks. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` until a fresh rollout is "
            "started after the limit."
        ),
    )
    async def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized down (-1) to up (1) translation. Zero stops this axis; the value is "
                "sampled at the next chunk boundary and held."
            ),
        ),
    ) -> CameraMotionChanged:
        """Set vertical motion and report the complete held camera state."""
        return await self._set_axis("vertical", vertical)

    @event(
        name="set_pitch",
        description=(
            "Set downward-to-upward camera pitch for the next chunk. Valid before the rollout "
            "limit; the value is held for later chunks. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` until a fresh rollout is "
            "started after the limit."
        ),
    )
    async def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized downward (-1) to upward (1) pitch. Zero stops this axis; the value "
                "is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> CameraMotionChanged:
        """Set pitch and report the complete held camera state."""
        return await self._set_axis("pitch", pitch)

    @event(
        name="set_yaw",
        description=(
            "Set left-to-right camera yaw for the next chunk. Valid before the rollout limit; "
            "the value is held for later chunks. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` until a fresh rollout is "
            "started after the limit."
        ),
    )
    async def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized left (-1) to right (1) yaw. Zero stops this axis; the value is "
                "sampled at the next chunk boundary and held."
            ),
        ),
    ) -> CameraMotionChanged:
        """Set yaw and report the complete held camera state."""
        return await self._set_axis("yaw", yaw)

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise-to-clockwise camera roll for the next chunk. Valid before the "
            "rollout limit; the value is held for later chunks. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` until a fresh rollout is "
            "started after the limit."
        ),
    )
    async def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized counterclockwise (-1) to clockwise (1) roll. Zero stops this axis; "
                "the value is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> CameraMotionChanged:
        """Set roll and report the complete held camera state."""
        return await self._set_axis("roll", roll)

    @event(
        name="reset",
        description=(
            "Restart from the selected image and prompt with continuous generation from chunk "
            "one. Valid when an anchor exists; progress, caches, and camera axes reset. Emits "
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
                "Seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain the active "
                "seed; a non-negative value becomes active when reset begins."
            ),
        ),
    ) -> RolloutResetQueued:
        """Queue a reproducible fresh rollout and report the state it replaces."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an anchor image before resetting."
            )
        if seed >= 0:
            self._seed = seed
        replaced = self._chunk_index
        self._request_restart()
        message = RolloutResetQueued(seed=self._seed, replaced_chunks=replaced)
        await self.send(self._state_update())
        return message

    async def process_input(self) -> LingbotV1Input:
        """Refuse until a world can be generated; otherwise say what the model gets.

        The anchor image rides on the input only until the model has reported
        the requested world back on a result, so it crosses once per fresh
        world and not once per chunk.
        """
        if self._selected_input is None or self._selected_intrinsics is None:
            raise ApplicationError("Select an anchor image before generating.")
        if self.state._limit_reached:
            raise ApplicationError("Reset the world after reaching the rollout limit.")
        planner = self._planner
        if planner is None:
            raise RuntimeError("LingBot-World v1 was not loaded")
        anchor: AnchorImage | None = None
        if self.state._world_id != self.state._applied_world_id:
            selected = self._selected_input
            if isinstance(selected, UploadedFile):
                image: Path | bytes = selected.data
                suffix = upload_suffix(selected)
            else:
                image = selected
                suffix = selected.suffix
            anchor = AnchorImage(
                image=image,
                suffix=suffix,
                intrinsics=self._selected_intrinsics,
                seed=self._seed,
            )
        poses = planner.plan_chunk(
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            forward=self.state.forward,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
        )
        return LingbotV1Input(
            world_id=self.state._world_id,
            anchor=anchor,
            prompt=self.state.prompt,
            poses=poses,
        )

    def generate(self, input: LingbotV1Input) -> LingbotV1Result:
        """One chunk of the world. The model half does the work."""
        if self._engine is None:
            raise RuntimeError("LingBot-World v1 was not loaded")
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> LingBotWorldOutput:
        """Publish one completed chunk and its state, propagating model failures."""
        if outcome.error is not None:
            raise outcome.error
        result: LingbotV1Result = outcome.result
        frames = normalize_output_frames(result.frames)
        self.state._applied_world_id = result.world_id
        self._chunk_index = result.chunk_index
        self._last_chunk_seconds = outcome.elapsed
        config = self._require_config()
        if self._chunk_index >= config.max_chunks:
            self.state._limit_reached = True
            self._clear_controls()
            await self.send(
                RolloutLimitReached(
                    completed_chunks=self._chunk_index,
                    max_chunks=config.max_chunks,
                )
            )
        await self.send(self._state_update())
        return LingBotWorldOutput(main_video=frames)

    async def _set_axis(self, name: str, value: float) -> CameraMotionChanged:
        """Set one validated axis, broadcast state, and report held camera motion."""
        self._require_available_rollout()
        setattr(self.state, name, value)
        message = self._camera_changed()
        await self.send(self._state_update())
        return message

    def _camera_changed(self) -> CameraMotionChanged:
        """Return the complete held camera state after a mutation."""
        return CameraMotionChanged(
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
            applies_to_chunk=self._next_control_chunk(),
        )

    def _select_sample(self, sample: Sample) -> None:
        """Make one public sample the active image, calibration, and prompt."""
        self._selected_input = sample.image
        self._selected_intrinsics = sample.intrinsics
        self.state.prompt = sample.prompt

    def _clear_controls(self) -> None:
        """Return every camera axis to neutral."""
        self.state.forward = 0.0
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _request_restart(self) -> None:
        """Ask the next chunk for a fresh world and release active camera motion.

        Bumping the world id is the whole request: the next ``process_input()``
        carries the anchor because the model has not reported this id yet.
        """
        self._clear_controls()
        self._planner_reset()
        self.state._world_id += 1
        self.state._limit_reached = False
        self._chunk_index = 0
        self._last_chunk_seconds = None
        self.output.flush()

    def _planner_reset(self) -> None:
        """Return the camera to the anchor pose for a fresh world."""
        if self._planner is not None:
            self._planner.reset()

    def _require_available_rollout(self) -> None:
        """Reject controls that cannot apply until a fresh rollout starts."""
        if self._selected_input is None:
            raise CommandError(
                "image_required",
                "Upload an image or select a random image before this command.",
            )
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset LingBot-World or select an image before requesting another chunk.",
            )

    def _require_config(self) -> LingBotConfig:
        """Return loaded configuration or fail with a lifecycle error."""
        if self._config is None:
            raise RuntimeError("LingBot-World v1 was not loaded")
        return self._config

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to consume newly accepted controls."""
        if self.state._world_id != self.state._applied_world_id:
            return 1
        return self._chunk_index + 1

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of shared world state."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        selected = self._selected_input
        if isinstance(selected, UploadedFile):
            image_source = "uploaded"
        elif selected is None:
            image_source = "none"
        else:
            image_source = "built_in"
        image_name = selected.name if selected is not None else ""
        if selected is None or self.state._limit_reached:
            next_chunk = None
            next_chunk_frames = None
        else:
            next_chunk = self._next_control_chunk()
            next_chunk_frames = 9 if next_chunk == 1 else 12
        return StateUpdate(
            prompt=self.state.prompt,
            image_source=image_source,
            image_name=image_name,
            seed=self._seed,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            last_chunk_seconds=self._last_chunk_seconds,
            next_chunk=next_chunk,
            next_chunk_frames=next_chunk_frames,
            max_chunks=max_chunks,
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
        )
