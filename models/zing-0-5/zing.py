"""Serve one native Zing 0.5 autoregressive video block per Reactor turn."""

from __future__ import annotations

import asyncio
from pathlib import Path
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    ApplicationError,
    StepOutcome,
    get_weights_path,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

from zing_assets import (
    ZingAdapterConfig,
    read_config,
    configure_environment,
    prepare_assets,
)
from zing_images import prepare_image
from zing_model import ZingModel, ZingInput, ZingResult
from zing_types import (
    ActionChanged,
    ChunkCompleted,
    ControlsReleased,
    ImageSelected,
    PromptQueued,
    RolloutLimitReached,
    RolloutReset,
    StateUpdate,
    ZingOutput,
    ZingState,
    ZingKey,
)

logger = get_logger(__name__)


class Zing(ReactorApp):
    """Generate a controllable Zing 0.5 world from text or one initial image."""

    state: ZingState
    buffer_size = 16

    def __init__(self) -> None:
        super().__init__()
        self._config: ZingAdapterConfig | None = None
        self.engine: ZingModel

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path, get_weights_path())
        self._config = config
        configure_environment(config)
        prepare_assets(config)
        self.engine = ZingModel()
        self.engine.load(config)
        logger.info(
            "Zing 0.5 ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.asset_revision,
            cache_window="97/9",
            frames_per_chunk=16,
        )

    @session_started
    def on_session_started(self) -> None:
        config = self._require_config()
        self.state.prompt = ""
        self.state._pressed_keys = frozenset()
        self.state._applied_world_id = None
        self.state._conditioning = "none"
        self.state._image = None
        self.state._image_name = None
        self.state._seed = config.seed
        self.state._active_prompt = None
        self.state._completed_chunks = 0
        self.state._limit_reached = False
        self.state._world_epoch = 0

    @session_ended
    def on_session_ended(self) -> None:
        self.engine.reset()
        self.output.flush()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held controls when a viewer disconnects."""
        self.state._pressed_keys = frozenset()
        await self.send(self._state_update())

    @event(
        name="set_prompt",
        description=(
            "Set the text condition without restarting the current world. Before generation, the "
            "prompt starts a text-to-video world; during generation, the normalized text is "
            "sampled when the next chunk begins and preserves prior world history. Emits "
            "`prompt_queued` and broadcasts `state_update` on success, or `command_error` for "
            "empty text."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene, appearance, subject, camera, and motion description, up to "
                "4096 characters. Whitespace is trimmed and the result is sampled when the next "
                "chunk starts."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and report the chunk expected to consume it."""
        normalized = prompt.strip()
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "The current world is exhausted; select an image or explicitly reset "
                "to start a new world.",
            )
        if not normalized:
            raise CommandError("empty_prompt", "Zing requires a non-empty prompt.")
        initial = (
            self.state._completed_chunks == 0 and self.state._active_prompt is None
        )
        self.state.prompt = normalized
        starts_text_rollout = initial and self.state._image is None
        if starts_text_rollout:
            self.state._conditioning = "text"
            self.state._image_name = None
            self._request_reset()
        message = PromptQueued(
            prompt=normalized,
            applies_to_chunk=self.state._completed_chunks + 1,
            resets_rollout=starts_text_rollout,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded anchor image and queue a fresh world with continuous generation. "
            "Valid at any time; the image replaces prior world history before chunk one. Emits "
            "`image_selected` and broadcasts `state_update` on success, or `command_error` when "
            "the upload is empty, oversized, mislabeled, or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description=(
                "Anchor uploaded through Reactor as JPEG, PNG, WebP, or BMP, up to 25 MiB and "
                "100 million pixels. EXIF orientation is applied before resizing to `main_video`."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene and motion description for the fresh world. An empty value uses "
                "the configured image-neutral prompt."
            ),
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> ImageSelected:
        """Validate an uploaded image and select it for a fresh world."""
        config = self._require_config()
        pixels = await asyncio.to_thread(
            prepare_image, image, config.width, config.height
        )
        if seed >= 0:
            self.state._seed = seed
        self.state.prompt = prompt.strip() or config.default_prompt
        self.state._conditioning = "uploaded"
        self.state._image = pixels
        self.state._image_name = image.name
        self._request_reset()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
            seed=self.state._seed,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="example_image",
        description=(
            "Select Zing's public example image and matching prompt, then queue a fresh world "
            "with continuous generation. Valid whenever the configured example is available and "
            "takes no input. Emits `image_selected` and broadcasts `state_update` on success, or "
            "`command_error` when the example is unavailable."
        ),
    )
    async def example_image(self) -> ImageSelected:
        """Select the public example image for a fresh world."""
        config = self._require_config()
        image = Path(__file__).parent / "example_images" / "case0.jpg"
        if not image.is_file():
            raise CommandError(
                "example_unavailable", "The example image is unavailable"
            )
        try:
            data = await asyncio.to_thread(image.read_bytes)
        except OSError as error:
            raise CommandError(
                "example_unavailable", "The example image cannot be read"
            ) from error
        pixels = await asyncio.to_thread(
            prepare_image,
            UploadedFile(image.name, "image/jpeg", data),
            config.width,
            config.height,
        )
        self.state.prompt = config.example_prompt
        self.state._conditioning = "built_in"
        self.state._image = pixels
        self.state._image_name = image.name
        self._request_reset()
        await self.send(self._state_update())
        return ImageSelected(
            source="built_in",
            filename=image.name,
            prompt=self.state.prompt,
            seed=self.state._seed,
        )

    @event(
        name="set_key",
        description=(
            "Press or release one held native key for forthcoming chunks. "
            "`w`: move forward; `a`: strafe left; `s`: move backward; `d`: strafe right; "
            "`i`: look up; `j`: look left; `k`: look down; `l`: look right. The complete "
            "held state is sampled when the next chunk begins and remains active until changed or "
            "released. Emits `action_changed` and broadcasts `state_update` on success."
        ),
    )
    async def set_key(
        self,
        key: ZingKey = InputField(
            description=(
                "Native key to change: `w`: move forward; `a`: strafe left; "
                "`s`: move backward; `d`: strafe right; `i`: look up; "
                "`j`: look left; `k`: look down; `l`: look right. "
            )
        ),
        pressed: bool = InputField(
            description="Whether to hold or release `key`; the change applies to the next chunk."
        ),
    ) -> ActionChanged:
        """Change one held control and report the complete held state."""
        if pressed and self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "The current world is exhausted; controls can only be released "
                "until an explicit reset.",
            )
        keys = set(self.state._pressed_keys)
        (keys.add if pressed else keys.discard)(key)
        self.state._pressed_keys = frozenset(keys)
        await self.send(self._state_update())
        return ActionChanged(
            key=key,
            pressed=pressed,
            pressed_keys=sorted(keys),
            applies_to_chunk=None
            if self.state._limit_reached
            else self.state._completed_chunks + 1,
        )

    @event(
        name="release_controls",
        description=(
            "Release every held movement and look control for forthcoming chunks. Neutral input "
            "is sampled when the next chunk begins. Emits `controls_released` and broadcasts "
            "`state_update` on success."
        ),
    )
    async def release_controls(self) -> ControlsReleased:
        """Release all held controls and report which keys changed."""
        released = sorted(self.state._pressed_keys)
        self.state._pressed_keys = frozenset()
        await self.send(self._state_update())
        return ControlsReleased(
            released_keys=released,
            applies_to_chunk=None
            if self.state._limit_reached
            else self.state._completed_chunks + 1,
        )

    @event(
        name="reset",
        description=(
            "Queue a fresh world from the selected text or image condition and current prompt. "
            "Use after selecting a prompt or image; the reset clears progress, releases held "
            "controls, and resumes continuous generation from chunk one. Emits `rollout_reset` "
            "and broadcasts `state_update` on success."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> RolloutReset:
        """Queue a fresh world and report the progress it replaces."""
        if self.state._conditioning == "none":
            raise CommandError(
                "scene_required", "Select a prompt or image before resetting"
            )
        if seed >= 0:
            self.state._seed = seed
        replaced = self.state._completed_chunks
        self._request_reset()
        await self.send(self._state_update())
        return RolloutReset(seed=self.state._seed, replaced_chunks=replaced)

    async def process_input(self) -> ZingInput:
        if self.state._conditioning == "none":
            raise ApplicationError("no prompt or image selected")
        if self.state._limit_reached:
            raise ApplicationError("rollout limit reached")
        fresh = self.state._world_epoch != self.state._applied_world_id
        return ZingInput(
            world_id=self.state._world_epoch,
            image=self.state._image if fresh else None,
            prompt=self.state.prompt,
            seed=self.state._seed,
            pressed_keys=self.state._pressed_keys,
            image_required=self.state._conditioning in {"uploaded", "built_in"},
        )

    def generate(self, input: ZingInput) -> ZingResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> ZingOutput | None:
        if outcome.error is not None:
            # Input refusal prevents exhausted/unseeded worlds; remaining failures
            # indicate an invalid contract or native inference failure and are fatal.
            raise outcome.error
        result: ZingResult = outcome.result
        self.state._applied_world_id = result.world_id
        self.state._completed_chunks = result.index
        self.state._active_prompt = self.state.prompt
        await self.send(
            ChunkCompleted(
                chunk=result.index,
                video_frames=int(result.frames.shape[0]),
                generation_seconds=outcome.elapsed,
                prompt=self.state.prompt,
                action_keys=sorted(self.state._pressed_keys),
            )
        )
        if result.complete:
            self.state._limit_reached = True
            self.state._pressed_keys = frozenset()
            await self.send(
                RolloutLimitReached(
                    completed_chunks=result.index,
                    max_chunks=self._require_config().max_chunks,
                    world_epoch=result.world_id,
                )
            )
        await self.send(self._state_update())
        return ZingOutput(main_video=result.frames)

    def _request_reset(self) -> None:
        self.state._pressed_keys = frozenset()
        self.state._active_prompt = None
        self.state._completed_chunks = 0
        self.state._limit_reached = False
        self.state._world_epoch += 1
        self.output.flush()

    def _state_update(self) -> StateUpdate:
        return StateUpdate(
            prompt=self.state.prompt,
            active_prompt=self.state._active_prompt,
            pressed_keys=sorted(self.state._pressed_keys),
            conditioning=self.state._conditioning,
            image_name=self.state._image_name,
            seed=self.state._seed,
            completed_chunks=self.state._completed_chunks,
            reset_queued=self.state._world_epoch != self.state._applied_world_id
            and self.state._world_epoch > 0,
            max_chunks=self._config.max_chunks if self._config is not None else 0,
            limit_reached=self.state._limit_reached,
            world_epoch=self.state._world_epoch,
        )

    def _require_config(self) -> ZingAdapterConfig:
        if self._config is None:
            raise RuntimeError("Zing is not loaded")
        return self._config
