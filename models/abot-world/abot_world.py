"""Serve ABot-World's native causal rollout through Reactor Runtime.

The adapter imports the pinned upstream causal pipeline directly. Each Reactor
turn samples W/A/S/D movement and I/J/K/L view controls once, generates one
three-latent block with the upstream rolling KV cache, and emits the decoded RGB
chunk. Prompt changes rebuild only the upstream cross-attention condition; image
changes and resets initialize a fresh autoregressive world.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from abot_world_assets import (
    ABotWorldConfig,
    prepare_assets,
    read_config,
)
from abot_world_controls import (
    KEY_CHOICES,
    KEY_ORDER,
    sample_key_snapshot,
    update_key_state,
)
from abot_world_images import upload_suffix, validate_uploaded_image
from abot_world_model import ABotAnchor, ABotInput, ABotResult, ABotWorldModel
from abot_world_types import (
    DEFAULT_PROMPT,
    ABotWorldOutput,
    ABotWorldState,
    ActionChanged,
    ControlsReleased,
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

FRAMES_PER_CHUNK = 12


class ABotWorld(ReactorApp):
    """Generate a prompt-, image-, and keyboard-controlled ABot world."""

    state: ABotWorldState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: ABotWorldConfig | None = None
        self._engine = ABotWorldModel()
        self._selected_input: Path | UploadedFile | None = None
        self._image_source: str | None = None
        self._image_name: str | None = None
        self._active_prompt = ""
        self._sampled_keys: frozenset[str] = frozenset()
        self._chunk_index = 0
        self._reset_in_flight = False
        self._chunk_in_flight = False

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load the native causal model once."""
        weights_root = get_weights_path()
        config = read_config(config_path, weights_root)
        prepare_assets(config, weights_root)
        self._config = config
        self._engine.load(config, weights_root)

    @session_started
    def on_session_started(self) -> None:
        """Initialize one shared world before its first viewer connects."""
        config = self._require_config()
        self.state.prompt = DEFAULT_PROMPT
        self.state._seed = config.seed
        self.state._world_id = 0
        self.state._applied_world_id = None
        self.state._limit_reached = False
        self._clear_controls()
        self._selected_input = None
        self._image_source = None
        self._image_name = None
        self._active_prompt = ""
        self._sampled_keys = frozenset()
        self._chunk_index = 0
        self._reset_in_flight = False
        self._chunk_in_flight = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared state to one joining viewer."""
        await self._send_state_update(client)

    @disconnected
    async def on_disconnected(self) -> None:
        """Release native controls when a viewer leaves the live session."""
        self._clear_controls()
        await self._send_state_update()

    @session_ended
    def on_session_ended(self) -> None:
        """Release session inputs while keeping loaded model weights resident."""
        self._clear_controls()
        self._selected_input = None
        self._image_source = None
        self._image_name = None
        self._active_prompt = ""
        self._sampled_keys = frozenset()
        self._chunk_index = 0
        self._engine.reset()
        self.state._applied_world_id = None

    @event(
        name="set_key_state",
        description=(
            "Hold or release one native ABot-World action key. Valid before the rollout limit; "
            "the state is sampled at the next autoregressive chunk boundary and held for later "
            "chunks. A press followed by a release before sampling is retained as one short tap. "
            "Emits `action_changed` and broadcasts `state_update` on success, or "
            "`command_error` with `rollout_limit_reached` when a fresh world is required."
        ),
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="W",
            choices=KEY_CHOICES,
            description=(
                "Native action key: W/S move forward/backward, A/D move left/right, I/K look "
                "up/down, and J/L look left/right. Multiple non-opposing keys can be held for "
                "combined movement and view actions."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true to hold `key` across chunks or false to release it. A released key "
                "still reaches the next chunk once when it was pressed since the last sample."
            ),
        ),
    ) -> ActionChanged:
        """Update one native key while preserving upstream short-tap behavior."""
        self._require_available_rollout()
        self.state._pressed_keys, self.state._activated_keys = update_key_state(
            self.state._pressed_keys,
            self.state._activated_keys,
            key=key,
            pressed=pressed,
        )
        message = ActionChanged(
            key=key,
            pressed=pressed,
            pressed_keys=self._ordered_keys(self.state._pressed_keys),
            queued_taps=self._ordered_keys(self.state._activated_keys),
            applies_to_chunk=self._next_chunk(),
        )
        await self._send_state_update()
        return message

    @event(
        name="release_controls",
        description=(
            "Release every held W/A/S/D/I/J/K/L key and discard queued taps. Valid at any "
            "session boundary; the next available chunk receives a neutral action. Emits "
            "`controls_released` and broadcasts `state_update` on success."
        ),
    )
    async def release_controls(self) -> ControlsReleased:
        """Return all native action channels to neutral."""
        self._clear_controls()
        message = ControlsReleased(applies_to_chunk=self._next_chunk())
        await self._send_state_update()
        return message

    @event(
        name="set_prompt",
        description=(
            "Queue a non-empty scene prompt without clearing the rolling KV cache. Valid before "
            "or after an image is selected; the text rebuilds upstream cross-attention for the "
            "next generated chunk. Emits `prompt_queued` and broadcasts `state_update` on "
            "success, or `command_error` when the prompt is empty or the rollout limit requires "
            "a reset."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Scene description up to 4096 characters. Whitespace is trimmed; the accepted "
                "value applies to the next generated chunk and remains active until changed."
            ),
        ),
    ) -> PromptQueued:
        """Queue prompt conditioning for the next upstream chunk."""
        self._require_available_rollout()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "ABot-World requires a non-empty prompt."
            )
        self.state.prompt = normalized
        message = PromptQueued(prompt=normalized, applies_to_chunk=self._next_chunk())
        await self._send_state_update()
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded starting frame and begin a fresh continuous rollout. Valid at "
            "any time; the upload is checked before replacing "
            "the active world. Emits `image_selected` and broadcasts `state_update` on success, "
            "or `command_error` when the file is too large, mislabeled, or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - Reactor reads schema metadata.
            moderate=True,
            description=(
                "Starting frame uploaded through Reactor's file protocol. JPEG, PNG, WebP, or "
                "BMP up to 25 MiB and 100 million pixels; upstream center-crop preprocessing "
                "fits it to the native 1280x704 canvas."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene prompt for the fresh world. An empty value preserves the queued "
                "prompt, while a non-empty value replaces it after trimming."
            ),
        ),
    ) -> ImageSelected:
        """Validate and select one uploaded first frame."""
        validate_uploaded_image(image)
        normalized = prompt.strip() or self.state.prompt.strip() or DEFAULT_PROMPT
        self._selected_input = image
        self._image_source = "uploaded"
        self._image_name = image.name
        self.state.prompt = normalized
        self._queue_fresh_rollout()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=normalized,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    @event(
        name="random_image",
        description=(
            "Select a built-in ABot-World starting frame and its matching prompt, then begin a "
            "fresh continuous rollout. Valid when examples "
            "are configured. Emits `image_selected` and broadcasts `state_update` on success, "
            "or `command_error` with `image_unavailable` when no built-in image exists."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a different built-in upstream scene when possible."""
        config = self._require_config()
        candidates = [
            scene for scene in config.examples if scene.image != self._selected_input
        ]
        if not candidates:
            candidates = list(config.examples)
        if not candidates:
            raise CommandError(
                "image_unavailable", "No built-in ABot-World images are configured."
            )
        scene = secrets.choice(candidates)
        self._selected_input = scene.image
        self._image_source = "built_in"
        self._image_name = scene.image.name
        self.state.prompt = scene.prompt
        self._queue_fresh_rollout()
        message = ImageSelected(
            source="built_in",
            filename=scene.image.name,
            prompt=scene.prompt,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    @event(
        name="reset",
        description=(
            "Restart the selected starting image as a fresh causal world. Valid after an image "
            "is selected; the reset applies at the next inference boundary, preserves the "
            "prompt, and clears controls and the rollout limit. Emits "
            "`rollout_reset_queued` and broadcasts `state_update` on success, or `command_error` "
            "when no image is selected or the seed is out of range."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Random seed for the fresh world. Use -1 to retain the current seed or provide "
                "a non-negative 32-bit signed value to replace it."
            ),
        ),
    ) -> RolloutResetQueued:
        """Queue a fresh rollout from the active first frame."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an image before resetting ABot-World."
            )
        replaced_chunks = self._chunk_index
        if seed >= 0:
            self.state._seed = seed
        self._queue_fresh_rollout()
        message = RolloutResetQueued(
            seed=self.state._seed,
            replaced_chunks=replaced_chunks,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    async def process_input(self) -> ABotInput:
        """Snapshot native controls once an anchor and available rollout exist."""
        if self._selected_input is None:
            raise ApplicationError("Select an image before generating.")
        if self.state._limit_reached:
            raise ApplicationError("Reset the world after reaching the rollout limit.")
        action, _ = sample_key_snapshot(
            self.state._pressed_keys, self.state._activated_keys
        )
        self.state._activated_keys = frozenset()
        new_world = self.state._world_id != self.state._applied_world_id
        anchor = None
        if new_world:
            image = self._selected_input
            anchor = ABotAnchor(
                image=image.data if isinstance(image, UploadedFile) else image,
                suffix=upload_suffix(image) if isinstance(image, UploadedFile) else "",
                seed=self.state._seed,
            )
        self._reset_in_flight = new_world
        self._chunk_in_flight = True
        return ABotInput(self.state._world_id, anchor, self.state.prompt, action)

    def generate(self, input: ABotInput) -> ABotResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> ABotWorldOutput:
        """Publish the decoded chunk and acknowledged world state."""
        self._reset_in_flight = False
        self._chunk_in_flight = False
        if outcome.error is not None:
            # An unexpected native failure is not repaired by discarding the world.
            raise outcome.error
        result: ABotResult = outcome.result
        self.state._applied_world_id = result.world_id
        self._sampled_keys = result.sampled_keys
        self._active_prompt = result.prompt
        self._chunk_index = result.chunk_index
        config = self._require_config()
        if result.complete:
            self.state._limit_reached = True
            self._clear_controls()
            await self.send(
                RolloutLimitReached(
                    completed_chunks=result.chunk_index,
                    max_chunks=config.max_chunks,
                )
            )
        await self.send(self._state_update())
        return ABotWorldOutput(main_video=result.frames)

    def _queue_fresh_rollout(self) -> None:
        """Queue a cache reset and clear controls without reloading weights."""
        self.state._world_id += 1
        self.state._limit_reached = False
        self._clear_controls()
        self.output.flush()

    def _clear_controls(self) -> None:
        """Release held keys and discard every queued short tap."""
        self.state._pressed_keys = frozenset()
        self.state._activated_keys = frozenset()

    def _require_available_rollout(self) -> None:
        """Reject commands that need a fresh world after the chunk limit."""
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset ABot-World or select another image before generating more chunks.",
            )

    def _next_chunk(self) -> int | None:
        """Return the one-based chunk expected to sample newly accepted state."""
        if self._selected_input is None or self.state._limit_reached:
            return None
        if self.state._world_id != self.state._applied_world_id:
            return 1
        return self._chunk_index + 1 + int(self._chunk_in_flight)

    def _state_update(self) -> StateUpdate:
        """Build a complete client-facing snapshot of shared world state."""
        config = self._require_config()
        image_source = self._image_source
        if image_source not in {None, "uploaded", "built_in"}:
            raise RuntimeError(f"Unexpected ABot-World image source: {image_source}")
        return StateUpdate(
            image_source=image_source,
            image_name=self._image_name,
            prompt=self.state.prompt,
            active_prompt=self._active_prompt or None,
            seed=self.state._seed,
            reset_queued=(
                self._selected_input is not None
                and self.state._world_id != self.state._applied_world_id
            ),
            generating=self._reset_in_flight or self._chunk_in_flight,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            next_chunk=self._next_chunk(),
            max_chunks=config.max_chunks,
            pressed_keys=self._ordered_keys(self.state._pressed_keys),
            queued_taps=self._ordered_keys(self.state._activated_keys),
            sampled_keys=self._ordered_keys(self._sampled_keys),
        )

    async def _send_state_update(self, client: ClientInfo | None = None) -> None:
        """Send the current complete state to one viewer or broadcast it."""
        message = self._state_update()
        if client is not None:
            await client.send(message)
            return
        await self.send(message)

    def _ordered_keys(self, keys: frozenset[str]) -> list[str]:
        """Return native keys in the model's fixed action-channel order."""
        return [key for key in KEY_ORDER if key in keys]

    def _require_config(self) -> ABotWorldConfig:
        """Return loaded configuration or report an invalid lifecycle call."""
        if self._config is None:
            raise RuntimeError("ABot-World was not loaded")
        return self._config
