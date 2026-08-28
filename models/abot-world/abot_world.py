"""Serve ABot-World's native causal rollout through Reactor Runtime.

The adapter imports the pinned upstream causal pipeline directly. Each Reactor
turn samples W/A/S/D movement and I/J/K/L view controls once, generates one
three-latent block with the upstream rolling KV cache, and emits the decoded RGB
chunk. Prompt changes rebuild only the upstream cross-attention condition; image
changes and resets initialize a fresh autoregressive world.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorPipeline,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

from abot_world_assets import (
    build_upstream_config,
    load_upstream_modules,
    prepare_assets,
    read_config,
)
from abot_world_controls import (
    KEY_CHOICES,
    KEY_ORDER,
    sample_key_snapshot,
    update_key_state,
)
from abot_world_images import materialized_image, validate_uploaded_image
from abot_world_types import (
    DEFAULT_PROMPT,
    ABotWorldConfig,
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

logger = get_logger(__name__)

FPS = 12
FRAMES_PER_CHUNK = 12
_EXPECTED_LATENTS_PER_CHUNK = 3
_EXPECTED_LOCAL_CACHE_LATENTS = 21


class _FrameCapture:
    """Collect frames from the upstream decoder's writer interface."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def append_data(self, frame: np.ndarray) -> None:
        """Append one decoded RGB frame."""
        self.frames.append(np.asarray(frame))


class ABotWorld(ReactorPipeline):
    """Generate a prompt-, image-, and keyboard-controlled ABot world."""

    state: ABotWorldState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: ABotWorldConfig | None = None
        self._modules: dict[str, Any] = {}
        self._pipeline: Any = None
        self._device: Any = None
        self._latent_shape: tuple[int, int, int, int, int] | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._image_source: str | None = None
        self._image_name: str | None = None
        self._active_prompt = ""
        self._sampled_keys: frozenset[str] = frozenset()
        self._chunk_index = 0
        self._reset_in_flight = False
        self._chunk_in_flight = False

    def load(self, config_path: Path | None) -> None:
        """Load the pinned public source and distilled checkpoint once.

        Args:
            config_path: Path to ``abot_world.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        prepare_assets(config)
        modules = load_upstream_modules(config)
        torch = modules["torch"]
        if not torch.cuda.is_available():
            raise RuntimeError("ABot-World requires a CUDA accelerator")
        device = torch.device("cuda")
        upstream_config = build_upstream_config(config, modules)
        num_frame_per_block = int(upstream_config.num_frame_per_block)
        local_attn_size = int(upstream_config.model_kwargs.local_attn_size)
        if num_frame_per_block != _EXPECTED_LATENTS_PER_CHUNK:
            raise ValueError(
                "ABot-World must retain its native three-latent autoregressive chunk size"
            )
        if local_attn_size != _EXPECTED_LOCAL_CACHE_LATENTS:
            raise ValueError(
                "ABot-World must retain its native 21-latent KV cache window"
            )

        modules["set_seed"](config.seed)
        torch.set_grad_enabled(False)
        vae = modules["create_vae"](upstream_config)
        pipeline = modules["pipeline_type"](upstream_config, device=device, vae=vae)
        try:
            modules["replace_norms"](pipeline.generator.model)
            modules["replace_rope"]()
            logger.info("ABot-World upstream Helios kernels enabled")
        except Exception as error:  # noqa: BLE001 - optional upstream kernels may be unavailable.
            logger.warning("ABot-World Helios kernels unavailable", error=str(error))
        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        if pipeline.encoder is not None:
            pipeline.encoder.to(device=device)
        pipeline.torch_dtype = torch.bfloat16

        vae_for_shape = (
            pipeline.encoder if pipeline.encoder is not None else pipeline.vae
        )
        upsampling = int(getattr(vae_for_shape, "upsampling_factor", 16))
        latent_channels = int(vae_for_shape.z_dim)
        self._latent_shape = (
            1,
            num_frame_per_block,
            latent_channels,
            config.height // upsampling,
            config.width // upsampling,
        )
        self._config = config
        self._modules = modules
        self._pipeline = pipeline
        self._device = device
        logger.info(
            "ABot-World model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.checkpoint.revision,
            latent_frames_per_chunk=num_frame_per_block,
            local_cache_latents=local_attn_size,
            output_fps=FPS,
            max_chunks=config.max_chunks,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize one shared world before its first viewer connects."""
        config = self._require_config()
        self.state.prompt = DEFAULT_PROMPT
        self.state._seed = config.seed
        self.state._reset_requested = False
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
        self._reset_upstream_stream()

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

    async def inference(self) -> AsyncGenerator[ABotWorldOutput | None, None]:
        """Generate one native causal chunk per turn and emit its RGB frame batch."""
        while True:
            selected_input = self._selected_input
            if selected_input is None or self.state._limit_reached:
                yield None
                continue

            if self.state._reset_requested:
                self.state._reset_requested = False
                self._reset_in_flight = True
                self.output.flush()
                try:
                    self._reset_rollout(
                        selected_input,
                        self.state.prompt,
                        self.state._seed,
                    )
                finally:
                    self._reset_in_flight = False
                await self._send_state_update()

            action, sampled = sample_key_snapshot(
                self.state._pressed_keys,
                self.state._activated_keys,
            )
            self.state._activated_keys = frozenset()
            prompt = self.state.prompt
            self._chunk_in_flight = True
            try:
                frames = self._generate_chunk(prompt, action)
            finally:
                self._chunk_in_flight = False
            self._sampled_keys = sampled
            self._active_prompt = prompt
            self._chunk_index += 1

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
            await self._send_state_update()
            yield ABotWorldOutput(main_video=frames)

    def _reset_rollout(
        self,
        selected_input: Path | UploadedFile,
        prompt: str,
        seed: int,
    ) -> None:
        """Initialize upstream conditions and rolling caches for a fresh world."""
        pipeline = self._require_pipeline()
        config = self._require_config()
        torch = self._modules["torch"]
        self._modules["set_seed"](seed)
        pipeline.set_prompts([prompt], device=self._device)
        empty_ref_dir = config.checkpoint.path / "unused-reference-slots"
        pipeline.set_ref_latent_mask_from_exists_paths(
            ref_dir=str(empty_ref_dir),
            device=self._device,
        )
        pipeline.reset_stream(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self._device,
            initial_latent=None,
        )
        with materialized_image(selected_input) as image_path:
            pipeline.set_first_frame_latent(
                str(image_path),
                height=config.height,
                width=config.width,
                device=self._device,
            )
        self._active_prompt = prompt
        self._sampled_keys = frozenset()
        self._chunk_index = 0
        self.state._limit_reached = False

    def _generate_chunk(self, prompt: str, action: dict[str, bool]) -> np.ndarray:
        """Run one upstream autoregressive block and cached VAE decode."""
        pipeline = self._require_pipeline()
        config = self._require_config()
        latent_shape = self._latent_shape
        if latent_shape is None:
            raise RuntimeError("ABot-World latent shape was not initialized")
        torch = self._modules["torch"]
        if prompt != self._active_prompt:
            pipeline.set_prompts([prompt], device=self._device)
        pipeline.set_act(
            action,
            height=config.height,
            width=config.width,
            num_frames=latent_shape[1],
            device=self._device,
        )
        noise = torch.randn(latent_shape, device=self._device, dtype=torch.bfloat16)
        latent_block = pipeline.generate_next_block(noise)
        capture = _FrameCapture()
        pipeline.decode_block_and_write(latent_block, capture)
        return self._normalize_frames(capture.frames)

    def _normalize_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return a contiguous native-resolution uint8 RGB frame batch."""
        config = self._require_config()
        if not frames:
            raise RuntimeError("ABot-World decoded an empty chunk")
        normalized: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.shape != (config.height, config.width, 3):
                raise RuntimeError(
                    f"ABot-World frame {index} has shape {array.shape}; expected "
                    f"{(config.height, config.width, 3)}"
                )
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            normalized.append(np.ascontiguousarray(array))
        return np.ascontiguousarray(np.stack(normalized))

    def _queue_fresh_rollout(self) -> None:
        """Queue a cache reset and clear controls without reloading weights."""
        self.state._reset_requested = True
        self.state._limit_reached = False
        self._clear_controls()
        self.output.flush()

    def _reset_upstream_stream(self) -> None:
        """Reset reusable upstream caches and decoder state after a session."""
        pipeline = self._pipeline
        if pipeline is None or self._device is None:
            return
        if pipeline.kv_cache1 is None:
            return
        torch = self._modules["torch"]
        pipeline.reset_stream(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self._device,
            initial_latent=None,
        )
        vae_model = getattr(pipeline.vae, "model", None)
        if vae_model is not None and hasattr(vae_model, "clear_cache"):
            vae_model.clear_cache()
        taehv = getattr(pipeline.vae, "taehv", None)
        if taehv is not None and hasattr(taehv, "reset"):
            taehv.reset()

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
        if self.state._reset_requested:
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
            reset_queued=self.state._reset_requested,
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

    def _require_pipeline(self) -> Any:
        """Return the loaded upstream pipeline or report an invalid lifecycle call."""
        if self._pipeline is None:
            raise RuntimeError("ABot-World was not loaded")
        return self._pipeline
