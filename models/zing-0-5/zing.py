"""Serve one native Zing 0.5 autoregressive video block per Reactor turn."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from reactor_runtime import (
    ClientInfo, CommandError, InputField, ReactorPipeline, UploadedFile,
    connected, disconnected, event, session_ended, session_started,
)
from reactor_runtime.log import get_logger

from zing_assets import (
    ZingAdapterConfig, activate_source, configure_environment, prepare_assets, read_config,
)
from zing_images import materialized_image, validate_image
from zing_types import (
    ActionChanged, ChunkCompleted, ControlsReleased, ImageSelected, PromptQueued,
    RolloutReset, StateUpdate, ZingOutput, ZingState,
)

logger = get_logger(__name__)
_KEYS = ("w", "a", "s", "d", "i", "j", "k", "l")


class _Backend(Protocol):
    def reset(self, *, image: Path | None, prompt: str, seed: int) -> None: ...
    def generate_chunk(self, *, prompt: str, pressed_keys: set[str]) -> np.ndarray: ...
    def cache_frames(self) -> int: ...
    def end_session(self) -> None: ...


class Zing(ReactorPipeline):
    """Generate a controllable Zing 0.5 world from text or one initial image."""

    state: ZingState
    buffer_size = 16

    def __init__(self) -> None:
        super().__init__()
        self._config: ZingAdapterConfig | None = None
        self._backend: _Backend | None = None
        self._conditioning: Literal["none", "text", "uploaded", "built_in"] = "none"
        self._image: Path | UploadedFile | None = None
        self._image_name: str | None = None
        self._seed = 42
        self._active_prompt: str | None = None
        self._completed_chunks = 0
        self._generating = False

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path)
        configure_environment(config)
        prepare_assets(config)
        activate_source(config)
        from zing_backend import ZingBackend
        self._config = config
        self._seed = config.seed
        self._backend = ZingBackend(config)
        logger.info(
            "Zing 0.5 ready", source_revision=config.source_revision,
            checkpoint_revision=config.asset_revision, cache_window="97/9",
            frames_per_chunk=16,
        )

    @session_started
    def on_session_started(self) -> None:
        config = self._require_config()
        self.state.prompt = ""
        self.state._pressed_keys = frozenset()
        self.state._reset_requested = False
        self._conditioning = "none"
        self._image = None
        self._image_name = None
        self._seed = config.seed
        self._active_prompt = None
        self._completed_chunks = 0
        self._generating = False

    @session_ended
    def on_session_ended(self) -> None:
        self.state._reset_requested = False
        self.state._pressed_keys = frozenset()
        self._conditioning = "none"
        self._image = None
        self._image_name = None
        self._active_prompt = None
        self._completed_chunks = 0
        self._generating = False
        if self._backend is not None:
            self._backend.end_session()

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
            max_length=4096, moderate=True,
            description=(
                "Non-empty scene, appearance, subject, camera, and motion description, up to "
                "4096 characters. Whitespace is trimmed and the result is sampled when the next "
                "chunk starts."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and report the chunk expected to consume it."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("empty_prompt", "Zing requires a non-empty prompt.")
        initial = self._completed_chunks == 0 and self._active_prompt is None
        self.state.prompt = normalized
        starts_text_rollout = initial and self._image is None
        if starts_text_rollout:
            self._conditioning = "text"
            self._image_name = None
            self._request_reset()
        message = PromptQueued(
            prompt=normalized, applies_to_chunk=self._completed_chunks + 1,
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
            default="", max_length=4096, moderate=True,
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
        validate_image(image)
        config = self._require_config()
        if seed >= 0:
            self._seed = seed
        self.state.prompt = prompt.strip() or config.default_prompt
        self._conditioning = "uploaded"
        self._image = image
        self._image_name = image.name
        self._request_reset()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
            seed=self._seed,
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
        image = config.source_path / "assets" / "case0.jpg"
        self.state.prompt = config.example_prompt
        self._conditioning = "built_in"
        self._image = image
        self._image_name = image.name
        self._request_reset()
        await self.send(self._state_update())
        return ImageSelected(
            source="built_in",
            filename=image.name,
            prompt=self.state.prompt,
            seed=self._seed,
        )

    @event(
        name="set_key",
        description=(
            "Press or release one held world control for forthcoming chunks. `w`, `a`, `s`, and "
            "`d` control movement; `i`, `j`, `k`, and `l` control look direction. The complete "
            "held state is sampled when the next chunk begins and remains active until changed or "
            "released. Emits `action_changed` and broadcasts `state_update` on success."
        ),
    )
    async def set_key(
        self,
        key: Literal["w", "a", "s", "d", "i", "j", "k", "l"] = InputField(
            description=(
                "Control to change: `w`/`a`/`s`/`d` move forward/left/backward/right, and "
                "`i`/`j`/`k`/`l` look up/left/down/right."
            )
        ),
        pressed: bool = InputField(
            description="Whether to hold or release `key`; the change applies to the next chunk."
        ),
    ) -> ActionChanged:
        """Change one held control and report the complete held state."""
        keys = set(self.state._pressed_keys)
        (keys.add if pressed else keys.discard)(key)
        self.state._pressed_keys = frozenset(keys)
        await self.send(self._state_update())
        return ActionChanged(
            key=key,
            pressed=pressed,
            pressed_keys=sorted(keys),
            applies_to_chunk=self._completed_chunks + 1,
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
        return ControlsReleased(released_keys=released, applies_to_chunk=self._completed_chunks + 1)

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
        if seed >= 0:
            self._seed = seed
        replaced = self._completed_chunks
        self._request_reset()
        await self.send(self._state_update())
        return RolloutReset(seed=self._seed, replaced_chunks=replaced)

    async def inference(self) -> AsyncGenerator[ZingOutput | None, None]:
        backend = self._require_backend()
        config = self._require_config()
        while True:
            if self._conditioning == "none" and not self.state._reset_requested:
                yield None
                continue
            if self._completed_chunks >= config.max_chunks:
                self._request_reset()
            if self.state._reset_requested:
                self._generating = True
                await self.send(self._state_update())
                temporary = config.asset_path / "uploads"
                selected_image = (
                    materialized_image(self._image, temporary)
                    if self._image is not None
                    else _null_image()
                )
                with selected_image as image:
                    backend.reset(image=image, prompt=self.state.prompt, seed=self._seed)
                self.state._reset_requested = False
                self._completed_chunks = 0
                self._active_prompt = self.state.prompt
            self._generating = True
            started = time.monotonic()
            sampled_prompt = self.state.prompt
            sampled_keys = set(self.state._pressed_keys)
            frames = backend.generate_chunk(
                prompt=sampled_prompt,
                pressed_keys=sampled_keys,
            )
            elapsed = time.monotonic() - started
            self._completed_chunks += 1
            self._active_prompt = sampled_prompt
            self._generating = False
            await self.send(ChunkCompleted(
                chunk=self._completed_chunks, video_frames=int(frames.shape[0]),
                generation_seconds=elapsed, prompt=sampled_prompt,
                action_keys=sorted(sampled_keys), cache_frames=backend.cache_frames(),
            ))
            await self.send(self._state_update())
            yield ZingOutput(main_video=frames)

    def _request_reset(self) -> None:
        self.state._reset_requested = True
        self.state._pressed_keys = frozenset()
        self._active_prompt = None
        self._completed_chunks = 0
        if getattr(self, "output", None) is not None:
            self.output.flush()

    def _state_update(self) -> StateUpdate:
        return StateUpdate(
            prompt=self.state.prompt, active_prompt=self._active_prompt,
            pressed_keys=sorted(self.state._pressed_keys), conditioning=self._conditioning,
            image_name=self._image_name, seed=self._seed,
            completed_chunks=self._completed_chunks, reset_queued=self.state._reset_requested,
            generating=self._generating,
        )

    def _require_config(self) -> ZingAdapterConfig:
        if self._config is None:
            raise RuntimeError("Zing is not loaded")
        return self._config

    def _require_backend(self) -> _Backend:
        if self._backend is None:
            raise RuntimeError("Zing is not loaded")
        return self._backend


class _null_image:
    def __enter__(self) -> None:
        return None
    def __exit__(self, *_: object) -> None:
        return None
