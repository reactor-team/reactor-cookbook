"""Serve SolarWM through Reactor's interactive pipeline API without shadowing upstream."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from pathlib import Path

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
from solarwm_backend import BackendSettings, SolarWMBackend
from solarwm_camera import CameraMotionPlanner, MotionConfig
from solarwm_config import SolarWMConfig, prepare_runtime, read_config
from solarwm_images import normalize_output_frames, validate_uploaded_image
from solarwm_types import (
    CameraMotionChanged,
    ImageSelected,
    PromptQueued,
    RolloutLimitReached,
    RolloutResetQueued,
    SolarWMOutput,
    SolarWMState,
    StateUpdate,
)


class SolarWM(ReactorPipeline):
    """Generate an image-, prompt-, and camera-controllable SolarWM world."""

    state: SolarWMState
    buffer_size = 12

    def __init__(self) -> None:
        super().__init__()
        self._config: SolarWMConfig | None = None
        self._backend: SolarWMBackend | None = None
        self._planner: CameraMotionPlanner | None = None
        self._selected_image: UploadedFile | None = None
        self._seed = 42
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._last_chunk_seconds: float | None = None

    def load(self, config_path: Path | None) -> None:
        """Prepare pinned upstream assets and load SolarWM once."""
        config = read_config(config_path)
        prepare_runtime(config)
        self._config = config
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            MotionConfig(
                config.translation_units_per_latent, config.rotation_degrees_per_latent
            )
        )
        self._backend = SolarWMBackend(
            BackendSettings(
                source_path=config.source_path,
                upstream_config=config.upstream_config,
                base_path=config.base_path,
                checkpoint_path=config.checkpoint_path,
                runtime_root=config.runtime_root,
            )
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty world that waits for an uploaded anchor image."""
        self.state.prompt = ""
        self._selected_image = None
        self._seed = self._require_config().seed
        self._clear_controls()
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._chunk_in_flight = False
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
        """Release causal caches while retaining loaded model weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._selected_image = None
        self._clear_controls()
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._last_chunk_seconds = None

    @event(
        name="set_image",
        description=(
            "Replace the anchor image and start a fresh world with continuous `main_video` "
            "generation. Valid at any time; progress and camera axes reset, and the upload is "
            "decoded before the command succeeds. Emits `image_selected` and broadcasts "
            "`state_update` on success, or `command_error` for invalid image bytes, media "
            "type, or dimensions."
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
                "fresh world; an empty value preserves the active prompt or uses the configured "
                "default when no prompt is active."
            ),
        ),
    ) -> ImageSelected:
        """Select an uploaded anchor and queue a fresh continuous rollout."""
        validate_uploaded_image(image)
        normalized = (
            prompt.strip()
            or self.state.prompt.strip()
            or self._require_config().default_prompt
        )
        self._selected_image = image
        self.state.prompt = normalized
        self._request_restart()
        result = ImageSelected(
            source="uploaded", filename=image.name, prompt=normalized, seed=self._seed
        )
        await self.send(self._state_update())
        return result

    @event(
        name="set_prompt",
        description=(
            "Replace the text condition and queue a fresh rollout from the selected image. The "
            "new prompt begins at chunk one and continuous generation resumes from the fresh "
            "world. Emits `prompt_queued` and broadcasts `state_update` on success, or "
            "`command_error` before image selection or for empty text."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene description up to 4096 characters for the fresh world."
            ),
        ),
    ) -> PromptQueued:
        """Queue text conditioning and restart from the uploaded anchor."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "SolarWM requires a prompt.")
        if self._selected_image is None:
            raise CommandError(
                "image_required", "Upload an anchor image before setting a prompt."
            )
        self.state.prompt = normalized
        self._request_restart()
        result = PromptQueued(prompt=normalized, applies_to_chunk=1)
        await self.send(self._state_update())
        return result

    @event(
        name="set_forward",
        description=(
            "Set backward-to-forward camera translation for the next chunk. Valid after image "
            "selection and before the rollout limit; the value is held for later chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
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
        """Set forward translation and report all held camera axes."""
        return await self._set_axis("forward", forward)

    @event(
        name="set_strafe",
        description=(
            "Set left-to-right camera translation for the next chunk. Valid after image "
            "selection and before the rollout limit; the value is held for later chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
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
        """Set strafe translation and report all held camera axes."""
        return await self._set_axis("strafe", strafe)

    @event(
        name="set_vertical",
        description=(
            "Set down-to-up camera translation for the next chunk. Valid after image selection "
            "and before the rollout limit; the value is held for later chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
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
        """Set vertical translation and report all held camera axes."""
        return await self._set_axis("vertical", vertical)

    @event(
        name="set_pitch",
        description=(
            "Set downward-to-upward camera pitch for the next chunk. Valid after image selection "
            "and before the rollout limit; the value is held for later chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
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
        """Set pitch and report all held camera axes."""
        return await self._set_axis("pitch", pitch)

    @event(
        name="set_yaw",
        description=(
            "Set left-to-right camera yaw for the next chunk. Valid after image selection and "
            "before the rollout limit; the value is held for later chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
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
        """Set yaw and report all held camera axes."""
        return await self._set_axis("yaw", yaw)

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise-to-clockwise camera roll for the next chunk. Valid after image "
            "selection and before the rollout limit; the value is held for later chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
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
        """Set roll and report all held camera axes."""
        return await self._set_axis("roll", roll)

    @event(
        name="release_camera",
        description=(
            "Return every held camera axis to neutral for forthcoming chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected or a fresh rollout is required."
        ),
    )
    async def release_camera(self) -> CameraMotionChanged:
        """Release all camera motion and report the neutral controls."""
        self._require_available_rollout()
        self._clear_controls()
        result = self._camera_changed()
        await self.send(self._state_update())
        return result

    @event(
        name="reset",
        description=(
            "Restart from the selected image and prompt with continuous generation from chunk "
            "one. Valid when an anchor exists; progress and camera axes reset. Emits "
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
        """Queue a reproducible fresh rollout from the selected anchor."""
        if self._selected_image is None:
            raise CommandError(
                "image_required", "Upload an anchor image before resetting."
            )
        if seed >= 0:
            self._seed = seed
        replaced = self._chunk_index
        self._request_restart()
        result = RolloutResetQueued(seed=self._seed, replaced_chunks=replaced)
        await self.send(self._state_update())
        return result

    async def inference(self) -> AsyncGenerator[SolarWMOutput | None, None]:
        """Generate synchronously and emit exactly one native causal chunk per iteration."""
        backend, planner = self._backend, self._planner
        if backend is None or planner is None:
            raise RuntimeError("SolarWM was not loaded")
        while True:
            if self.state._restart_requested:
                if self._selected_image is None:
                    yield None
                    continue
                self.state._restart_requested = False
                backend.reset(self._seed, self._selected_image, self.state.prompt)
                planner.reset()
                self._chunk_index = 0
            if self.state._limit_reached:
                yield None
                continue
            poses = planner.plan_chunk(
                strafe=self.state.strafe,
                vertical=self.state.vertical,
                forward=self.state.forward,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                roll=self.state.roll,
            )
            self._chunk_in_flight = True
            started = time.perf_counter()
            try:
                frames = backend.generate_chunk(poses)
            finally:
                self._chunk_in_flight = False
            self._last_chunk_seconds = time.perf_counter() - started
            self._chunk_index += 1
            config = self._require_config()
            if self._chunk_index >= config.max_chunks:
                self.state._limit_reached = True
                self._clear_controls()
                await self.send(
                    RolloutLimitReached(
                        completed_chunks=self._chunk_index, max_chunks=config.max_chunks
                    )
                )
            await self.send(self._state_update())
            yield SolarWMOutput(main_video=normalize_output_frames(frames))

    async def _set_axis(self, name: str, value: float) -> CameraMotionChanged:
        self._require_available_rollout()
        setattr(self.state, name, value)
        result = self._camera_changed()
        await self.send(self._state_update())
        return result

    def _request_restart(self) -> None:
        self._clear_controls()
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._last_chunk_seconds = None
        self.output.flush()

    def _require_available_rollout(self) -> None:
        if self._selected_image is None:
            raise CommandError(
                "image_required", "Upload an anchor image before this command."
            )
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset SolarWM before requesting another chunk.",
            )

    def _clear_controls(self) -> None:
        for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll"):
            setattr(self.state, name, 0.0)

    def _camera_changed(self) -> CameraMotionChanged:
        return CameraMotionChanged(
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
            applies_to_chunk=self._next_control_chunk(),
        )

    def _next_control_chunk(self) -> int:
        return (
            1
            if self.state._restart_requested
            else self._chunk_index + 1 + int(self._chunk_in_flight)
        )

    def _require_config(self) -> SolarWMConfig:
        if self._config is None:
            raise RuntimeError("SolarWM was not loaded")
        return self._config

    def _state_update(self) -> StateUpdate:
        available = self._selected_image is not None and not self.state._limit_reached
        next_chunk = self._next_control_chunk() if available else None
        return StateUpdate(
            prompt=self.state.prompt,
            image_source="uploaded" if self._selected_image is not None else "none",
            image_name=self._selected_image.name
            if self._selected_image is not None
            else "",
            seed=self._seed,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            last_chunk_seconds=self._last_chunk_seconds,
            next_chunk=next_chunk,
            next_chunk_frames=(9 if next_chunk == 1 else 12) if next_chunk else None,
            max_chunks=self._config.max_chunks if self._config else 0,
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
        )
