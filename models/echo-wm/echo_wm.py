"""Serve one native Echo-WM Flash audio-video block per Reactor turn."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from echo_wm_assets import (
    EchoWMConfig,
    ExampleScene,
    activate_source,
    configure_cache_environment,
    load_examples,
    prepare_assets,
    read_config,
)
from echo_wm_camera import EchoCameraPlanner, MotionConfig
from echo_wm_images import validate_uploaded_image
from echo_wm_model import EchoAnchor, EchoInput, EchoModel, EchoResult
from echo_wm_schema import (
    AutomaticResetQueued,
    CameraMotionChanged,
    ChunkCompleted,
    EchoWMOutput,
    EchoWMState,
    ImageSelected,
    PromptQueued,
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


class EchoWM(ReactorApp):
    """Generate a prompt-, image-, and pure-camera-controlled audiovisual world."""

    state: EchoWMState
    # Keep the generated 48 kHz audio aligned with Echo-WM's native video time.
    fps = 24
    buffer_size = 24

    def __init__(self) -> None:
        super().__init__()
        self._config: EchoWMConfig | None = None
        self._engine = EchoModel()
        self._planner: EchoCameraPlanner | None = None
        self._examples: tuple[ExampleScene, ...] = ()
        self._selected_image: Path | UploadedFile | None = None
        self._image_source: Literal["uploaded", "built_in"] | None = None
        self._image_name: str | None = None
        self._active_prompt: str | None = None
        self._seed = 0
        self._chunk_index = 0
        self._generating = False

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load Echo-WM weights once at startup.

        Args:
            config_path: Path to ``echo_wm.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path, get_weights_path())
        configure_cache_environment(config)
        prepare_assets(config)
        activate_source(config)
        self._config = config
        self._examples = load_examples(config)
        self._seed = config.seed
        self._planner = EchoCameraPlanner(
            MotionConfig(
                fps=config.fps,
                translation_speed=config.translation_speed,
                rotation_speed_degrees=config.rotation_speed_degrees,
                pitch_speed_degrees=config.pitch_speed_degrees,
                pitch_limit_degrees=config.pitch_limit_degrees,
            )
        )
        self._engine.load(config)

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty continuous world before the first viewer connects."""
        config = self._require_loaded()
        self.state.prompt = ""
        self.state._applied_world_id = None
        self.state._fov_degrees = config.fov_degrees
        self._clear_camera()
        self._selected_image = None
        self._image_source = None
        self._image_name = None
        self._active_prompt = None
        self._seed = config.seed
        self._chunk_index = 0
        self._generating = False

    @session_ended
    def on_session_ended(self) -> None:
        """Release causal state when the shared session ends."""
        self._clear_camera()
        self.state._applied_world_id = None
        self._selected_image = None
        self._image_source = None
        self._image_name = None
        self._active_prompt = None
        self._chunk_index = 0
        self._generating = False
        self._engine.reset()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera motion when a viewer disconnects."""
        self._clear_camera()
        await self.send(self._state_update())

    @event(
        name="set_image",
        description=(
            "Select an uploaded first frame and start a fresh audiovisual world with continuous "
            "generation. Supply a prompt or leave it blank to use the configured image-neutral "
            "default. Emits "
            "`image_selected` and broadcasts `state_update` on success, or `command_error` "
            "when the upload is invalid."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "JPEG, PNG, WebP, or BMP first frame, up to 25 MiB and 100 million pixels."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Scene, motion, sound, music, and speech description. An empty value uses the "
                "configured image-neutral default for the new scene."
            ),
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> ImageSelected:
        """Validate an upload and select it as the next world anchor."""
        validate_uploaded_image(image)
        config = self._require_loaded()
        normalized = prompt.strip() or config.default_upload_prompt
        if seed >= 0:
            self._seed = seed
        self._selected_image = image
        self._image_source = "uploaded"
        self._image_name = image.name
        self.state.prompt = normalized
        self._request_reset()
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
            "Select a different public Echo-WM example with its matching prompt, field of "
            "view, and seed, then start continuous generation from it. Emits `image_selected` "
            "and broadcasts `state_update` on success, "
            "or `command_error` when no example is configured."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a different upstream example when possible."""
        if not self._examples:
            raise CommandError(
                "image_unavailable", "No Echo-WM examples are configured."
            )
        candidates = [
            scene for scene in self._examples if scene.image != self._selected_image
        ]
        scene = secrets.choice(candidates or list(self._examples))
        self._selected_image = scene.image
        self._image_source = "built_in"
        self._image_name = scene.image.name
        self.state.prompt = scene.prompt
        self.state._fov_degrees = scene.fov_degrees
        self._seed = scene.seed
        self._request_reset()
        message = ImageSelected(
            source="built_in",
            filename=scene.image.name,
            prompt=scene.prompt,
            seed=scene.seed,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_prompt",
        description=(
            "Replace the text condition and queue a fresh rollout from the selected image. "
            "The new prompt begins at chunk one and continuous generation resumes from the "
            "fresh world. Emits `prompt_queued` and broadcasts "
            "`state_update` on success, or `command_error` before image selection or for empty "
            "text."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene, character, perspective, sound, music, and speech description."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt-conditioned fresh rollout from the selected image."""
        self._require_selected()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "Echo-WM requires a non-empty prompt."
            )
        self.state.prompt = normalized
        self._request_reset()
        message = PromptQueued(prompt=normalized, applies_to_chunk=1)
        await self.send(self._state_update())
        return message

    @event(
        name="set_camera_motion",
        description=(
            "Set Echo-WM's four native held camera axes atomically. Values are sampled at the "
            "next chunk boundary and remain active until changed or released. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` before image selection."
        ),
    )
    async def set_camera_motion(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Backward (-1) to forward (1); zero is neutral.",
        ),
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Left (-1) to right (1); zero is neutral.",
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
        """Set every native camera axis and report the complete held state."""
        self._require_selected()
        self.state._forward = forward
        self.state._strafe = strafe
        self.state._pitch = pitch
        self.state._yaw = yaw
        message = self._camera_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="release_camera",
        description=(
            "Return every held camera axis to neutral for forthcoming chunks. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` before image selection."
        ),
    )
    async def release_camera(self) -> CameraMotionChanged:
        """Release native camera motion and report neutral state."""
        self._require_selected()
        self._clear_camera()
        message = self._camera_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="set_fov",
        description=(
            "Set the horizontal camera field of view for forthcoming chunks. The value is "
            "sampled at the next chunk boundary and remains active. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` before image selection."
        ),
    )
    async def set_fov(
        self,
        fov_degrees: float = InputField(
            default=70.0,
            ge=30.0,
            le=120.0,
            description="Horizontal field of view in degrees from 30 through 120.",
        ),
    ) -> CameraMotionChanged:
        """Set and report the held camera field of view."""
        self._require_selected()
        self.state._fov_degrees = fov_degrees
        message = self._camera_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="reset",
        description=(
            "Queue a fresh bounded rollout from the selected image and current prompt. The "
            "command releases camera motion and continues generation from chunk one. Emits "
            "`rollout_reset_queued` and broadcasts `state_update` on success, or "
            "`command_error` before image selection."
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
    ) -> RolloutResetQueued:
        """Queue a fresh rollout and report the state it replaces."""
        self._require_selected()
        if seed >= 0:
            self._seed = seed
        replaced = self._chunk_index
        self._request_reset()
        message = RolloutResetQueued(seed=self._seed, replaced_chunks=replaced)
        await self.send(self._state_update())
        return message

    async def process_input(self) -> EchoInput:
        """Plan camera poses and carry an anchor until the model acknowledges its world."""
        if self._selected_image is None:
            raise ApplicationError("Select an image before generating.")
        new_world = self.state._world_id != self.state._applied_world_id
        prompt = self.state.prompt.strip() if new_world else self._active_prompt
        if not prompt:
            raise ApplicationError("Echo-WM requires a prompt before generating.")
        config = self._require_loaded()
        planner = self._planner
        if planner is None:
            raise RuntimeError("Echo-WM was not loaded")
        anchor = None
        if new_world:
            selected = self._selected_image
            image = selected.data if isinstance(selected, UploadedFile) else selected
            anchor = EchoAnchor(
                image=image,
                suffix=Path(selected.name).suffix.lower() or ".png",
                seed=self._seed,
            )
        camera = planner.plan_chunk(
            forward=self.state._forward,
            strafe=self.state._strafe,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
            frame_count=config.frames_per_chunk,
        )
        self._generating = True
        return EchoInput(
            world_id=self.state._world_id,
            anchor=anchor,
            prompt=prompt,
            poses=camera.latent_poses,
            fov_degrees=self.state._fov_degrees,
        )

    def generate(self, input: EchoInput) -> EchoResult:
        """Forward one native audiovisual step to the independent model."""
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> EchoWMOutput:
        """Publish native media, generation timings, and rollout state."""
        self._generating = False
        if outcome.error is not None:
            # Missing anchors and exhausted worlds are prevented by the application.
            # Unexpected model failures must terminate the session, not hide a reset.
            raise outcome.error
        result: EchoResult = outcome.result
        self.state._applied_world_id = result.world_id
        self._active_prompt = result.prompt
        self._chunk_index = result.chunk_index
        profile = result.profile
        logger.info(
            "Echo-WM chunk complete",
            chunk=self._chunk_index,
            generation_seconds=round(outcome.elapsed, 3),
            **{name: round(value, 4) for name, value in profile.items()},
        )
        await self.send(
            ChunkCompleted(
                chunk=result.chunk_index,
                video_frames=int(result.video.shape[0]),
                audio_samples=int(result.audio.shape[-1]),
                generation_seconds=round(outcome.elapsed, 3),
                denoise_seconds=_profile_value(profile, "denoise_seconds"),
                cache_commit_seconds=_profile_value(profile, "cache_commit_seconds"),
                video_decode_seconds=_profile_value(profile, "video_decode_seconds"),
                audio_decode_seconds=_profile_value(profile, "audio_decode_seconds"),
                cuda_total_seconds=_profile_value(profile, "cuda_total_seconds"),
                prompt=result.prompt,
                forward=self.state._forward,
                strafe=self.state._strafe,
                pitch=self.state._pitch,
                yaw=self.state._yaw,
            )
        )
        if result.complete:
            # Preserve automatic continuation and held controls without cutting this output.
            self.state._world_id += 1
            if self._planner is not None:
                self._planner.reset()
            await self.send(
                AutomaticResetQueued(
                    completed_chunks=result.chunk_index,
                    max_chunks=self._require_loaded().max_chunks,
                    seed=self._seed,
                )
            )
        await self.send(self._state_update())
        return EchoWMOutput(main_video=result.video, main_audio=result.audio)

    def _request_reset(self) -> None:
        """Queue a clean rollout and clear prior playout, controls, and progress."""
        self.output.flush()
        self._clear_camera()
        self.state._world_id += 1
        if self._planner is not None:
            self._planner.reset()
        self._chunk_index = 0

    def _clear_camera(self) -> None:
        """Return every native camera axis to neutral."""
        self.state._forward = 0.0
        self.state._strafe = 0.0
        self.state._pitch = 0.0
        self.state._yaw = 0.0

    def _camera_changed(self) -> CameraMotionChanged:
        """Return the complete held camera state after a mutation."""
        return CameraMotionChanged(
            forward=self.state._forward,
            strafe=self.state._strafe,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
            fov_degrees=self.state._fov_degrees,
            applies_to_chunk=self._next_chunk(),
        )

    def _require_loaded(self) -> EchoWMConfig:
        """Return startup configuration or fail before model load completes."""
        if self._config is None:
            raise RuntimeError("Echo-WM was not loaded")
        return self._config

    def _require_selected(self) -> None:
        """Reject commands that need a first-frame image before one is selected."""
        if self._selected_image is None:
            raise CommandError(
                "image_required",
                "Upload an image or select a random image before this command.",
            )

    def _next_chunk(self) -> int | None:
        """Return the one-based chunk expected to consume newly accepted input."""
        if self._selected_image is None:
            return None
        if self.state._world_id != self.state._applied_world_id:
            return 1
        return self._chunk_index + 1 + int(self._generating)

    def _state_update(self) -> StateUpdate:
        """Return a complete client-visible snapshot of the shared world."""
        config = self._config
        return StateUpdate(
            image_source=self._image_source,
            image_name=self._image_name,
            prompt=self.state.prompt or None,
            active_prompt=self._active_prompt,
            seed=self._seed,
            reset_queued=(
                self._selected_image is not None
                and self.state._world_id != self.state._applied_world_id
            ),
            generating=self._generating,
            completed_chunks=self._chunk_index,
            next_chunk=self._next_chunk(),
            max_chunks=config.max_chunks if config is not None else 0,
            forward=self.state._forward,
            strafe=self.state._strafe,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
            fov_degrees=self.state._fov_degrees,
        )


def _profile_value(profile: dict[str, float], name: str) -> float | None:
    """Return a rounded CUDA timing when profiling is enabled."""
    value = profile.get(name)
    return round(value, 4) if value is not None else None
