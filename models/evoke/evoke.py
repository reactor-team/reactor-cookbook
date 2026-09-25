"""Serve the public EVOKE post-distillation world model through Reactor Runtime."""

from __future__ import annotations

from pathlib import Path

from evoke_camera import CameraMotionPlanner, MotionConfig
from evoke_config import EvokeConfig, prepare_runtime, read_config
from evoke_images import (
    upload_suffix,
    validate_uploaded_image,
    validate_uploaded_pose,
    validate_uploaded_video,
)
from evoke_model import EvokeAnchor, EvokeInput, EvokeModel, EvokeResult
from evoke_types import (
    CommandApplied,
    EvokeOutput,
    EvokeState,
    RolloutRestarted,
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
from upstream_backend import WorkerSettings

logger = get_logger(__name__)

FPS = 24
FRAMES_PER_CHUNK = 36
CAMERA_POSES_PER_CHUNK = FRAMES_PER_CHUNK


class Evoke(ReactorApp):
    """Generate an autoregressive EVOKE world from image, video, or text conditioning."""

    state: EvokeState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: EvokeConfig | None = None
        self._engine = EvokeModel()
        self._planner: CameraMotionPlanner | None = None
        self._mode = "i2v"
        self._media: Path | UploadedFile | None = None
        self._pose: UploadedFile | None = None
        self._input_source = "none"
        self._input_name = ""
        self._pose_name = ""
        self._source_fps = 30
        self._source_height = 720
        self._source_width = 1280
        self._stability_prompt = ""
        self._seed = 42
        self._chunk_index = 0
        self._replaced_chunks = 0

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load EVOKE weights once in its Python 3.10 worker."""
        config = read_config(config_path, get_weights_path())
        prepare_runtime(config)
        self._config = config
        self._stability_prompt = config.stability_prompt
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            MotionConfig(
                fps=FPS,
                translation_units_per_second=config.translation_units_per_second,
                rotation_degrees_per_second=config.rotation_degrees_per_second,
            )
        )
        self._engine.load(
            WorkerSettings(
                python_executable=config.worker_python,
                source_path=config.source_path,
                base_model=config.base_model,
                transformer=config.transformer,
                vigeo_path=config.vigeo_path,
                default_image=config.default_image,
                stability_prompt=config.stability_prompt,
                seed=config.seed,
                max_chunks=config.max_chunks,
                reference_seconds=config.reference_seconds,
            )
        )
        logger.info(
            "EVOKE model ready",
            source_revision=config.source_revision,
            weight_revision=config.weights.revision,
            fps=FPS,
        )

    @session_started
    def on_session_started(self) -> None:
        """Wait for explicit image, video, or text conditioning."""
        config = self._require_config()
        self._mode = "i2v"
        self._media = None
        self._pose = None
        self._input_source = "none"
        self._input_name = ""
        self._pose_name = ""
        self.state.prompt = config.stability_prompt
        self.state._world_id = 0
        self.state._applied_world_id = None
        self._replaced_chunks = 0
        self._seed = config.seed
        self._chunk_index = 0
        self._clear_controls()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera controls when a viewer leaves the session."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        """Release rollout caches and session-owned uploads while retaining model weights."""
        try:
            self._engine.reset()
        finally:
            self._clear_controls()
            self.state._applied_world_id = None
            self._replaced_chunks = 0
            self._media = None
            self._pose = None
            self._chunk_index = 0

    @event(
        name="set_prompt",
        description=(
            "Set the text condition without resetting the active world. Empty text restores "
            "the scene-neutral exposure and temporal-stability prompt. The text is encoded at "
            "the next native chunk boundary while latent history, the persistent VAE cache, "
            "and geometric state remain intact. Emits `command_applied` and broadcasts "
            "`state_update` on success."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene description, up to 4096 characters. Empty text selects the "
                "configured scene-neutral stability prompt."
            ),
        ),
    ) -> CommandApplied:
        """Queue a prompt change and confirm its chunk boundary."""
        value = self._resolve_prompt(prompt)
        self.state.prompt = value
        detail = (
            "Neutral stability prompt restored"
            if not prompt.strip()
            else f"Prompt queued: {value}"
        )
        message = self._confirmation("set_prompt", detail)
        await self._send_state_update()
        return message

    @event(
        name="set_image",
        description=(
            "Start a fresh camera-controlled i2v world from an uploaded image. Valid at any "
            "time; it clears rollout progress and releases all camera axes. "
            "Emits `command_applied` and broadcasts `state_update` on success, or "
            "`command_error` for invalid image bytes."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description=(
                "Anchor image uploaded through Reactor. Accepts JPEG, PNG, WebP, or BMP up to "
                "25 MiB and 100 million decoded pixels."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene prompt for the fresh world. Empty text uses the documented "
                "scene-neutral exposure and temporal-stability prompt."
            ),
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> CommandApplied:
        """Select an uploaded i2v anchor and confirm the fresh rollout."""
        validate_uploaded_image(image)
        self._mode = "i2v"
        self._media = image
        self._pose = None
        self._input_source = "uploaded"
        self._input_name = image.name
        self._pose_name = ""
        self.state.prompt = self._resolve_prompt(prompt)
        if seed >= 0:
            self._seed = seed
        self._request_restart()
        message = self._confirmation("set_image", f"i2v anchor selected: {image.name}")
        await self._send_state_update()
        return message

    @event(
        name="set_reference_video",
        description=(
            "Start a fresh v2v world from a reference video plus its camera-pose NPZ. The "
            "configured five-second prefix is encoded with the uploaded poses, then six-axis "
            "controls continue from its final pose. Emits `command_applied` and broadcasts "
            "`state_update` on success, or `command_error` for invalid media, pose arrays, "
            "dimensions, or frame rate."
        ),
    )
    async def set_reference_video(
        self,
        video: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description="MP4, MOV, or WebM reference video up to 250 MiB.",
        ),
        pose: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description=(
                "NPZ camera track containing cam_c2w/extrinsic/data and "
                "intrinsics/intrinsic/K arrays aligned with the reference video."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional prompt; empty text uses the documented scene-neutral stability prompt."
            ),
        ),
        source_fps: int = InputField(
            default=30,
            ge=1,
            le=240,
            description="Frame rate shared by the uploaded reference video and pose track.",
        ),
        source_height: int = InputField(
            default=720,
            ge=64,
            le=8192,
            description="Pixel height at which the uploaded intrinsics were calibrated.",
        ),
        source_width: int = InputField(
            default=1280,
            ge=64,
            le=8192,
            description="Pixel width at which the uploaded intrinsics were calibrated.",
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> CommandApplied:
        """Select uploaded v2v conditioning and confirm the fresh rollout."""
        validate_uploaded_video(video)
        validate_uploaded_pose(pose)
        self._mode = "v2v"
        self._media = video
        self._pose = pose
        self._input_source = "uploaded"
        self._input_name = video.name
        self._pose_name = pose.name
        self._source_fps = source_fps
        self._source_height = source_height
        self._source_width = source_width
        self.state.prompt = self._resolve_prompt(prompt)
        if seed >= 0:
            self._seed = seed
        self._request_restart()
        message = self._confirmation(
            "set_reference_video",
            f"v2v reference selected: {video.name} with pose {pose.name}",
        )
        await self._send_state_update()
        return message

    @event(
        name="start_text",
        description=(
            "Start a fresh prompt-only t2v rollout. EVOKE produces autoregressive video but "
            "disables geometric warp and camera input in this mode, matching upstream. Emits "
            "`command_applied` and broadcasts `state_update` on success."
        ),
    )
    async def start_text(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional text condition used from chunk 1. Empty text uses the documented "
                "scene-neutral stability prompt."
            ),
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> CommandApplied:
        """Select prompt-only generation and confirm the fresh rollout."""
        value = self._resolve_prompt(prompt)
        self._mode = "t2v"
        self._media = None
        self._pose = None
        self._input_source = "none"
        self._input_name = ""
        self._pose_name = ""
        self.state.prompt = value
        if seed >= 0:
            self._seed = seed
        self._request_restart()
        message = self._confirmation("start_text", "Prompt-only t2v rollout selected")
        await self._send_state_update()
        return message

    async def _set_axis(self, name: str, value: float) -> CommandApplied:
        if self._mode == "t2v":
            raise CommandError(
                "camera_unavailable", "EVOKE t2v mode does not consume camera poses."
            )
        setattr(self.state, name, value)
        axes = (
            f"forward={self.state.forward:.2f}, strafe={self.state.strafe:.2f}, "
            f"vertical={self.state.vertical:.2f}, pitch={self.state.pitch:.2f}, "
            f"yaw={self.state.yaw:.2f}, roll={self.state.roll:.2f}"
        )
        message = self._confirmation(f"set_{name}", axes)
        await self._send_state_update()
        return message

    @event(
        name="set_forward",
        description=(
            "Set backward-to-forward camera translation for camera-controlled modes. The value "
            "is sampled at the next 36-pose chunk boundary and held. Emits `command_applied` "
            "and broadcasts `state_update` on success, or `command_error` in prompt-only t2v mode."
        ),
    )
    async def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized backward (-1) to forward (1) translation; 0 stops this axis.",
        ),
    ) -> CommandApplied:
        """Queue forward motion and confirm the complete camera state."""
        return await self._set_axis("forward", forward)

    @event(
        name="set_strafe",
        description=(
            "Set left-to-right camera translation at the next native chunk boundary. Emits "
            "`command_applied` and broadcasts `state_update` on success, or `command_error` "
            "in prompt-only t2v mode."
        ),
    )
    async def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized left (-1) to right (1) translation; 0 stops this axis.",
        ),
    ) -> CommandApplied:
        """Queue strafe motion and confirm the complete camera state."""
        return await self._set_axis("strafe", strafe)

    @event(
        name="set_vertical",
        description=(
            "Set down-to-up camera translation at the next native chunk boundary. Emits "
            "`command_applied` and broadcasts `state_update` on success, or `command_error` "
            "in prompt-only t2v mode."
        ),
    )
    async def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized down (-1) to up (1) translation; 0 stops this axis.",
        ),
    ) -> CommandApplied:
        """Queue vertical motion and confirm the complete camera state."""
        return await self._set_axis("vertical", vertical)

    @event(
        name="set_pitch",
        description=(
            "Set downward-to-upward pitch at the next native chunk boundary. Emits "
            "`command_applied` and broadcasts `state_update` on success, or `command_error` "
            "in prompt-only t2v mode."
        ),
    )
    async def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized downward (-1) to upward (1) pitch; 0 stops this axis.",
        ),
    ) -> CommandApplied:
        """Queue pitch motion and confirm the complete camera state."""
        return await self._set_axis("pitch", pitch)

    @event(
        name="set_yaw",
        description=(
            "Set left-to-right yaw at the next native chunk boundary. Emits `command_applied` "
            "and broadcasts `state_update` on success, or `command_error` in prompt-only t2v mode."
        ),
    )
    async def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized left (-1) to right (1) yaw; 0 stops this axis.",
        ),
    ) -> CommandApplied:
        """Queue yaw motion and confirm the complete camera state."""
        return await self._set_axis("yaw", yaw)

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise-to-clockwise roll at the next native chunk boundary. Emits "
            "`command_applied` and broadcasts `state_update` on success, or `command_error` "
            "in prompt-only t2v mode."
        ),
    )
    async def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized counterclockwise (-1) to clockwise (1) roll; 0 stops this axis.",
        ),
    ) -> CommandApplied:
        """Queue roll motion and confirm the complete camera state."""
        return await self._set_axis("roll", roll)

    @event(
        name="reset",
        description=(
            "Start a fresh rollout from the active image, video, or text conditioning. It "
            "clears generated progress and all camera axes while preserving the prompt. "
            "Emits `command_applied` and broadcasts `state_update` on success, or "
            "`command_error` if conditioning is missing."
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
    ) -> CommandApplied:
        """Queue a fresh rollout and confirm its retained seed."""
        if self._mode != "t2v" and self._media is None:
            raise CommandError(
                "conditioning_required", "Select EVOKE conditioning before reset."
            )
        if seed >= 0:
            self._seed = seed
        self._request_restart()
        message = self._confirmation(
            "reset", f"Fresh rollout queued with seed {self._seed}"
        )
        await self._send_state_update()
        return message

    async def process_input(self) -> EvokeInput:
        """Snapshot controls and apply the native rollout-length boundary."""
        if self._mode != "t2v" and self._media is None:
            raise ApplicationError(
                "Upload an image or select text/video conditioning first.",
            )
        planner = self._planner
        if planner is None:
            raise RuntimeError("EVOKE was not loaded")
        anchor = None
        if self.state._world_id != self.state._applied_world_id:
            anchor = EvokeAnchor(
                mode=self._mode,
                media=self._media.data
                if isinstance(self._media, UploadedFile)
                else self._media,
                media_suffix=upload_suffix(self._media)
                if isinstance(self._media, UploadedFile)
                else "",
                pose=self._pose.data if self._pose is not None else None,
                pose_suffix=upload_suffix(self._pose) if self._pose is not None else "",
                seed=self._seed,
                source_fps=self._source_fps,
                source_height=self._source_height,
                source_width=self._source_width,
            )
        trajectory = None
        if self._mode != "t2v":
            trajectory = planner.plan_chunk(
                strafe=0.0 if self._replaced_chunks else self.state.strafe,
                vertical=0.0 if self._replaced_chunks else self.state.vertical,
                forward=0.0 if self._replaced_chunks else self.state.forward,
                pitch=0.0 if self._replaced_chunks else self.state.pitch,
                yaw=0.0 if self._replaced_chunks else self.state.yaw,
                roll=0.0 if self._replaced_chunks else self.state.roll,
                frame_count=CAMERA_POSES_PER_CHUNK,
            )
        return EvokeInput(self.state._world_id, anchor, self.state.prompt, trajectory)

    def generate(self, input: EvokeInput) -> EvokeResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> EvokeOutput:
        """Publish the completed chunk and its shared state."""
        if outcome.error is not None:
            # A worker failure cannot be repaired by silently losing the current world.
            raise outcome.error
        result: EvokeResult = outcome.result
        if self._replaced_chunks:
            self.output.flush()
            self._clear_controls()
            await self.send(
                RolloutRestarted(
                    replaced_chunks=self._replaced_chunks,
                    max_chunks=self._require_config().max_chunks,
                    seed=result.seed,
                )
            )
            self._replaced_chunks = 0
        self.state._applied_world_id = result.world_id
        self._chunk_index = result.chunk_index
        if result.complete:
            self._replaced_chunks = result.chunk_index
            self.state._world_id += 1
            self._planner.reset()
        await self.send(self._state_update())
        return EvokeOutput(main_video=result.frames)

    def _request_restart(self) -> None:
        self.output.flush()
        self.state._world_id += 1
        self._replaced_chunks = 0
        if self._planner is not None:
            self._planner.reset()
        self._chunk_index = 0
        self._clear_controls()

    def _clear_controls(self) -> None:
        self.state.forward = 0.0
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _state_update(self) -> StateUpdate:
        config = self._require_config()
        return StateUpdate(
            mode=self._mode,
            prompt=self.state.prompt,
            input_source=self._input_source,
            input_name=self._input_name,
            pose_name=self._pose_name,
            seed=self._seed,
            completed_chunks=self._chunk_index,
            next_chunk=self._chunk_index + 1,
            max_chunks=config.max_chunks,
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
        )

    async def _send_state_update(self) -> None:
        """Broadcast the complete observable session state."""
        await self.send(self._state_update())

    def _confirmation(self, action: str, detail: str) -> CommandApplied:
        """Return a successful command result tied to its first affected chunk."""
        return CommandApplied(
            action=action,
            applies_to_chunk=self._chunk_index + 1,
            detail=detail,
        )

    def _resolve_prompt(self, prompt: str) -> str:
        """Return explicit text or the configured scene-neutral stability condition."""
        return prompt.strip() or self._stability_prompt

    def _require_config(self) -> EvokeConfig:
        if self._config is None:
            raise RuntimeError("EVOKE configuration is unavailable before load")
        return self._config
