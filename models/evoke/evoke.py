"""Serve the public EVOKE post-distillation world model through Reactor Runtime."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Protocol

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

from evoke_camera import CameraMotionPlanner, MotionConfig
from evoke_config import EvokeConfig, prepare_runtime, read_config
from evoke_images import (
    normalize_output_frames,
    validate_uploaded_image,
    validate_uploaded_pose,
    validate_uploaded_video,
)
from evoke_types import (
    CommandApplied,
    EvokeOutput,
    EvokeState,
    RolloutRestarted,
    StateUpdate,
)
from upstream_backend import EvokeWorkerBackend, WorkerSettings

logger = get_logger(__name__)

FPS = 24
FRAMES_PER_CHUNK = 36
CAMERA_POSES_PER_CHUNK = FRAMES_PER_CHUNK


class _Backend(Protocol):
    def reset(
        self,
        *,
        mode: str,
        media: Path | UploadedFile | None,
        pose: UploadedFile | None,
        prompt: str,
        seed: int,
        source_fps: int = 30,
        source_height: int = 720,
        source_width: int = 1280,
    ) -> None: ...

    def generate_chunk(
        self,
        trajectory_c2w: np.ndarray | None,
        *,
        seed: int,
        prompt: str,
    ) -> np.ndarray: ...

    def end_session(self) -> None: ...


class Evoke(ReactorPipeline):
    """Generate an autoregressive EVOKE world from image, video, or text conditioning."""

    state: EvokeState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: EvokeConfig | None = None
        self._backend: _Backend | None = None
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

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load EVOKE weights once in its Python 3.10 worker."""
        config = read_config(config_path)
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
        self._backend = EvokeWorkerBackend(
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
        """Initialize a built-in i2v world before its first viewer connects."""
        config = self._require_config()
        self._mode = "i2v"
        self._media = config.default_image
        self._pose = None
        self._input_source = "built_in"
        self._input_name = config.default_image.name
        self._pose_name = ""
        self.state.prompt = config.stability_prompt
        self.state._restart_requested = True
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
        backend = self._backend
        try:
            if backend is not None:
                backend.end_session()
        finally:
            self._clear_controls()
            self.state._restart_requested = True
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

    async def inference(self) -> AsyncGenerator[EvokeOutput | None, None]:
        """Generate and stream one native chunk at a time from the active rollout."""
        while True:
            backend = self._backend
            planner = self._planner
            if backend is None or planner is None:
                raise RuntimeError("EVOKE was not loaded")
            if self.state._restart_requested:
                backend.reset(
                    mode=self._mode,
                    media=self._media,
                    pose=self._pose,
                    prompt=self.state.prompt,
                    seed=self._seed,
                    source_fps=self._source_fps,
                    source_height=self._source_height,
                    source_width=self._source_width,
                )
                planner.reset()
                self._chunk_index = 0
                self.state._restart_requested = False
                await self.send(self._state_update())

            trajectory = None
            if self._mode != "t2v":
                trajectory = planner.plan_chunk(
                    strafe=self.state.strafe,
                    vertical=self.state.vertical,
                    forward=self.state.forward,
                    pitch=self.state.pitch,
                    yaw=self.state.yaw,
                    roll=self.state.roll,
                    frame_count=CAMERA_POSES_PER_CHUNK,
                )
            frames = backend.generate_chunk(
                trajectory,
                seed=self._seed,
                prompt=self.state.prompt,
            )
            frames = normalize_output_frames(frames)
            expected = 33 if self._mode == "t2v" and self._chunk_index == 0 else 36
            if int(frames.shape[0]) != expected:
                raise RuntimeError(
                    f"EVOKE chunk {self._chunk_index + 1} produced {frames.shape[0]} frames; "
                    f"expected {expected}"
                )
            self._chunk_index += 1
            await self.send(self._state_update())
            yield EvokeOutput(main_video=frames)

            config = self._require_config()
            if self._chunk_index >= config.max_chunks:
                replaced = self._chunk_index
                self._request_restart()
                await self.send(
                    RolloutRestarted(
                        replaced_chunks=replaced,
                        max_chunks=config.max_chunks,
                        seed=self._seed,
                    )
                )

    def _request_restart(self) -> None:
        self.output.flush()
        self.state._restart_requested = True
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
