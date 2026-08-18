"""Serve public Matrix-Game-3.5 distilled inference through Reactor Runtime.

The adapter accepts a reference image and text prompt, then expands normalized
six-axis camera motion into Matrix's native camera-to-world matrices. A
persistent worker owns the upstream 5B model and keeps its causal rollout alive
while generating 12-frame chunks. Each chunk uses the latest motion state while
reusing the rollout's KV cache, dynamic visual context, and Patch Memory.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from reactor_runtime import (
    ClientInfo,
    CommandError,
    Idle,
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

if TYPE_CHECKING:
    from examples.matrix_game_3_5.camera_motion import (
        CameraMotionPlanner,
        MotionConfig,
    )
    from examples.matrix_game_3_5.matrix_config import MatrixConfig, prepare_runtime, read_config
    from examples.matrix_game_3_5.matrix_images import (
        normalize_output_frames,
        validate_uploaded_image,
    )
    from examples.matrix_game_3_5.matrix_schema import (
        MatrixGame35Output,
        MatrixGame35State,
        RolloutLimitReached,
        StateUpdate,
    )
    from examples.matrix_game_3_5.upstream_backend import (
        MatrixWorkerBackend,
        WorkerSettings,
    )
else:
    module_prefix = f"{__package__}." if __package__ else ""
    camera_motion = importlib.import_module(f"{module_prefix}camera_motion")
    matrix_config = importlib.import_module(f"{module_prefix}matrix_config")
    matrix_images = importlib.import_module(f"{module_prefix}matrix_images")
    matrix_schema = importlib.import_module(f"{module_prefix}matrix_schema")
    upstream_backend = importlib.import_module(f"{module_prefix}upstream_backend")
    CameraMotionPlanner = camera_motion.CameraMotionPlanner
    MotionConfig = camera_motion.MotionConfig
    MatrixConfig = matrix_config.MatrixConfig
    prepare_runtime = matrix_config.prepare_runtime
    read_config = matrix_config.read_config
    normalize_output_frames = matrix_images.normalize_output_frames
    validate_uploaded_image = matrix_images.validate_uploaded_image
    MatrixGame35Output = matrix_schema.MatrixGame35Output
    MatrixGame35State = matrix_schema.MatrixGame35State
    RolloutLimitReached = matrix_schema.RolloutLimitReached
    StateUpdate = matrix_schema.StateUpdate
    MatrixWorkerBackend = upstream_backend.MatrixWorkerBackend
    WorkerSettings = upstream_backend.WorkerSettings

logger = get_logger(__name__)

FPS = 16
_CAMERA_POSES_PER_CHUNK = 12


class _Backend(Protocol):
    """Define the blocking model operations used by the Reactor loop."""

    def reset(
        self,
        seed: int,
        anchor_image: Path | UploadedFile,
        prompt: str,
    ) -> None:
        """Reset the causal rollout from an image and text condition."""

    def generate_chunk(
        self,
        trajectory_c2w: np.ndarray,
        seed: int,
        prompt: str,
    ) -> np.ndarray:
        """Generate one RGB chunk for camera and text conditions."""

    def end_session(self) -> None:
        """Release causal state owned by the completed session."""


class MatrixGame35(ReactorPipeline):
    """Generate an image-, prompt-, and camera-controllable Matrix world."""

    state: MatrixGame35State
    output: MatrixGame35Output
    fps = FPS
    buffer_size = 1

    def __init__(self) -> None:
        super().__init__()
        self._config: MatrixConfig | None = None
        self._backend: _Backend | None = None
        self._planner: CameraMotionPlanner | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._default_prompt = ""
        self._seed = 0
        self._chunk_index = 0
        self._chunk_in_flight = False

    def load(self, config_path: Path | None) -> None:
        """Validate configuration and load Matrix weights in a persistent worker.

        Args:
            config_path: Path to ``matrix_game_3_5.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        initial_pose, intrinsics = prepare_runtime(config)
        self._config = config
        self._default_prompt = config.prompt_file.read_text(encoding="utf-8").strip()
        if not self._default_prompt:
            raise ValueError(f"prompt file is empty: {config.prompt_file}")
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            initial_pose,
            MotionConfig(
                fps=FPS,
                translation_meters_per_second=config.translation_meters_per_second,
                rotation_degrees_per_second=config.rotation_degrees_per_second,
            ),
        )
        self._backend = MatrixWorkerBackend(
            WorkerSettings(
                python_executable=config.worker_python,
                source_path=config.source_path,
                inference_config=config.inference_config,
                checkpoint=config.checkpoint.path,
                wan_dir=config.wan.path,
                tokenizer_dir=config.tokenizer_dir,
                da3_dir=config.depth.path,
                anchor_image=config.anchor_image,
                default_camera=config.camera,
                prompt_file=config.prompt_file,
                seed=config.seed,
                max_chunks=config.max_chunks,
            ),
            intrinsics,
        )
        logger.info(
            "Matrix-Game-3.5 model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.checkpoint.revision,
            fps=FPS,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize the shared world before its first viewer connects."""
        config = self._config
        if config is None:
            raise RuntimeError("Matrix-Game-3.5 was not loaded")
        self._selected_input = config.anchor_image
        self.state.prompt = self._default_prompt
        self._seed = config.seed
        self._clear_controls()
        self.state.paused = False
        self.state._step_requested = False
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._chunk_in_flight = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the current shared world state to one joining viewer."""
        await client.send(self._state_update())

    @session_ended
    async def on_session_ended(self) -> None:
        """Release causal state and controls owned by the completed world."""
        backend = self._backend
        try:
            if backend is not None:
                await asyncio.to_thread(backend.end_session)
        finally:
            self._clear_controls()
            self.state._step_requested = False
            self.state._restart_requested = True
            self.state._limit_reached = False
            self._selected_input = None
            self._chunk_index = 0
            self._chunk_in_flight = False

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held controls when their viewer leaves the live session."""
        self._clear_controls()
        await self.send(self._state_update())

    @event(
        name="set_prompt",
        description=(
            "Set the scene prompt without restarting the current world. Valid when an anchor "
            "image is selected; the text takes effect on the next generated 12-frame "
            "`main_video` chunk and does not clear the rollout limit. Returns `state_update` "
            "on success, or `command_error` when `prompt` is empty or no image is selected."
        ),
    )
    def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Scene description, up to 4096 characters and non-empty after trimming. "
                "Applied at the next generated chunk boundary; if the rollout limit has been "
                "reached, it is retained until `reset` or `set_image` starts a fresh world."
            ),
        ),
    ) -> StateUpdate:
        """Queue a prompt and return the complete shared world state."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "Matrix-Game-3.5 requires a prompt.")
        if self._selected_input is None:
            raise CommandError("image_required", "Select an image before setting a prompt.")
        self.state.prompt = normalized
        return self._state_update()

    @event(
        name="set_image",
        description=(
            "Replace the anchor image and start a fresh world. Valid at any time; the new "
            "image applies before the next generated `main_video` chunk, clears progress and "
            "the rollout limit, releases all camera axes, and resumes generation. Returns "
            "`state_update` on success, or `command_error` when the upload is not a valid "
            "JPEG, PNG, WebP, or BMP within the size limits."
        ),
    )
    def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            description=(
                "Anchor image provided through the Reactor upload protocol. JPEG, PNG, WebP, "
                "or BMP; at most 25 MiB and 100 million pixels. Replaces the active anchor "
                "when the fresh world begins."
            )
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Optional scene description, up to 4096 characters. A non-empty value is used "
                "for the fresh world's first chunk; an empty value preserves the active prompt."
            ),
        ),
    ) -> StateUpdate:
        """Select an uploaded anchor and return the fresh shared world state."""
        validate_uploaded_image(image)
        normalized = prompt.strip() or self.state.prompt.strip() or self._default_prompt
        if not normalized:
            raise CommandError("prompt_required", "Matrix-Game-3.5 requires a prompt.")
        self._selected_input = image
        self.state.prompt = normalized
        self.state.paused = False
        self._request_restart()
        return self._state_update()

    @event(
        name="set_forward",
        description=(
            "Set backward-to-forward camera translation. Valid before the rollout limit; the "
            "value is sampled at the next 12-frame chunk boundary and held for later chunks. "
            "Returns `state_update` on success, or `command_error` with "
            "`rollout_limit_reached` until `reset` or `set_image` starts a fresh world."
        ),
    )
    def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized backward (-1) to forward (1) translation, where 0 stops this "
                "axis. Applied at the next chunk boundary and held until another command "
                "changes or releases the camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Queue forward motion and return the complete shared world state."""
        self._require_available_rollout()
        self.state.forward = forward
        return self._state_update()

    @event(
        name="set_strafe",
        description=(
            "Set left-to-right camera translation. Valid before the rollout limit; the value "
            "is sampled at the next 12-frame chunk boundary and held for later chunks. Returns "
            "`state_update` on success, or `command_error` with `rollout_limit_reached` until "
            "`reset` or `set_image` starts a fresh world."
        ),
    )
    def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized left (-1) to right (1) translation, where 0 stops this axis. "
                "Applied at the next chunk boundary and held until another command changes or "
                "releases the camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Queue strafe motion and return the complete shared world state."""
        self._require_available_rollout()
        self.state.strafe = strafe
        return self._state_update()

    @event(
        name="set_vertical",
        description=(
            "Set down-to-up camera translation. Valid before the rollout limit; the value is "
            "sampled at the next 12-frame chunk boundary and held for later chunks. Returns "
            "`state_update` on success, or `command_error` with `rollout_limit_reached` until "
            "`reset` or `set_image` starts a fresh world."
        ),
    )
    def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized down (-1) to up (1) translation, where 0 stops this axis. Applied "
                "at the next chunk boundary and held until another command changes or releases "
                "the camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Queue vertical motion and return the complete shared world state."""
        self._require_available_rollout()
        self.state.vertical = vertical
        return self._state_update()

    @event(
        name="set_pitch",
        description=(
            "Set downward-to-upward camera pitch. Valid before the rollout limit; the value is "
            "sampled at the next 12-frame chunk boundary and held for later chunks. Returns "
            "`state_update` on success, or `command_error` with `rollout_limit_reached` until "
            "`reset` or `set_image` starts a fresh world."
        ),
    )
    def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized downward (-1) to upward (1) pitch, where 0 stops this axis. Applied "
                "at the next chunk boundary and held until another command changes or releases "
                "the camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Queue pitch motion and return the complete shared world state."""
        self._require_available_rollout()
        self.state.pitch = pitch
        return self._state_update()

    @event(
        name="set_yaw",
        description=(
            "Set left-to-right camera yaw. Valid before the rollout limit; the value is sampled "
            "at the next 12-frame chunk boundary and held for later chunks. Returns "
            "`state_update` on success, or `command_error` with `rollout_limit_reached` until "
            "`reset` or `set_image` starts a fresh world."
        ),
    )
    def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized left (-1) to right (1) yaw, where 0 stops this axis. Applied at the "
                "next chunk boundary and held until another command changes or releases the "
                "camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Queue yaw motion and return the complete shared world state."""
        self._require_available_rollout()
        self.state.yaw = yaw
        return self._state_update()

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise-to-clockwise camera roll. Valid before the rollout limit; the "
            "value is sampled at the next 12-frame chunk boundary and held for later chunks. "
            "Returns `state_update` on success, or `command_error` with "
            "`rollout_limit_reached` until `reset` or `set_image` starts a fresh world."
        ),
    )
    def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized counterclockwise (-1) to clockwise (1) roll, where 0 stops this "
                "axis. Applied at the next chunk boundary and held until another command "
                "changes or releases the camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Queue roll motion and return the complete shared world state."""
        self._require_available_rollout()
        self.state.roll = roll
        return self._state_update()

    @event(
        name="set_paused",
        description=(
            "Pause continuous generation before the next 12-frame chunk, or resume it. Pausing "
            "is valid at any time; resuming requires an available chunk. Either choice releases "
            "all camera axes and cancels a queued `step`. Returns `state_update` on success, or "
            "`command_error` with `rollout_limit_reached` when resuming at the limit."
        ),
    )
    def set_paused(
        self,
        paused: bool = InputField(
            default=False,
            description=(
                "Set to true to pause before the next chunk, or false to resume continuous "
                "generation. Takes effect after any in-flight chunk and releases all camera "
                "axes plus a queued `step`."
            ),
        ),
    ) -> StateUpdate:
        """Set pause state and return the complete shared world state."""
        if not paused:
            self._require_available_rollout()
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_controls()
        return self._state_update()

    @event(
        name="step",
        description=(
            "Queue one 12-frame `main_video` chunk without leaving paused mode. Valid only "
            "while generation is paused and a chunk remains available. Returns `state_update` "
            "on success, or `command_error` when continuous generation is running or the "
            "rollout limit has been reached."
        ),
    )
    def step(self) -> StateUpdate:
        """Queue one paused chunk and return the complete shared world state."""
        self._require_available_rollout()
        if not self.state.paused:
            raise CommandError("pause_required", "Pause Matrix-Game-3.5 before stepping.")
        self.state._step_requested = True
        return self._state_update()

    @event(
        name="reset",
        description=(
            "Restart from the selected anchor image and active prompt. Valid when an image is "
            "selected; the reset applies before the next generated chunk, clears progress and "
            "the rollout limit, releases all camera axes, and preserves paused mode. Returns "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Random seed for the fresh world, from 0 to 2147483647. Use -1 to retain the "
                "active seed; a non-negative value becomes active when the reset begins."
            ),
        ),
    ) -> StateUpdate:
        """Request a reset and return the complete shared world state."""
        if self._selected_input is None:
            raise CommandError("image_required", "Select an image before resetting.")
        if seed >= 0:
            self._seed = seed
        self._request_restart()
        return self._state_update()

    async def inference(self) -> AsyncGenerator[object, None]:
        """Generate chunks off-loop and emit their frames at Matrix's native FPS."""
        backend = self._backend
        planner = self._planner
        config = self._config
        if backend is None or planner is None or config is None:
            raise RuntimeError("Matrix-Game-3.5 was not loaded")

        while True:
            if self.state._restart_requested:
                selected_input = self._selected_input
                if selected_input is None:
                    yield Idle
                    continue
                prompt = self.state.prompt.strip()
                if not prompt:
                    raise RuntimeError("Matrix-Game-3.5 requires a non-empty prompt")
                self.state._restart_requested = False
                await asyncio.to_thread(
                    backend.reset,
                    self._seed,
                    selected_input,
                    prompt,
                )
                if self.state._restart_requested:
                    continue
                planner.reset()
                self._chunk_index = 0

            if self.state._limit_reached:
                yield Idle
                continue

            if self.state.paused and not self.state._step_requested:
                yield Idle
                continue

            self.state._step_requested = False
            trajectory = planner.plan_block(
                strafe=self.state.strafe,
                vertical=self.state.vertical,
                forward=self.state.forward,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                roll=self.state.roll,
                frame_count=_CAMERA_POSES_PER_CHUNK,
            )
            self._chunk_in_flight = True
            try:
                frames = await asyncio.to_thread(
                    backend.generate_chunk,
                    trajectory,
                    self._seed,
                    self.state.prompt,
                )
            finally:
                self._chunk_in_flight = False
            frames = normalize_output_frames(frames)
            if self.state._restart_requested:
                continue
            self._chunk_index += 1
            if self._chunk_index >= config.max_chunks:
                self.state._limit_reached = True
                self.state.paused = True
                self._clear_controls()
                await self.send(
                    RolloutLimitReached(
                        completed_chunks=self._chunk_index,
                        max_chunks=config.max_chunks,
                    )
                )
            await self.send(self._state_update())

            for frame in frames:
                if self.state._restart_requested:
                    break
                yield MatrixGame35Output(main_video=frame)

    def _clear_controls(self) -> None:
        """Return every camera axis to neutral."""
        self.state.forward = 0.0
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _request_restart(self) -> None:
        """Queue a fresh causal rollout and release active camera motion."""
        self._clear_controls()
        self.state._step_requested = False
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to consume new camera motion."""
        if self.state._restart_requested:
            return 1
        return self._chunk_index + 1 + int(self._chunk_in_flight)

    def _require_available_rollout(self) -> None:
        """Reject controls that cannot apply until the client resets the rollout."""
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset Matrix-Game-3.5 before requesting another chunk.",
            )

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of the shared world state."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        selected = self._selected_input
        image_source = "upload" if isinstance(selected, UploadedFile) else "built_in"
        image_name = selected.name if selected is not None else ""
        return StateUpdate(
            prompt=self.state.prompt,
            image_source=image_source,
            image_name=image_name,
            seed=self._seed,
            paused=self.state.paused,
            step_queued=self.state._step_requested,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            next_chunk=None if self.state._limit_reached else self._next_control_chunk(),
            max_chunks=max_chunks,
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
        )
