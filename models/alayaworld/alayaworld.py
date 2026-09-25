"""Serve AlayaWorld distilled autoregressive inference through Reactor Runtime.

The adapter keeps AlayaWorld's public ``FlashAlayaPipeline`` intact. Reactor
controls provide normalized six-axis camera motion, which is expanded into the
camera-to-world trajectory consumed by the upstream action and spatial-memory
paths. Prompt updates and camera controls are sampled at chunk boundaries.
"""

from __future__ import annotations

import secrets
from copy import deepcopy
from pathlib import Path
from typing import Literal

from alayaworld_assets import (
    AlayaWorldConfig,
    read_config,
    scene_image_path,
    scene_prompt_path,
)
from alayaworld_camera import CameraMotionPlanner, MotionConfig
from alayaworld_images import validate_uploaded_image
from alayaworld_model import (
    FPS,
    FRAMES_PER_CHUNK,
    AlayaInput,
    AlayaResult,
    AlayaWorldModel,
)
from alayaworld_types import (
    AlayaWorldOutput,
    AlayaWorldState,
    CameraMotionChanged,
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
_UPLOAD_DEFAULT_PROMPT = "Continue the visual scene shown in the reference image."


class AlayaWorld(ReactorApp):
    """Run AlayaWorld with live prompt and six-axis camera controls."""

    state: AlayaWorldState
    # No `fps` pin: playout follows the measured cost of each variable-length
    # turn, and one chunk of queued frames is the smallest bound holding a turn.
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self.engine = AlayaWorldModel()
        self._config: AlayaWorldConfig | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._seed = 0
        self._ar_index = 0
        self._active_prompt = ""
        self._reset_in_flight = False
        self._chunk_in_flight = False
        self._camera: CameraMotionPlanner | None = None
        self._frames_wanted = 0
        self._pending_camera: CameraMotionPlanner | None = None

    def load(self, config_path: Path | None) -> None:
        """Load the native model and retain its interaction configuration."""
        self._config = read_config(config_path, get_weights_path())
        self.engine.load(self._config)

    @session_started
    def on_session_started(self) -> None:
        """Wait for an uploaded or randomly selected image."""
        config = self._config
        if config is None:
            raise RuntimeError("AlayaWorld was not loaded")
        self.state.prompt = ""
        self._clear_camera_controls()
        self.state._world_id = 0
        self.state._applied_world_id = None
        self._seed = config.seed
        self._selected_input = None
        self.engine.reset()
        self._frames_wanted = 0
        self._pending_camera = None
        self._camera = None
        self._ar_index = 0
        self._active_prompt = ""
        self._reset_in_flight = False
        self._chunk_in_flight = False

    @session_ended
    def on_session_ended(self) -> None:
        """Release the selected image and rollout at session end."""
        self._clear_camera_controls()
        self.state._world_id = 0
        self.state._applied_world_id = None
        self._selected_input = None
        self.engine.reset()
        self._frames_wanted = 0
        self._pending_camera = None
        self._camera = None
        self._ar_index = 0
        self._active_prompt = ""
        self._reset_in_flight = False
        self._chunk_in_flight = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera motion when a viewer leaves the live session."""
        self._clear_camera_controls()
        await self._send_state_update()

    @event(
        name="set_prompt",
        description=(
            "Set the scene prompt for the next 32-frame chunk without resetting the world. "
            "Requires a selected image. Emits `prompt_queued` and broadcasts `state_update` "
            "on success, or `command_error` when the prompt is empty or no image is selected."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene prompt, trimmed and sampled when the next 32-frame chunk "
                "starts. Requires an image selected by `set_image` or `random_image`."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and confirm the chunk expected to consume it."""
        self._require_selected_image()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "AlayaWorld requires a non-empty prompt."
            )
        self.state.prompt = normalized
        message = PromptQueued(
            prompt=normalized, applies_to_chunk=self._next_control_chunk()
        )
        await self._send_state_update()
        return message

    @event(
        name="set_forward",
        description=(
            "Set backward or forward camera velocity for forthcoming chunks. Requires a "
            "selected image. Emits `camera_motion_changed` and broadcasts `state_update` on "
            "success, or `command_error` when no image is selected."
        ),
    )
    async def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Backward (-1) to forward (1) velocity sampled when the next chunk starts "
                "and held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue forward motion and return the complete camera state."""
        self._require_selected_image()
        self.state.forward = forward
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_strafe",
        description=(
            "Set left or right camera velocity for forthcoming chunks. Requires a selected "
            "image. Emits `camera_motion_changed` and broadcasts `state_update` on success, "
            "or `command_error` when no image is selected."
        ),
    )
    async def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Left (-1) to right (1) velocity sampled when the next chunk starts and held "
                "until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue strafe motion and return the complete camera state."""
        self._require_selected_image()
        self.state.strafe = strafe
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_vertical",
        description=(
            "Set down or up camera velocity for forthcoming chunks. Requires a selected image. "
            "Emits `camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected."
        ),
    )
    async def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Down (-1) to up (1) velocity sampled when the next chunk starts and held "
                "until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue vertical motion and return the complete camera state."""
        self._require_selected_image()
        self.state.vertical = vertical
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_pitch",
        description=(
            "Set down or up camera pitch velocity for forthcoming chunks. Requires a selected "
            "image. Emits `camera_motion_changed` and broadcasts `state_update` on success, "
            "or `command_error` when no image is selected."
        ),
    )
    async def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Look down (-1) to up (1) velocity sampled when the next chunk starts and held "
                "until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue pitch motion and return the complete camera state."""
        self._require_selected_image()
        self.state.pitch = pitch
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_yaw",
        description=(
            "Set left or right camera yaw velocity for forthcoming chunks. Requires a selected "
            "image. Emits `camera_motion_changed` and broadcasts `state_update` on success, "
            "or `command_error` when no image is selected."
        ),
    )
    async def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Turn left (-1) to right (1) velocity sampled when the next chunk starts and "
                "held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue yaw motion and return the complete camera state."""
        self._require_selected_image()
        self.state.yaw = yaw
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise or clockwise camera roll velocity for forthcoming chunks. "
            "Requires a selected image. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Counterclockwise (-1) to clockwise (1) velocity sampled when the next chunk "
                "starts and held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue roll motion and return the complete camera state."""
        self._require_selected_image()
        self.state.roll = roll
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="reset",
        description=(
            "Queue a fresh rollout from the selected image, current prompt, and neutral camera "
            "motion. Requires a selected image. Emits `rollout_reset_queued` and broadcasts "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Random seed for the fresh rollout. Use -1 to retain the active seed; a "
                "non-negative value replaces it when the reset begins."
            ),
        ),
    ) -> RolloutResetQueued:
        """Request a fresh rollout and report the seed and replaced chunk count."""
        self._require_selected_image()
        if seed >= 0:
            self._seed = seed
        completed_chunks = self._ar_index
        self._clear_camera_controls()
        self.state._world_id += 1
        self.output.flush()
        message = RolloutResetQueued(
            trigger="manual",
            seed=self._seed,
            completed_chunks=completed_chunks,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded image, queue a fresh rollout, and resume continuous generation. "
            "Can replace the image at any time. Emits `image_selected` and broadcasts "
            "`state_update` on success, or `command_error` when the upload is missing, too "
            "large, mislabeled, or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "Reference image uploaded through the Reactor upload protocol. JPEG, PNG, WebP, "
                "or BMP up to 25 MiB and 100 million pixels; EXIF orientation is applied before "
                "the image is center-cropped to `main_video` resolution."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene prompt for the fresh rollout. Whitespace is trimmed; an empty "
                "value retains the current prompt or uses a generic continuation prompt when "
                "none exists."
            ),
        ),
    ) -> ImageSelected:
        """Validate uploaded image bytes and select them for the next rollout."""
        validate_uploaded_image(image)
        self._selected_input = image
        self.state.prompt = (
            prompt.strip() or self.state.prompt.strip() or _UPLOAD_DEFAULT_PROMPT
        )
        self.state._world_id += 1
        self.output.flush()
        self._clear_camera_controls()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    @event(
        name="random_image",
        description=(
            "Select a built-in image and its matching prompt, queue a fresh rollout, and resume "
            "continuous generation. Valid when built-in examples are configured. Emits "
            "`image_selected` and broadcasts `state_update` on success, or `command_error` when "
            "no usable image or prompt is available."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a different configured example image when possible."""
        config = self._config
        if config is None or not config.random_inputs:
            raise CommandError(
                "image_unavailable", "No built-in images are configured."
            )
        candidates = [
            path for path in config.random_inputs if path != self._selected_input
        ]
        selected = secrets.choice(candidates or list(config.random_inputs))
        self._selected_input = selected
        prompt = scene_prompt_path(selected).read_text(encoding="utf-8").strip()
        if not prompt:
            raise CommandError(
                "prompt_unavailable", "The selected built-in image has no prompt."
            )
        self.state.prompt = prompt
        self.state._world_id += 1
        self.output.flush()
        self._clear_camera_controls()
        message = ImageSelected(
            source="built_in",
            filename=scene_image_path(selected).name,
            prompt=prompt,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    async def process_input(self) -> AlayaInput:
        """Resolve the anchor and plan the next native camera window."""
        if self._selected_input is None:
            raise ApplicationError("Select an image before generating.")
        if self._config is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        new_world = self.state._world_id != self.state._applied_world_id
        image = None
        trajectory = None
        if new_world:
            image = (
                self._selected_input.data
                if isinstance(self._selected_input, UploadedFile)
                else self._selected_input
            )
        else:
            if self._camera is None:
                raise RuntimeError("AlayaWorld camera planner was not initialized")
            self._pending_camera = deepcopy(self._camera)
            trajectory = self._pending_camera.plan(
                strafe=self.state.strafe,
                vertical=self.state.vertical,
                forward=self.state.forward,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                roll=self.state.roll,
                frame_count=self._frames_wanted,
            )
        self._reset_in_flight = new_world
        self._chunk_in_flight = not new_world
        return AlayaInput(
            world_id=self.state._world_id,
            prompt=self.state.prompt,
            seed=self._seed,
            image=image,
            trajectory=trajectory,
        )

    def generate(self, input: AlayaInput) -> AlayaResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> AlayaWorldOutput | None:
        """Acknowledge model progress and publish successfully generated frames."""
        self._reset_in_flight = False
        self._chunk_in_flight = False
        if outcome.error is not None:
            self._pending_camera = None
            # Native inference may have mutated its cache; automatic retry could
            # silently skip a chunk. Surface the failure and require an explicit reset.
            raise outcome.error
        result: AlayaResult = outcome.result
        if result.initial_pose is not None:
            config = self._config
            self._camera = CameraMotionPlanner(
                result.initial_pose,
                MotionConfig(
                    fps=FPS,
                    strafe_units_per_second=config.strafe_units_per_second,
                    vertical_units_per_second=config.vertical_units_per_second,
                    forward_units_per_second=config.forward_units_per_second,
                    pitch_degrees_per_second=config.pitch_degrees_per_second,
                    yaw_degrees_per_second=config.yaw_degrees_per_second,
                    roll_degrees_per_second=config.roll_degrees_per_second,
                ),
            )
        elif self._pending_camera is not None:
            self._camera = self._pending_camera
        self._pending_camera = None
        self.state._applied_world_id = result.world_id
        self._frames_wanted = result.frames_wanted
        self._ar_index = result.completed_chunks
        self._active_prompt = result.active_prompt
        if result.frames is None:
            return None
        logger.info(
            "AlayaWorld chunk ready",
            chunk=result.completed_chunks,
            frames=int(result.frames.shape[0]),
            seconds=round(outcome.elapsed, 3),
        )
        if result.completed_chunks >= self._config.max_chunks_per_rollout:
            self._clear_camera_controls()
            self.state._world_id += 1
            await self.send(
                RolloutResetQueued(
                    trigger="automatic_chunk_limit",
                    seed=self._seed,
                    completed_chunks=result.completed_chunks,
                    applies_to_chunk=1,
                )
            )
        await self._send_state_update()
        return AlayaWorldOutput(main_video=result.frames)

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of the shared world state."""
        selected = self._selected_input
        image_source: Literal["uploaded", "built_in"] | None = None
        image_name: str | None = None
        if isinstance(selected, UploadedFile):
            image_source = "uploaded"
            image_name = selected.name
        elif selected is not None:
            image_source = "built_in"
            try:
                image_name = scene_image_path(selected).name
            except FileNotFoundError:
                image_name = selected.name
        config = self._config
        return StateUpdate.from_state(
            self.state,
            image_source=image_source,
            image_name=image_name,
            active_prompt=self._active_prompt or None,
            seed=self._seed,
            generating=self._reset_in_flight or self._chunk_in_flight,
            completed_chunks=self._ar_index,
            next_chunk=None if selected is None else self._next_control_chunk(),
            max_chunks=config.max_chunks_per_rollout if config is not None else 0,
        )

    async def _send_state_update(self) -> None:
        """Broadcast the complete observable session state."""
        await self.send(self._state_update())

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to consume a new control value."""
        starts_new_rollout = (
            self._selected_input is None
            or self._reset_in_flight
            or self.state._world_id != self.state._applied_world_id
        )
        if starts_new_rollout:
            return 1
        return self._ar_index + 1 + int(self._chunk_in_flight)

    def _camera_motion_changed(self) -> CameraMotionChanged:
        """Describe the complete camera state after a control event."""
        return CameraMotionChanged(
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            forward=self.state.forward,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
            applies_to_chunk=self._next_control_chunk(),
        )

    def _clear_camera_controls(self) -> None:
        """Return all camera controls to neutral."""
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.forward = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _require_selected_image(self) -> None:
        """Require a rollout origin before accepting world controls."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Upload an image or select Random Image first."
            )
