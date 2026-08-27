"""Serve SANA-WM streaming inference through Reactor Runtime.

The adapter preserves the upstream 4-step self-forcing Stage-1 sampler,
chunk-causal refiner KV window, and causal VAE feature cache. Each Reactor turn
advances those three persistent states once and emits one 24-frame video chunk.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image, ImageOps
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

from sana_wm_assets import prepare_source, read_config
from sana_wm_backend import (
    PIXEL_FRAMES_PER_CHUNK,
    SanaStreamingBackend,
    TrajectoryCompleteError,
    load_numpy_upload,
)
from sana_wm_types import (
    Control,
    ControlChanged,
    ControlsReleased,
    ImageSelected,
    IntrinsicsSource,
    PromptChanged,
    RolloutResetQueued,
    SanaWMConfig,
    SanaWMOutput,
    SanaWMState,
    StateUpdate,
    TrajectoryExhausted,
    TrajectorySelected,
)

logger = get_logger(__name__)

_CONTROL_ORDER: tuple[Control, ...] = (
    "forward",
    "back",
    "strafe_left",
    "strafe_right",
    "yaw_left",
    "yaw_right",
    "pitch_up",
    "pitch_down",
)
_CONTROL_SET = frozenset(_CONTROL_ORDER)
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000
_DEFAULT_PROMPT = "Continue the visual scene shown in the reference image."


def _validate_image(file: UploadedFile) -> None:
    """Reject oversized or undecodable first-frame uploads."""
    import io

    if not file.data:
        raise CommandError("image_empty", "Upload a non-empty image.")
    if file.size > _MAX_UPLOAD_BYTES:
        raise CommandError("image_too_large", "Images must be at most 25 MiB.")
    try:
        with Image.open(io.BytesIO(file.data)) as image:
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise CommandError(
                    "image_too_many_pixels",
                    "Images must contain at most 100 million pixels.",
                )
            ImageOps.exif_transpose(image).convert("RGB").load()
    except CommandError:
        raise
    except Exception as exc:
        raise CommandError(
            "image_invalid", "Upload a valid JPEG, PNG, WebP, or BMP image."
        ) from exc


def _validate_numpy(file: UploadedFile, *, kind: str) -> np.ndarray:
    """Decode a NumPy upload and turn parser failures into command errors."""
    if not file.data:
        raise CommandError(f"{kind}_empty", f"Upload a non-empty {kind} .npy file.")
    if file.size > _MAX_UPLOAD_BYTES:
        raise CommandError(f"{kind}_too_large", f"{kind} files must be at most 25 MiB.")
    try:
        return load_numpy_upload(file)
    except ValueError as exc:
        raise CommandError(f"{kind}_invalid", str(exc)) from exc


def _validate_intrinsics(file: UploadedFile) -> None:
    """Reject calibration arrays that cannot enter the upstream camera path."""
    raw = _validate_numpy(file, kind="intrinsics")
    supported = (
        raw.shape in {(4,), (3, 3)}
        or (raw.ndim == 2 and raw.shape[0] > 0 and raw.shape[1] == 4)
        or (raw.ndim == 3 and raw.shape[0] > 0 and raw.shape[1:] == (3, 3))
    )
    if not supported:
        raise CommandError(
            "intrinsics_shape",
            "Intrinsics must have shape (4,), (F,4), (3,3), or (F,3,3) with F positive.",
        )
    try:
        values = raw.astype(np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CommandError(
            "intrinsics_values", "Intrinsics must contain real numeric values."
        ) from exc
    if not np.isfinite(values).all():
        raise CommandError(
            "intrinsics_values", "Intrinsics must contain only finite values."
        )
    if values.ndim == 1:
        focal_lengths = values[:2]
    elif values.shape[-1] == 4:
        focal_lengths = values[..., :2]
    else:
        focal_lengths = values[..., (0, 1), (0, 1)]
    if np.any(focal_lengths <= 0):
        raise CommandError(
            "intrinsics_values", "Intrinsics focal lengths must be positive."
        )


class SanaWM(ReactorPipeline):
    """Generate an image-, prompt-, and camera-controllable SANA-WM world."""

    state: SanaWMState
    buffer_size = PIXEL_FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: SanaWMConfig | None = None
        self._backend: SanaStreamingBackend | None = None
        self._selected_image: Path | UploadedFile | None = None
        self._image_source: Literal["uploaded", "built_in"] | None = None
        self._intrinsics_input: Path | UploadedFile | None = None
        self._intrinsics_source: IntrinsicsSource | None = None
        self._trajectory: np.ndarray | None = None
        self._trajectory_name: str | None = None
        self._seed = 0
        self._active_prompt: str | None = None
        self._chunk_index = 0
        self._generating = False
        self._chunk_in_flight = False

    def load(self, config_path: Path | None) -> None:
        """Prepare pinned public assets and load SANA-WM weights once.

        Args:
            config_path: Path to `sana_wm.yaml` supplied by the Runtime launcher.
        """
        config = read_config(config_path)
        prepare_source(config)
        self._config = config
        self._seed = config.seed
        self._backend = SanaStreamingBackend(config)
        logger.info(
            "SANA-WM model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.streaming.revision,
            chunk_frames=PIXEL_FRAMES_PER_CHUNK,
            max_chunks=config.max_chunks,
            stage1_cached_blocks=config.num_cached_blocks,
            refiner_kv_frames=config.refiner_kv_max_frames,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty shared world for continuous generation."""
        self._selected_image = None
        self._image_source = None
        self._intrinsics_input = None
        self._intrinsics_source = None
        self._trajectory = None
        self._trajectory_name = None
        self._active_prompt = None
        self._chunk_index = 0
        self._generating = False
        self._chunk_in_flight = False
        self.state.prompt = ""
        self.state._trajectory_exhausted = False
        self.state._held_controls = set()
        self.state._reset_requested = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send a complete shared-state snapshot to a joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held controls when their viewer leaves the session."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        """Release per-world caches while retaining startup-owned model weights."""
        backend = self._backend
        if backend is not None:
            backend.end_session()
        self._selected_image = None
        self._clear_controls()

    @event(
        name="set_image",
        description=(
            "Select an uploaded first frame and begin continuous 24-frame chunk generation. "
            "An optional native intrinsics .npy avoids Pi3X estimation. "
            "Emits `image_selected` and `state_update` on success, or `command_error` when an "
            "upload is empty, too large, undecodable, or has an unsupported calibration shape."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description=(
                "First-frame JPEG, PNG, WebP, or BMP uploaded through Reactor; at most 25 MiB "
                "and 100 million pixels."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional non-empty scene description. An empty value preserves the active "
                "prompt or uses a generic continuation prompt."
            ),
        ),
        intrinsics: UploadedFile | None = InputField(  # noqa: B008
            default=None,
            moderate=True,
            description=(
                "Optional NumPy .npy calibration shaped (4,), (F,4), (3,3), or (F,3,3) in "
                "input-image pixels. Pi3X estimates calibration when omitted."
            ),
        ),
    ) -> ImageSelected:
        """Validate uploads and queue a fresh continuously generated world."""
        _validate_image(image)
        if intrinsics is not None:
            _validate_intrinsics(intrinsics)
        effective_prompt = (
            prompt.strip() or self.state.prompt.strip() or _DEFAULT_PROMPT
        )
        self._selected_image = image
        self._image_source = "uploaded"
        self._intrinsics_input = intrinsics
        self._intrinsics_source = "uploaded" if intrinsics is not None else "estimated"
        self._trajectory = None
        self._trajectory_name = None
        self.state.prompt = effective_prompt
        self._queue_reset()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=effective_prompt,
            intrinsics_source=self._intrinsics_source,
        )
        await self._send_state_update()
        return message

    @event(
        name="random_image",
        description=(
            "Select a different built-in SANA-WM first frame and prompt, then begin "
            "continuous 24-frame chunk generation. Emits `image_selected` and "
            "`state_update` on success, or `command_error` if the selected example has no prompt."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Choose a built-in upstream example for an immediately visible first turn."""
        config = self._require_config()
        current_name = (
            self._selected_image.name
            if isinstance(self._selected_image, Path)
            else None
        )
        candidates = [
            scene for scene in config.scenes if scene.image.name != current_name
        ]
        scene = secrets.choice(candidates or list(config.scenes))
        prompt = scene.prompt.read_text(encoding="utf-8").strip()
        if not prompt:
            raise CommandError("prompt_unavailable", f"{scene.name} has no prompt.")
        self._selected_image = scene.image
        self._image_source = "built_in"
        self._intrinsics_input = scene.intrinsics
        self._intrinsics_source = "built_in"
        self._trajectory = None
        self._trajectory_name = None
        self.state.prompt = prompt
        self._queue_reset()
        message = ImageSelected(
            source="built_in",
            filename=scene.image.name,
            prompt=prompt,
            intrinsics_source="built_in",
        )
        await self._send_state_update()
        return message

    @event(
        name="set_prompt",
        description=(
            "Set non-empty scene text and restart generation from the selected first frame so "
            "the new prompt applies from chunk 1. Emits "
            "`prompt_changed` and `state_update` on success, or `command_error` before image "
            "selection or when the trimmed prompt is empty."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Non-empty scene description, trimmed before cache initialization.",
        ),
    ) -> PromptChanged:
        """Queue prompt-conditioned fresh caches and report their first affected chunk."""
        self._require_image()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "SANA-WM requires a non-empty prompt."
            )
        self.state.prompt = normalized
        self._queue_reset()
        message = PromptChanged(prompt=normalized, applies_to_chunk=1)
        await self._send_state_update()
        return message

    @event(
        name="set_control",
        description=(
            "Press or release one native SANA-WM control. Values are held and sampled at the "
            "next 24-frame boundary; simultaneous controls are preserved. Valid only in live "
            "interactive mode after image selection. Emits `control_changed` and `state_update` "
            "on success, or `command_error` for an unavailable mode or control."
        ),
    )
    async def set_control(
        self,
        control: Control = InputField(  # noqa: B008 - schema field declaration
            default="forward",
            description=(
                "Canonical control: forward/back, strafe_left/strafe_right, yaw_left/yaw_right, "
                "or pitch_up/pitch_down."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description="True holds the control; false releases it.",
        ),
    ) -> ControlChanged:
        """Change one held canonical control and return the complete control set."""
        self._require_image()
        if self._trajectory is not None:
            raise CommandError(
                "trajectory_active",
                "Switch to interactive controls before changing held controls.",
            )
        if control not in _CONTROL_SET:
            raise CommandError("control_invalid", f"Unknown SANA-WM control: {control}")
        if pressed:
            self.state._held_controls.add(control)
        else:
            self.state._held_controls.discard(control)
        held = self._ordered_controls()
        message = ControlChanged(
            control=control,
            pressed=pressed,
            held_controls=held,
            applies_to_chunk=self._next_chunk(),
        )
        await self._send_state_update()
        return message

    @event(
        name="release_controls",
        description=(
            "Release every held live camera control. The upstream velocity smoother coasts "
            "toward neutral on the next chunk. Valid after image selection. Emits "
            "`controls_released` and `state_update` on success, or `command_error` before an "
            "image is selected."
        ),
    )
    async def release_controls(self) -> ControlsReleased:
        """Return all live camera controls to neutral."""
        self._require_image()
        self._clear_controls()
        message = ControlsReleased(applies_to_chunk=self._next_chunk())
        await self._send_state_update()
        return message

    @event(
        name="set_camera_trajectory",
        description=(
            "Upload SANA-WM's native (F,4,4) camera-to-world NumPy trajectory and queue a "
            "fresh rollout that consumes it in 24-frame chunks. Valid after image selection. "
            "Emits `trajectory_selected` and `state_update` on success, or `command_error` for "
            "a missing image, invalid upload, unsupported shape, or fewer than 25 poses."
        ),
    )
    async def set_camera_trajectory(
        self,
        trajectory: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description=(
                "NumPy .npy array shaped (F,4,4), with at least 25 camera-to-world poses. "
                "Frame zero anchors the relative trajectory."
            ),
        ),
    ) -> TrajectorySelected:
        """Select an upstream-native finite trajectory for a fresh rollout."""
        self._require_image()
        raw = _validate_numpy(trajectory, kind="trajectory")
        try:
            values = raw.astype(np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError(
                "trajectory_values",
                "Camera trajectory must contain real numeric values.",
            ) from exc
        if values.ndim != 3 or values.shape[1:] != (4, 4):
            raise CommandError(
                "trajectory_shape",
                f"Camera trajectory must have shape (F,4,4); got {values.shape}.",
            )
        if values.shape[0] < PIXEL_FRAMES_PER_CHUNK + 1:
            raise CommandError(
                "trajectory_too_short",
                "Camera trajectory must contain at least 25 poses.",
            )
        if not np.isfinite(values).all():
            raise CommandError(
                "trajectory_values",
                "Camera trajectory must contain only finite values.",
            )
        try:
            np.linalg.inv(values[0])
        except np.linalg.LinAlgError as exc:
            raise CommandError(
                "trajectory_origin_singular",
                "The first camera pose must be invertible.",
            ) from exc
        self._trajectory = values
        self._trajectory_name = trajectory.name
        self._clear_controls()
        self._queue_reset()
        message = TrajectorySelected(
            filename=trajectory.name,
            frames=int(values.shape[0]),
            available_chunks=(int(values.shape[0]) - 1) // PIXEL_FRAMES_PER_CHUNK,
        )
        await self._send_state_update()
        return message

    @event(
        name="use_interactive_controls",
        description=(
            "Leave finite trajectory playback and queue a fresh rollout controlled by native "
            "held camera actions. Valid after image selection. Emits `rollout_reset_queued` and "
            "`state_update` on success, or `command_error` before an image is selected."
        ),
    )
    async def use_interactive_controls(self) -> RolloutResetQueued:
        """Restore live controls as the camera source for a fresh world."""
        self._require_image()
        replaced = self._chunk_index
        self._trajectory = None
        self._trajectory_name = None
        self._clear_controls()
        self._queue_reset()
        message = RolloutResetQueued(
            trigger="manual",
            seed=self._seed,
            replaced_chunks=replaced,
        )
        await self._send_state_update()
        return message

    @event(
        name="reset",
        description=(
            "Restart generation from the selected first frame and current prompt. An optional "
            "non-negative seed replaces the active seed, and continuous generation resumes. "
            "Emits `rollout_reset_queued` and `state_update` on success, or `command_error` "
            "before an image is selected."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh rollout seed, or -1 to preserve the active seed.",
        ),
    ) -> RolloutResetQueued:
        """Queue a reproducible fresh rollout without reloading model weights."""
        self._require_image()
        if seed >= 0:
            self._seed = seed
        replaced = self._chunk_index
        self._queue_reset()
        message = RolloutResetQueued(
            trigger="manual", seed=self._seed, replaced_chunks=replaced
        )
        await self._send_state_update()
        return message

    async def inference(self) -> AsyncGenerator[SanaWMOutput | None, None]:
        """Advance and emit one native upstream chunk off-loop."""
        backend = self._backend
        config = self._require_config()
        if backend is None:
            raise RuntimeError("SANA-WM backend was not loaded")
        while True:
            if self._selected_image is None:
                yield None
                continue

            if (
                self._chunk_index >= config.max_chunks
                and not self.state._reset_requested
            ):
                replaced = self._chunk_index
                self._queue_reset()
                await self.send(
                    RolloutResetQueued(
                        trigger="automatic_chunk_limit",
                        seed=self._seed,
                        replaced_chunks=replaced,
                    )
                )
                await self._send_state_update()

            if self.state._reset_requested:
                self.state._reset_requested = False
                self._generating = True
                self.output.flush()
                await self._send_state_update()
                try:
                    backend.reset(
                        self._selected_image,
                        self.state.prompt,
                        self._seed,
                        intrinsics_source=self._intrinsics_input,
                        trajectory=self._trajectory,
                    )
                    self._chunk_index = 0
                    self._active_prompt = self.state.prompt
                finally:
                    self._generating = False
                await self._send_state_update()

            if self.state._trajectory_exhausted:
                yield None
                continue

            controls = set(self._ordered_controls())
            self._chunk_in_flight = True
            self._generating = True
            await self._send_state_update()
            trajectory_exhausted = False
            try:
                frames = backend.generate_chunk(controls)
            except TrajectoryCompleteError:
                trajectory_exhausted = True
            finally:
                self._chunk_in_flight = False
                self._generating = False
            if trajectory_exhausted:
                self.state._trajectory_exhausted = True
                trajectory_frames = backend.trajectory_frames or 0
                await self.send(
                    TrajectoryExhausted(
                        completed_chunks=self._chunk_index,
                        trajectory_frames=trajectory_frames,
                    )
                )
                await self._send_state_update()
                yield None
                continue
            self._chunk_index = backend.chunk_index
            await self._send_state_update()
            yield SanaWMOutput(main_video=frames)

    def _require_config(self) -> SanaWMConfig:
        """Return loaded configuration or raise an internal lifecycle error."""
        if self._config is None:
            raise RuntimeError("SANA-WM was not loaded")
        return self._config

    def _require_image(self) -> None:
        """Reject commands whose state has no first-frame image."""
        if self._selected_image is None:
            raise CommandError("image_required", "Select an image before this command.")

    def _queue_reset(self) -> None:
        """Queue fresh upstream caches for the next inference boundary."""
        self._clear_controls()
        self.state._trajectory_exhausted = False
        self.state._reset_requested = True
        self._active_prompt = None
        self._chunk_index = 0

    def _clear_controls(self) -> None:
        """Release the complete canonical held-control set."""
        self.state._held_controls = set()

    def _ordered_controls(self) -> list[Control]:
        """Return held controls in stable schema order."""
        return [
            control
            for control in _CONTROL_ORDER
            if control in self.state._held_controls
        ]

    def _next_chunk(self) -> int:
        """Return the one-based chunk expected to consume a newly accepted command."""
        if self.state._reset_requested:
            return 1
        return self._chunk_index + 1 + int(self._chunk_in_flight)

    async def _send_state_update(self) -> None:
        """Broadcast a complete observable-state snapshot."""
        await self.send(self._state_update())

    def _state_update(self) -> StateUpdate:
        """Build the complete client-facing state snapshot."""
        selected = self._selected_image
        image_name = selected.name if selected is not None else None
        return StateUpdate(
            image_source=self._image_source,
            image_name=image_name,
            intrinsics_source=self._intrinsics_source,
            prompt=self.state.prompt.strip() or None,
            active_prompt=self._active_prompt,
            control_mode="trajectory"
            if self._trajectory is not None
            else "interactive",
            trajectory_name=self._trajectory_name,
            trajectory_frames=(
                int(self._trajectory.shape[0]) if self._trajectory is not None else None
            ),
            held_controls=cast(list[Control], self._ordered_controls()),
            seed=self._seed,
            trajectory_exhausted=self.state._trajectory_exhausted,
            reset_queued=self.state._reset_requested,
            generating=self._generating,
            completed_chunks=self._chunk_index,
            next_chunk=self._next_chunk() if selected is not None else None,
            max_chunks=self._config.max_chunks if self._config is not None else 0,
        )
