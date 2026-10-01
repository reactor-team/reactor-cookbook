"""Serve the original Open-Oasis 500M sampler as a playable Reactor world."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from open_oasis_assets import (
    download_checkpoints,
    prepare_source,
    OpenOasisConfig,
    read_config,
)
from open_oasis_images import decode_image, decode_video
from open_oasis_model import OpenOasisModel, OpenOasisInput, OpenOasisResult
from open_oasis_types import (
    KEYS,
    MOUSE_BUTTONS,
    ActionChanged,
    ConditioningChanged,
    OpenOasisOutput,
    OpenOasisState,
    RolloutReset,
    StateUpdate,
)
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    ApplicationError,
    StepOutcome,
    get_weights_path,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)
_KEY_ACTION = {
    "e": "inventory",
    "escape": "ESC",
    **{str(i): f"hotbar.{i}" for i in range(1, 10)},
    "w": "forward",
    "s": "back",
    "a": "left",
    "d": "right",
    "space": "jump",
    "shift": "sneak",
    "ctrl": "sprint",
    "f": "swapHands",
    "q": "drop",
}
_MOUSE_ACTION = {"left": "attack", "right": "use", "middle": "pickItem"}
_ACTION_KEYS = [
    "inventory",
    "ESC",
    *[f"hotbar.{i}" for i in range(1, 10)],
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]
_IMAGE_FIELD = InputField(
    moderate=True,
    description=(
        "Minecraft starting frame uploaded through Reactor's file-upload flow. The file "
        "must have an `image/*` media type and decode successfully; EXIF orientation is "
        "applied before it is resized to 640x360. It replaces the previous image or video "
        "context at the next generated frame."
    ),
)
_VIDEO_FIELD = InputField(
    moderate=True,
    description=(
        "Minecraft visual context uploaded through Reactor's file-upload flow. The file "
        "must have a `video/*` media type and contain every frame requested by `offset` and "
        "`prompt_frames`; accepted frames replace the previous context at the next generated "
        "frame."
    ),
)


class OpenOasis(ReactorApp):
    """Generate one action-conditioned Minecraft frame per model step."""

    state: OpenOasisState
    buffer_size = 1

    def __init__(self) -> None:
        super().__init__()
        self._config: OpenOasisConfig | None = None
        self.engine: OpenOasisModel
        self._sample: np.ndarray | None = None

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path)
        self._config = config
        prepare_source(config)
        model_path, vae_path = download_checkpoints(config, get_weights_path())
        # Decode the explicitly selected, attributed application sample once.
        self._sample = decode_image(
            (Path(__file__).parent / "example_images/sample_image_0.png").read_bytes()
        )
        self.engine = OpenOasisModel()
        self.engine.load(config, model_path, vae_path)
        logger.info(
            "Open-Oasis ready",
            revision=config.source_revision,
            ddim_steps=config.ddim_steps,
            context_frames=config.context_frames,
        )

    @session_started
    def on_session_started(self) -> None:
        if self._config is None:
            raise RuntimeError("Open-Oasis was not loaded")
        self.state._seed = self._config.seed
        self.state._world_id += 1
        self.state._applied_world_id = None
        self.state._conditioning = None
        self.state._conditioning_name = "none"
        self._clear_controls()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release controls without destroying the shared session's world."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        self.output.flush()
        self.engine.reset()

    @event(
        name="set_key_state",
        description=(
            "Hold or release one Minecraft keyboard key for subsequent frames. Valid while "
            "the session is active; a newly pressed key is also preserved for one frame if it "
            "is released before inference samples it. Emits `action_changed` and broadcasts "
            "`state_update`. Unsupported values are rejected before state changes."
        ),
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=KEYS,
            description=(
                "Minecraft keyboard key to hold or release. WASD moves; `space` jumps; `shift` "
                "sneaks; `ctrl` sprints; `e` opens inventory; `escape` pauses; `f` swaps "
                "hands; `q` drops; and 1-9 selects a hotbar slot. The state starts with the "
                "next generated frame and persists until another `set_key_state` changes it "
                "or controls are cleared."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true to hold `key` on subsequent generated frames or false to release it."
            ),
        ),
    ) -> ActionChanged:
        if pressed and key not in self.state._pressed_keys:
            self.state._pending_key_pulses = self.state._pending_key_pulses.union(
                (key,)
            )
        self.state._pressed_keys = (
            self.state._pressed_keys.union((key,))
            if pressed
            else self.state._pressed_keys.difference((key,))
        )
        await self.send(self._state_update())
        return self._action_changed()

    @event(
        name="set_mouse_button_state",
        description=(
            "Hold or release one Minecraft mouse button for subsequent frames. Valid while "
            "the session is active; a newly pressed button is also preserved for one frame if "
            "it is released before inference samples it. Emits `action_changed` and broadcasts "
            "`state_update`. Unsupported values are rejected before state changes."
        ),
    )
    async def set_mouse_button_state(
        self,
        button: str = InputField(
            default="left",
            choices=MOUSE_BUTTONS,
            description=(
                "Minecraft mouse button to hold or release: `left` attacks, `right` uses, and "
                "`middle` picks the targeted item. The state starts with the next generated "
                "frame and persists until another `set_mouse_button_state` changes it or "
                "controls are cleared."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true to hold `button` on subsequent generated frames or false to release "
                "it."
            ),
        ),
    ) -> ActionChanged:
        if pressed and button not in self.state._pressed_mouse_buttons:
            self.state._pending_mouse_pulses = self.state._pending_mouse_pulses.union(
                (button,)
            )
        self.state._pressed_mouse_buttons = (
            self.state._pressed_mouse_buttons.union((button,))
            if pressed
            else self.state._pressed_mouse_buttons.difference((button,))
        )
        await self.send(self._state_update())
        return self._action_changed()

    @event(
        name="mouse_move",
        description=(
            "Queue normalized camera movement for the next generated frame. Valid while the "
            "session is active; calls before that frame accumulate and clamp each axis to "
            "[-1, 1], and movement is consumed after one frame. Emits `action_changed` and "
            "broadcasts `state_update`. Out-of-range values are rejected before state changes."
        ),
    )
    async def mouse_move(
        self,
        camera_x: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Relative horizontal camera movement in [-1, 1] to add to the next generated "
                "frame. Negative turns left, positive turns right, and repeated calls "
                "accumulate and clamp to that range."
            ),
        ),
        camera_y: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Relative vertical camera movement in [-1, 1] to add to the next generated "
                "frame. Negative and positive move pitch in opposite trained directions; "
                "repeated calls accumulate and clamp to that range."
            ),
        ),
    ) -> ActionChanged:
        self.state._camera_x = float(np.clip(self.state._camera_x + camera_x, -1, 1))
        self.state._camera_y = float(np.clip(self.state._camera_y + camera_y, -1, 1))
        await self.send(self._state_update())
        return self._action_changed()

    @event(
        name="release_controls",
        description=(
            "Release every held keyboard key and mouse button and discard queued camera "
            "movement. Valid while the session is active and takes effect on the next generated "
            "frame. Emits `action_changed` and broadcasts `state_update`."
        ),
    )
    async def release_controls(self) -> ActionChanged:
        self._clear_controls()
        await self.send(self._state_update())
        return self._action_changed()

    @event(
        name="set_image",
        description=(
            "Select an uploaded Minecraft image as the starting frame for a fresh rollout. "
            "Valid any time during a session; the image replaces previous starting context at the "
            "next generated frame and clears all controls. Emits `conditioning_changed` and "
            "broadcasts `state_update` on success, or `command_error` when the upload is "
            "mislabeled or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = _IMAGE_FIELD,
    ) -> ConditioningChanged:
        if not image.mime_type.startswith("image/"):
            raise CommandError(
                "unsupported_media", "set_image requires an image upload"
            )
        try:
            self.state._conditioning = await asyncio.to_thread(decode_image, image.data)
        except (OSError, ValueError) as error:
            raise CommandError("invalid_image", str(error)) from error
        return await self._select_conditioning("image", image.name)

    @event(
        name="set_video",
        description=(
            "Select consecutive frames from an uploaded video as the starting context for a "
            "fresh rollout. Valid any time during a session; the frames replace prior "
            "context at the next generated frame and clear all controls. Emits "
            "`conditioning_changed` and broadcasts `state_update` on success, or "
            "`command_error` when the upload is mislabeled, undecodable, or too short."
        ),
    )
    async def set_video(
        self,
        video: UploadedFile = _VIDEO_FIELD,
        offset: int = InputField(
            default=0,
            ge=0,
            description=(
                "Zero-based source frame at which the consecutive starting context begins. "
                "Applied when `set_video` queues the fresh rollout."
            ),
        ),
        prompt_frames: int = InputField(
            default=1,
            ge=1,
            le=32,
            description=(
                "Number of consecutive source frames to use as starting context, from 1 "
                "through 32. The upload must contain at least `offset + prompt_frames` frames."
            ),
        ),
    ) -> ConditioningChanged:
        if not video.mime_type.startswith("video/"):
            raise CommandError("unsupported_media", "set_video requires a video upload")
        try:
            self.state._conditioning = await asyncio.to_thread(
                decode_video, video.data, offset, prompt_frames
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise CommandError("invalid_video", str(error)) from error
        return await self._select_conditioning("video", video.name)

    @event(
        name="random_scene",
        description=(
            "Select the built-in Open-Oasis Minecraft sample as the starting frame for a fresh "
            "rollout. Valid any time during a session; it replaces uploaded context at the next "
            "generated frame and clears all controls. Emits `conditioning_changed` and "
            "broadcasts `state_update` on success."
        ),
    )
    async def random_scene(self) -> ConditioningChanged:
        if self._sample is None:
            raise CommandError(
                "sample_unavailable", "The built-in sample is unavailable"
            )
        self.state._conditioning = self._sample
        return await self._select_conditioning("built_in", "official_sample")

    @event(
        name="reset",
        description=(
            "Restart the selected starting context. Valid any time during a session; the reset "
            "takes effect at the next generated frame and clears all controls. Emits "
            "`rollout_reset` and broadcasts `state_update` on success; out-of-range seeds are "
            "rejected before state changes. If no context is selected, the model remains idle "
            "until `set_image`, `set_video`, or `random_scene` supplies one."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Random seed for the restarted rollout in [-1, 2147483647]. Use -1 to retain "
                "the current seed; a non-negative value replaces it when the reset is queued."
            ),
        ),
    ) -> RolloutReset:
        if seed >= 0:
            self.state._seed = seed
        self._queue_reset()
        await self.send(self._state_update())
        return RolloutReset(
            seed=self.state._seed, conditioning=self.state._conditioning_name
        )

    async def process_input(self) -> OpenOasisInput:
        if self.state._conditioning is None:
            raise ApplicationError("no starting context selected")
        fresh = self.state._world_id != self.state._applied_world_id
        action = self._build_action()
        return OpenOasisInput(
            world_id=self.state._world_id,
            conditioning=self.state._conditioning if fresh else None,
            seed=self.state._seed,
            action=action,
        )

    def generate(self, input: OpenOasisInput) -> OpenOasisResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> OpenOasisOutput | None:
        if outcome.error is not None:
            # Unexpected sampler failures cannot be repaired by restarting a world.
            raise outcome.error
        result: OpenOasisResult = outcome.result
        self.state._applied_world_id = result.world_id
        if result.index:
            changed = bool(self.state._camera_x or self.state._camera_y)
            self.state._pending_key_pulses = frozenset()
            self.state._pending_mouse_pulses = frozenset()
            self.state._camera_x = self.state._camera_y = 0.0
            if changed:
                await self.send(self._state_update())
        return OpenOasisOutput(main_video=result.frame)

    async def _select_conditioning(self, source: str, name: str) -> ConditioningChanged:
        self.state._conditioning_name = name
        self._queue_reset()
        await self.send(self._state_update())
        assert self.state._conditioning is not None
        return ConditioningChanged(
            source=source, selection=name, prompt_frames=len(self.state._conditioning)
        )

    def _queue_reset(self) -> None:
        self.output.flush()
        self.state._world_id += 1
        self._clear_controls()

    def _clear_controls(self) -> None:
        self.state._pressed_keys = frozenset()
        self.state._pressed_mouse_buttons = frozenset()
        self.state._pending_key_pulses = frozenset()
        self.state._pending_mouse_pulses = frozenset()
        self.state._camera_x = self.state._camera_y = 0.0

    def _build_action(self) -> np.ndarray:
        values = np.zeros(25, dtype=np.float32)
        for key in self.state._pressed_keys.union(self.state._pending_key_pulses):
            values[_ACTION_KEYS.index(_KEY_ACTION[key])] = 1
        for button in self.state._pressed_mouse_buttons.union(
            self.state._pending_mouse_pulses
        ):
            values[_ACTION_KEYS.index(_MOUSE_ACTION[button])] = 1
        values[_ACTION_KEYS.index("cameraX")] = self.state._camera_x
        values[_ACTION_KEYS.index("cameraY")] = self.state._camera_y
        return values

    def _action_changed(self) -> ActionChanged:
        return ActionChanged(
            pressed_keys=[key for key in KEYS if key in self.state._pressed_keys],
            pressed_mouse_buttons=[
                button
                for button in MOUSE_BUTTONS
                if button in self.state._pressed_mouse_buttons
            ],
            camera_x=self.state._camera_x,
            camera_y=self.state._camera_y,
        )

    def _state_update(self) -> StateUpdate:
        return StateUpdate(
            pressed_keys=[key for key in KEYS if key in self.state._pressed_keys],
            pressed_mouse_buttons=[
                button
                for button in MOUSE_BUTTONS
                if button in self.state._pressed_mouse_buttons
            ],
            camera_x=self.state._camera_x,
            camera_y=self.state._camera_y,
            seed=self.state._seed,
            conditioning=self.state._conditioning_name,
        )
