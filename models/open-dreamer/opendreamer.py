"""Serve OpenDreamer as an interactive Minecraft world model.

The adapter loads the public OpenDreamer tokenizer and EMA dynamics checkpoint,
seeds their KV caches from consecutive Minecraft frames with aligned VPT
actions, and turns Reactor input events into the action representation used
during training. It emits one RGB frame for every autoregressive model step.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from opendreamer_model import (
    OpenDreamerAnchor,
    OpenDreamerModel,
    OpenDreamerResult,
    OpenDreamerStepState,
)
from opendreamer_types import (
    ActionChanged,
    ConditioningChanged,
    OpenDreamerOutput,
    OpenDreamerState,
    RolloutReset,
    StateUpdate,
)
from opendreamer_utils import (
    DEMO_CHOICES,
    OpenDreamerConfig,
    RolloutConditioning,
    decode_conditioning_image,
    prepare_process_environment,
    read_config,
    upstream_root,
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
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger
from reactor_runtime.paths import get_weights_path

logger = get_logger(__name__)

_KEYS = [
    "w",
    "a",
    "s",
    "d",
    "space",
    "shift",
    "ctrl",
    "e",
    "q",
    "escape",
    "f",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "f3",
]
_MOUSE_BUTTONS = ["left", "right", "middle"]
_CAMERA_DELTA_MIN = -200.0
_CAMERA_DELTA_MAX = 200.0
FRAMES_PER_CHUNK = 1


class OpenDreamer(ReactorApp):
    """Stream an interactive Minecraft rollout from a dataset demo or uploaded image."""

    state: OpenDreamerState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._model = OpenDreamerModel()
        self._config: OpenDreamerConfig | None = None
        self._model_frame_shape: tuple[int, int, int] | None = None
        self._demos: dict[str, RolloutConditioning] = {}
        self._conditioning_source = "random"
        self._uploaded_conditioning: RolloutConditioning | None = None
        self._demo_rng = np.random.default_rng()
        self._world_id = uuid4().hex
        self._last_result: OpenDreamerResult | None = None

    def load(self, config_path: Path | None) -> None:
        """Load weights and obtain CPU-only checkpoint and demo metadata."""
        self._config = read_config(config_path)
        prepare_process_environment(self._config)
        setup = self._model.load(self._config, get_weights_path(), upstream_root())
        self._model_frame_shape = setup.frame_shape
        self._demos = setup.demos

    @session_started
    def on_session_started(self) -> None:
        """Initialize one playable world before its first viewer connects."""
        if self._config is None:
            raise RuntimeError("OpenDreamer was not loaded")
        self._world_id = uuid4().hex
        self._last_result = None
        self.state._seed = self._config.seed
        self._conditioning_source = self._random_demo_name()
        self._uploaded_conditioning = None
        self._clear_controls()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the current shared world state to one joining viewer."""
        await self._send_state_update(client)

    @session_ended
    def on_session_ended(self) -> None:
        """Release controls and uploaded conditioning owned by the completed world."""
        self._clear_controls()
        self._uploaded_conditioning = None
        self._conditioning_source = "random"
        self._model.reset()
        self._last_result = None

    @event(
        name="set_key_state",
        description=(
            "Hold or release one Minecraft keyboard key for subsequent frames. Valid while the "
            "session is active. Emits `action_changed` and "
            "`state_update`. Unsupported values are rejected before state changes."
        ),
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=_KEYS,
            description=(
                "Minecraft keyboard key to hold or release. The key state starts "
                "with the next generated frame and persists until another `set_key_state` "
                "changes it or controls are cleared."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true to hold `key` on subsequent generated frames or false to release it."
            ),
        ),
    ) -> ActionChanged:
        """Update one held keyboard key and report the controls now in effect."""
        if pressed:
            self.state._pressed_keys = self.state._pressed_keys.union((key,))
        else:
            self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        await self._send_state_update()
        return self._action_changed(control="set_key_state")

    @event(
        name="set_mouse_button_state",
        description=(
            "Hold or release one Minecraft mouse button for subsequent frames. Valid while the "
            "session is active. Emits `action_changed` and "
            "`state_update`. Unsupported values are rejected before state changes."
        ),
    )
    async def set_mouse_button_state(
        self,
        button: str = InputField(
            default="left",
            choices=_MOUSE_BUTTONS,
            description=(
                "Minecraft mouse button to hold or release. The button state "
                "starts with the next generated frame and persists until another "
                "`set_mouse_button_state` changes it or controls are cleared."
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
        """Update one held mouse button and report the controls now in effect."""
        if pressed:
            self.state._pressed_mouse_buttons = self.state._pressed_mouse_buttons.union(
                (button,)
            )
        else:
            self.state._pressed_mouse_buttons = (
                self.state._pressed_mouse_buttons.difference((button,))
            )
        await self._send_state_update()
        return self._action_changed(control="set_mouse_button_state")

    @event(
        name="mouse_move",
        description=(
            "Queue relative camera movement for the next generated frame. Valid while the "
            "session is active; calls before that frame accumulate within [-200, 200] on each "
            "axis, and movement is consumed after one frame. Emits `action_changed` and "
            "`state_update`. Out-of-range values are rejected before state "
            "changes."
        ),
    )
    async def mouse_move(
        self,
        delta_x: float = InputField(
            default=0.0,
            ge=_CAMERA_DELTA_MIN,
            le=_CAMERA_DELTA_MAX,
            description=(
                "Relative horizontal mouse movement in [-200, 200] to add to the next generated "
                "frame. Multiple calls accumulate and clamp to that range."
            ),
        ),
        delta_y: float = InputField(
            default=0.0,
            ge=_CAMERA_DELTA_MIN,
            le=_CAMERA_DELTA_MAX,
            description=(
                "Relative vertical mouse movement in [-200, 200] to add to the next generated "
                "frame. Multiple calls accumulate and clamp to that range."
            ),
        ),
    ) -> ActionChanged:
        """Queue camera motion and report the movement accepted for the next frame."""
        self.state._delta_x = float(
            np.clip(self.state._delta_x + delta_x, _CAMERA_DELTA_MIN, _CAMERA_DELTA_MAX)
        )
        self.state._delta_y = float(
            np.clip(self.state._delta_y + delta_y, _CAMERA_DELTA_MIN, _CAMERA_DELTA_MAX)
        )
        await self._send_state_update()
        return self._action_changed(
            control="mouse_move",
            delta_x=delta_x,
            delta_y=delta_y,
        )

    @event(
        name="mouse_wheel",
        description=(
            "Queue a Minecraft hotbar scroll for the next generated frame. Valid while the "
            "session is active; calls before that frame accumulate and only the resulting "
            "direction is applied. Emits `action_changed` and "
            "`state_update`. Values outside [-1, 1] are rejected before state changes."
        ),
    )
    async def mouse_wheel(
        self,
        delta: int = InputField(
            default=0,
            ge=-1,
            le=1,
            description=(
                "Hotbar movement for the next generated frame: -1 scrolls down, 1 scrolls up, "
                "and 0 leaves the selection unchanged."
            ),
        ),
    ) -> ActionChanged:
        """Queue a hotbar scroll and report the movement accepted for the next frame."""
        self.state._wheel_delta += delta
        await self._send_state_update()
        return self._action_changed(control="mouse_wheel", wheel_delta=delta)

    @event(
        name="reset",
        description=(
            "Restart the selected starting scene from its conditioning frames. Valid any time "
            "during a session; the reset takes effect at the next inference boundary "
            "and clears all controls. Emits `rollout_reset` and "
            "`state_update` on success; out-of-range seeds are rejected before state changes."
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
        """Restart the rollout and report the seed and starting scene it will use."""
        if seed >= 0:
            self.state._seed = seed
        self._queue_rollout_reset()
        await self._send_state_update()
        return RolloutReset(
            seed=self.state._seed,
            conditioning=self._conditioning_source,
        )

    @event(
        name="set_demo",
        description=(
            "Select a configured dataset demo as the next starting scene. Valid any time during "
            "a session; the selection resets the rollout at the next inference boundary and "
            "clears all controls. Emits `conditioning_changed` and `state_update` on success, "
            "or `command_error` with `demo_unavailable` when the demo is not configured."
        ),
    )
    async def set_demo(
        self,
        demo: str = InputField(
            default="demo_1",
            choices=DEMO_CHOICES,
            description=(
                "Configured dataset demo to use for the next rollout. The selection takes "
                "effect at the next inference boundary and replaces an uploaded image or the "
                "previous demo."
            ),
        ),
    ) -> ConditioningChanged:
        """Select a dataset demo and report the starting scene now in effect."""
        if demo not in self._demos:
            raise CommandError("demo_unavailable", f"{demo} is not configured.")
        self._conditioning_source = demo
        self._queue_rollout_reset()
        await self._send_state_update()
        return ConditioningChanged(source="demo", selection=demo)

    @event(
        name="random_demo",
        description=(
            "Select a random configured dataset demo as the next starting scene. Valid any time "
            "during a session; the selection resets the rollout at the next inference boundary "
            "and clears all controls. Emits `conditioning_changed` and `state_update` on "
            "success, or `command_error` with `demo_unavailable` when no demos are configured."
        ),
    )
    async def random_demo(self) -> ConditioningChanged:
        """Select a random dataset demo and report the chosen starting scene."""
        demo = self._random_demo_name()
        self._conditioning_source = demo
        self._queue_rollout_reset()
        await self._send_state_update()
        logger.info("selected random conditioning demo", demo=demo)
        return ConditioningChanged(source="demo", selection=demo)

    @event(
        name="set_conditioning_image",
        description=(
            "Select an uploaded Minecraft screenshot as the next starting scene. Valid after "
            "the model has loaded; the image is prepared immediately, then resets the rollout "
            "at the next inference boundary and clears all controls. Emits "
            "`conditioning_changed` and `state_update` on success, or `command_error` with "
            "`model_not_ready`, `unsupported_media`, or `invalid_image` when validation fails."
        ),
    )
    async def set_conditioning_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008  # Reactor reads this schema metadata.
            moderate=True,
            description=(
                "Minecraft screenshot uploaded through Reactor's file-upload flow. The file "
                "must have an `image/*` media type and decode successfully; it is "
                "orientation-corrected, center-cropped, and resized to the model resolution "
                "before the next rollout starts."
            ),
        ),
    ) -> ConditioningChanged:
        """Use one image as the starting scene and report the accepted filename."""
        if self._model_frame_shape is None or self._config is None:
            raise CommandError("model_not_ready", "OpenDreamer is still loading.")
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} must be an image.")
        try:
            frame = decode_conditioning_image(image.data, self._model_frame_shape)
        except (ValueError, OSError) as error:
            raise CommandError("invalid_image", str(error)) from error
        self._uploaded_conditioning = RolloutConditioning(
            frames=np.repeat(
                frame[None],
                self._config.conditioning_frames,
                axis=0,
            ).copy(),
            actions=None,
        )
        self._conditioning_source = "uploaded"
        self._queue_rollout_reset()
        await self._send_state_update()
        return ConditioningChanged(source="uploaded", selection=image.name)

    def _queue_rollout_reset(self) -> None:
        """Queue fresh autoregressive state and discard pending media."""
        self.output.flush()
        self._world_id = uuid4().hex
        self._clear_controls()

    async def process_input(self) -> OpenDreamerStepState:
        """Snapshot controls and supply an anchor until the model acknowledges it.

        The anchor rides on the input only while the model has not reported the
        current world id back on a result, so it crosses once per fresh world.
        """
        conditioning = self._select_conditioning()
        if conditioning is None:
            raise ApplicationError("Select conditioning before generating.")
        anchor = (
            OpenDreamerAnchor(conditioning, self.state._seed)
            if self._world_id != self.state._applied_world_id
            else None
        )
        return OpenDreamerStepState(
            world_id=self._world_id,
            anchor=anchor,
            pressed_keys=self.state._pressed_keys,
            pressed_mouse_buttons=self.state._pressed_mouse_buttons,
            delta_x=self.state._delta_x,
            delta_y=self.state._delta_y,
            wheel_delta=self.state._wheel_delta,
        )

    def generate(self, input: OpenDreamerStepState) -> OpenDreamerResult:
        """Forward one immutable input to the isolated model."""
        return self._model.generate(input)

    async def process_output(self, outcome: StepOutcome) -> OpenDreamerOutput | None:
        """Acknowledge model progress and consume deltas only for generated frames."""
        if outcome.error is not None:
            # Keep the pending anchor and controls intact when inference fails.
            raise outcome.error
        result = outcome.result
        if result.world_id != self._world_id:
            return None
        self._last_result = result
        self.state._applied_world_id = result.world_id
        if result.frame is None:
            return None
        self._consume_transient_controls()
        return OpenDreamerOutput(main_video=result.frame)

    def _select_conditioning(self) -> RolloutConditioning | None:
        """Return the uploaded sequence or resolve the active configured demo."""
        if self._conditioning_source == "uploaded":
            return self._uploaded_conditioning
        if not self._demos:
            return None
        name = self._conditioning_source
        if name == "random":
            name = self._random_demo_name()
            self._conditioning_source = name
            logger.info("selected random conditioning demo", demo=name)
        return self._demos.get(name)

    def _random_demo_name(self) -> str:
        """Return one configured demo name from the session RNG."""
        if not self._demos:
            raise CommandError(
                "demo_unavailable", "No conditioning demos are configured."
            )
        names = tuple(self._demos)
        return names[int(self._demo_rng.integers(len(names)))]

    def _action_changed(
        self,
        *,
        control: str,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        wheel_delta: int = 0,
    ) -> ActionChanged:
        """Describe the current native input state for an event response."""
        return ActionChanged(
            control=control,
            pressed_keys=[key for key in _KEYS if key in self.state._pressed_keys],
            pressed_mouse_buttons=[
                button
                for button in _MOUSE_BUTTONS
                if button in self.state._pressed_mouse_buttons
            ],
            delta_x=delta_x,
            delta_y=delta_y,
            wheel_delta=wheel_delta,
        )

    async def _send_state_update(self, client: ClientInfo | None = None) -> None:
        """Send a complete client-facing snapshot of the shared world state."""
        message = StateUpdate.from_state(
            self.state,
            conditioning=self._conditioning_source,
        )
        if client is not None:
            await client.send(message)
            return
        await self.send(message)

    def _consume_transient_controls(self) -> None:
        """Consume camera and wheel deltas after one generated frame."""
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0
        self.state._wheel_delta = 0

    def _clear_controls(self) -> None:
        """Release held controls and discard transient input."""
        self.state._pressed_keys = frozenset()
        self.state._pressed_mouse_buttons = frozenset()
        self._consume_transient_controls()
