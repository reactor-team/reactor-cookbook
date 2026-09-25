"""Serve the DIAMOND Counter-Strike world model through Reactor Runtime.

The adapter keeps DIAMOND's inference implementation intact and translates
Reactor commands into the keyboard and mouse action representation expected by
the upstream CSGO model. It produces one generated RGB frame on ``main_video``
for every world-model step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from diamond_assets import (
    decode_spawn_image,
    upstream_root,
)
from diamond_model import DiamondAnchor, DiamondInput, DiamondModel, DiamondResult
from diamond_types import (
    CONTROLLERS,
    DELTA_X_MAX,
    DELTA_X_MIN,
    DELTA_Y_MAX,
    DELTA_Y_MIN,
    KEYS,
    MOUSE_BUTTONS,
    ActionChanged,
    DiamondOutput,
    DiamondState,
    SceneChanged,
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

PLAYBACK_FPS = 15
PLAYBACK_BUFFER_FRAMES = 4
_SPAWN_IMAGE_FIELD = InputField(
    moderate=True,
    description=(
        "Image uploaded through the Reactor upload protocol. Must contain decodable image "
        "bytes with an `image/*` MIME type; it is center-cropped to the native aspect ratio, "
        "resized, and applied when the fresh world starts at the next model-step boundary."
    ),
)


class Diamond(ReactorApp):
    """Stream one shared Counter-Strike world controlled by native game inputs."""

    fps = PLAYBACK_FPS
    buffer_size = PLAYBACK_BUFFER_FRAMES
    state: DiamondState

    def __init__(self) -> None:
        super().__init__()
        self._engine = DiamondModel()
        self._spawn_dirs: tuple[Path, ...] = ()
        self._seed = 0
        self._rng = np.random.default_rng(self._seed)
        self._full_resolution = (150, 280)
        self._low_resolution = (30, 56)
        self._pending_scene: DiamondAnchor | None = None
        self._controller = "human"
        self._world_id = 0
        self._applied_world_id: int | None = None
        self._session_active = False
        self._initial_world_id: int | None = None

    def load(self, config_path: Path | None) -> None:
        """Load the native model and retain its client-visible setup metadata."""
        setup = self._engine.load(config_path, get_weights_path(), upstream_root())
        self._spawn_dirs = setup.spawn_dirs
        self._seed = setup.seed
        self._rng = np.random.default_rng(self._seed)
        self._full_resolution = setup.full_resolution
        self._low_resolution = setup.low_resolution
        logger.info("DIAMOND CSGO model ready")

    @session_started
    def _start_session(self) -> None:
        """Request a native built-in spawn for the new shared session."""
        self._pending_scene = None
        self._controller = "human"
        self._rng = np.random.default_rng(self._seed)
        self._world_id += 1
        self._initial_world_id = self._world_id
        self._applied_world_id = None
        self._session_active = True
        self._clear_controls()

    @session_ended
    def _end_session(self) -> None:
        """Release session buffers while retaining loaded model resources."""
        try:
            self._engine.reset()
        finally:
            self._pending_scene = None
            self._applied_world_id = None
            self._session_active = False
            self._initial_world_id = None
            self._controller = "human"
            self._clear_controls()

    @connected
    async def _connected(self, client: ClientInfo) -> None:
        """Send the complete durable control state to one joining viewer."""
        await client.send(StateUpdate.from_state(self.state))

    @disconnected
    async def _disconnected(self) -> None:
        """Release held controls when a viewer leaves the live session."""
        self._clear_controls()
        await self._send_state_update()

    @event(
        name="reset",
        description=(
            "Queue a fresh built-in spawn for the next model-step boundary and release all "
            "held human controls. Available throughout an active session. Emits "
            "`action_changed` and broadcasts `state_update` on success."
        ),
    )
    async def reset(self) -> ActionChanged:
        """Request a reset and return the released native input state."""
        self._pending_scene = None
        self._queue_scene_reset()
        message = self._action_changed()
        await self._send_state_update()
        return message

    @event(
        name="random_scene",
        description=(
            "Select a random built-in spawn and queue it for the next model-step boundary, "
            "including recorded actions available to replay. Available throughout an active "
            "session. Emits `scene_changed` and broadcasts `state_update` on success, or "
            "`command_error` if the built-in scenes are unavailable."
        ),
    )
    async def random_scene(self) -> SceneChanged:
        """Queue a random official spawn and return its identifier."""
        if not self._spawn_dirs:
            raise CommandError(
                "scene_unavailable", "No built-in DIAMOND spawn scenes are available."
            )
        scene_index = int(self._rng.integers(len(self._spawn_dirs)))
        scene_dir = self._spawn_dirs[scene_index]
        self._pending_scene = DiamondAnchor(scene=scene_dir)
        self._queue_scene_reset()
        logger.info("built-in scene selected", scene=scene_dir.name)
        message = SceneChanged(source="built_in", scene=scene_dir.name)
        await self._send_state_update()
        return message

    @event(
        name="set_spawn_image",
        description=(
            "Start a fresh world from an uploaded image at the next model-step boundary, "
            "switch to human input, and release held controls. Available throughout an active "
            "session. Emits `scene_changed` and broadcasts `state_update` on success, or "
            "`command_error` if the upload is not a decodable image."
        ),
    )
    async def set_spawn_image(
        self,
        image: UploadedFile = _SPAWN_IMAGE_FIELD,
    ) -> SceneChanged:
        """Queue an uploaded image as a repeated neutral initial condition.

        Args:
            image: Uploaded CSGO image fetched by Reactor Runtime.

        Raises:
            CommandError: If the upload is not labeled as an image.
        """
        if not image.mime_type.startswith("image/"):
            raise CommandError(
                "invalid_image",
                f"expected an image upload, got {image.mime_type!r}",
            )
        full_res, low_res = decode_spawn_image(
            image.data,
            full_resolution=self._full_resolution,
            low_resolution=self._low_resolution,
        )
        self._pending_scene = DiamondAnchor(full_res=full_res, low_res=low_res)
        self._controller = "human"
        self.state.controller = self._controller
        self._queue_scene_reset()
        logger.info("uploaded scene selected", name=image.name, size=len(image.data))
        message = SceneChanged(source="uploaded", scene=image.name)
        await self._send_state_update()
        return message

    @event(
        name="set_controller",
        description=(
            "Select client-controlled input or the built-in scene's recorded replay. A change "
            "queues a fresh compatible world and releases held controls. Emits "
            "`action_changed` and broadcasts `state_update` on success, or `command_error` "
            "when `controller` is unsupported."
        ),
    )
    async def set_controller(
        self,
        controller: str = InputField(
            default="human",
            choices=CONTROLLERS,
            description=(
                'Action source used from the next model step: "human" applies client keyboard '
                'and mouse commands, while "replay" follows a built-in scene\'s recorded '
                "actions. Changing it queues a fresh world and releases held controls."
            ),
        ),
    ) -> ActionChanged:
        """Switch controller and return the resulting native input state."""
        if self._controller != controller:
            if (
                controller == "replay"
                and self._pending_scene is not None
                and self._pending_scene.full_res is not None
            ):
                self._pending_scene = None
            self._controller = controller
            self.state.controller = controller
            self._queue_scene_reset()
        message = self._action_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_key_state",
        description=(
            "Hold or release one native game key from the next generated frame until changed. "
            "Only human input uses the value; replay acknowledges and ignores it. Emits "
            "`action_changed` and, for human input, broadcasts `state_update`, or returns "
            "`command_error` when `key` is unsupported."
        ),
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=KEYS,
            description=(
                "Native key to change: `w`, `a`, `s`, and `d` move; `space` jumps; `ctrl` "
                "crouches; `shift` walks; `1`, `2`, and `3` select weapon slots; `r` reloads. "
                "Used only while `controller` is `human`."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "True holds `key` from the next generated frame until released; false releases "
                "it. Resetting or changing controller releases every key."
            ),
        ),
    ) -> ActionChanged:
        """Update one held keyboard key and return the resulting input state."""
        if self._controller == "human":
            if pressed:
                self.state._pressed_keys = self.state._pressed_keys.union((key,))
            else:
                self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        message = self._action_changed()
        if self._controller == "human":
            await self._send_state_update()
        return message

    @event(
        name="set_mouse_button_state",
        description=(
            "Hold or release one native mouse button from the next generated frame until "
            "changed. Only human input uses the value; replay acknowledges and ignores it. "
            "Emits `action_changed` and, for human input, broadcasts `state_update`, or returns "
            "`command_error` when `button` is unsupported."
        ),
    )
    async def set_mouse_button_state(
        self,
        button: str = InputField(
            default="left",
            choices=MOUSE_BUTTONS,
            description=(
                "Native mouse button to change: `left` fires and `right` uses the secondary "
                "action or scope. Used only while `controller` is `human`."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "True holds `button` from the next generated frame until released; false "
                "releases it. Resetting or changing controller releases every button."
            ),
        ),
    ) -> ActionChanged:
        """Update one held mouse button and return the resulting input state."""
        if self._controller == "human":
            if pressed:
                self.state._pressed_mouse_buttons = (
                    self.state._pressed_mouse_buttons.union((button,))
                )
            else:
                self.state._pressed_mouse_buttons = (
                    self.state._pressed_mouse_buttons.difference((button,))
                )
        message = self._action_changed()
        if self._controller == "human":
            await self._send_state_update()
        return message

    @event(
        name="mouse_move",
        description=(
            "Apply one relative camera movement to the next generated frame. Human input "
            "consumes it once; replay acknowledges and ignores it. Emits `action_changed`, or "
            "returns `command_error` when either delta is outside its supported range."
        ),
    )
    def mouse_move(
        self,
        delta_x: float = InputField(
            default=0.0,
            ge=DELTA_X_MIN,
            le=DELTA_X_MAX,
            description=(
                "Horizontal relative movement in native DIAMOND units, from -1000 to 1000. "
                "Applied once on the next human-controlled frame; zero leaves yaw unchanged."
            ),
        ),
        delta_y: float = InputField(
            default=0.0,
            ge=DELTA_Y_MIN,
            le=DELTA_Y_MAX,
            description=(
                "Vertical relative movement in native DIAMOND units, from -200 to 200. Applied "
                "once on the next human-controlled frame; zero leaves pitch unchanged."
            ),
        ),
    ) -> ActionChanged:
        """Store one raw mouse delta and return the resulting input state."""
        if self._controller == "human":
            self.state._delta_x = delta_x
            self.state._delta_y = delta_y
            return self._action_changed(delta_x=delta_x, delta_y=delta_y)
        return self._action_changed()

    async def process_input(self) -> DiamondInput:
        """Snapshot the requested spawn and next native keyboard/mouse action."""
        if not self._session_active:
            raise ApplicationError("No active DIAMOND session.")
        self.state.controller = self._controller
        anchor = None
        if self._world_id != self._applied_world_id:
            anchor = self._pending_scene or DiamondAnchor()
        return DiamondInput(
            world_id=self._world_id,
            anchor=anchor,
            controller=self._controller,
            pressed_keys=self.state._pressed_keys,
            pressed_mouse_buttons=self.state._pressed_mouse_buttons,
            delta_x=self.state._delta_x,
            delta_y=self.state._delta_y,
        )

    def generate(self, input: DiamondInput) -> DiamondResult:
        """Forward one prepared native step to the independent model."""
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> DiamondOutput:
        """Apply completed-step input effects and publish its frame."""
        if outcome.error is not None:
            # Unexpected inference failures cannot be repaired by discarding the world.
            raise outcome.error
        result: DiamondResult = outcome.result
        self._applied_world_id = result.world_id
        self._pending_scene = None
        initial_session_frame = (
            result.index == 0 and result.world_id == self._initial_world_id
        )
        if result.clear_controls and not initial_session_frame:
            self._clear_controls()
        elif result.consumed_mouse:
            self.state._delta_x = 0.0
            self.state._delta_y = 0.0
        if result.terminal:
            # Preserve the terminal frame; the next step emits a fresh native spawn.
            self._world_id += 1
        return DiamondOutput(main_video=result.frame)

    def _queue_scene_reset(self) -> None:
        """Cut pending playout and request a fresh world identity."""
        self.output.flush()
        self._world_id += 1
        self._clear_controls()

    def _action_changed(
        self, *, delta_x: float = 0.0, delta_y: float = 0.0
    ) -> ActionChanged:
        """Describe the current native input state for an event response."""
        return ActionChanged(
            controller=self._controller,
            pressed_keys=[key for key in KEYS if key in self.state._pressed_keys],
            pressed_mouse_buttons=[
                button
                for button in MOUSE_BUTTONS
                if button in self.state._pressed_mouse_buttons
            ],
            delta_x=delta_x,
            delta_y=delta_y,
        )

    async def _send_state_update(self) -> None:
        """Broadcast the complete durable control state."""
        await self.send(StateUpdate.from_state(self.state))

    def _clear_controls(self) -> None:
        """Release held controls and discard pending mouse movement."""
        self.state._pressed_keys = frozenset()
        self.state._pressed_mouse_buttons = frozenset()
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0
