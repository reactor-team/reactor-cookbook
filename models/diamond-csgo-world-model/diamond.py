"""Serve the DIAMOND Counter-Strike world model through Reactor Runtime.

The adapter keeps DIAMOND's inference implementation intact and translates
Reactor commands into the keyboard and mouse action representation expected by
the upstream CSGO model. It produces one generated RGB frame on ``main_video``
for every world-model step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

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
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

from diamond_assets import (
    decode_spawn_image,
    load_adapter_dependencies,
    load_upstream_modules,
    read_config,
    resolve_upstream_eval,
    select_device,
    to_video_frame,
    upstream_root,
)
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
    PreparedScene,
    SceneChanged,
    StateUpdate,
)

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


def _weights_cache_path() -> Path:
    return get_weights_path() / "diamond-csgo-world-model" / "huggingface"


class Diamond(ReactorPipeline):
    """Stream one shared Counter-Strike world controlled by native game inputs."""

    fps = PLAYBACK_FPS
    buffer_size = PLAYBACK_BUFFER_FRAMES
    state: DiamondState

    def __init__(self) -> None:
        super().__init__()
        self._agent: Any = None
        self._world: Any = None
        self._torch: Any = None
        self._action_type: Any = None
        self._encode_action: Callable[..., Any] | None = None
        self._key_codes: dict[str, int] = {}
        self._spawn_dirs: tuple[Path, ...] = ()
        self._seed = 0
        self._rng = np.random.default_rng(self._seed)
        self._sequence_length = 0
        self._full_resolution = (150, 280)
        self._low_resolution = (30, 56)
        self._pending_scene: PreparedScene | None = None
        self._initial_observation: Any | None = None
        self._reset_requested = True
        self._controller = "human"
        self._replay_step = 0

    def load(self, config_path: Path | None) -> None:
        """Load the upstream DIAMOND model and its CSGO spawn states.

        Args:
            config_path: Path to the adapter YAML named by ``reactor.yaml``.
        """
        config = read_config(config_path)
        dependencies = load_adapter_dependencies()
        torch = dependencies["torch"]
        snapshot_download = dependencies["snapshot_download"]
        compose = dependencies["compose"]
        initialize_config_dir = dependencies["initialize_config_dir"]
        instantiate = dependencies["instantiate"]
        omega_conf = dependencies["omega_conf"]
        upstream = upstream_root()
        modules = load_upstream_modules(upstream)
        agent_type = modules["agent"].Agent
        world_type = modules["world"].WorldModelEnv
        action_module = modules["action"]
        pygame = modules["pygame"]

        omega_conf.register_new_resolver("eval", resolve_upstream_eval, replace=True)
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(upstream / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=[f"world_model_env={config.profile}"],
            )

        snapshot = Path(
            snapshot_download(
                repo_id=config.repo_id,
                revision=config.revision,
                allow_patterns="csgo/*",
                cache_dir=_weights_cache_path(),
            )
        )
        cfg.agent = omega_conf.load(snapshot / "csgo/config/agent/csgo.yaml")
        cfg.env = omega_conf.load(snapshot / "csgo/config/env/csgo.yaml")

        device = select_device(config.device, torch)
        torch.manual_seed(config.seed)
        self._agent = agent_type(
            instantiate(cfg.agent, num_actions=cfg.env.num_actions)
        )
        self._agent = self._agent.to(device).eval()
        self._agent.load(snapshot / "csgo/model/csgo.pt")

        sequence_length = cfg.agent.denoiser.inner_model.num_steps_conditioning
        if self._agent.upsampler is not None:
            sequence_length = max(
                sequence_length,
                cfg.agent.upsampler.inner_model.num_steps_conditioning,
            )
        world_config = instantiate(cfg.world_model_env, num_batches_to_preload=1)
        spawn_root = snapshot / "csgo/spawn"
        self._world = world_type(
            self._agent.denoiser,
            self._agent.upsampler,
            self._agent.rew_end_model,
            spawn_root,
            1,
            sequence_length,
            world_config,
            return_denoising_trajectory=False,
        )

        self._action_type = action_module.CSGOAction
        self._encode_action = action_module.encode_csgo_action
        self._torch = torch
        self._spawn_dirs = tuple(
            sorted(path for path in spawn_root.iterdir() if path.is_dir())
        )
        self._seed = config.seed
        self._rng = np.random.default_rng(self._seed)
        self._sequence_length = int(sequence_length)
        height, width = (int(value) for value in cfg.env.train.size)
        self._full_resolution = (height, width)
        upsampling_factor = int(cfg.agent.upsampler.upsampling_factor)
        self._low_resolution = (height // upsampling_factor, width // upsampling_factor)
        self._key_codes = {
            "w": pygame.K_w,
            "a": pygame.K_a,
            "s": pygame.K_s,
            "d": pygame.K_d,
            "space": pygame.K_SPACE,
            "ctrl": pygame.K_LCTRL,
            "shift": pygame.K_LSHIFT,
            "1": pygame.K_1,
            "2": pygame.K_2,
            "3": pygame.K_3,
            "r": pygame.K_r,
        }
        logger.info(
            "DIAMOND CSGO model ready",
            device=str(device),
            profile=config.profile,
            revision=config.revision,
        )

    @session_started
    def _start_session(self) -> None:
        """Initialize the world shared by every client in a new session."""
        self._pending_scene = None
        self._initial_observation = None
        self._controller = "human"
        self._replay_step = 0
        self._rng = np.random.default_rng(self._seed)
        self._reset_requested = True
        self._reset_world()

    @session_ended
    def _end_session(self) -> None:
        """Release session state while retaining loaded model resources."""
        self._pending_scene = None
        self._initial_observation = None
        self._reset_requested = False
        self._controller = "human"
        self._replay_step = 0
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
        self._pending_scene = self._prepare_dataset_scene(scene_dir)
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
        self._pending_scene = self._prepare_uploaded_scene(full_res, low_res)
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
                and self._pending_scene.next_act is None
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

    def inference(self) -> Iterator[DiamondOutput | None]:
        """Generate CSGO frames while applying the latest client controls."""
        if self._world is None or self._agent is None or self._encode_action is None:
            raise RuntimeError("DIAMOND model was not loaded")

        self.state.controller = self._controller
        while True:
            if self._reset_requested:
                self._reset_world()

            if self._initial_observation is not None:
                observation = self._initial_observation
                self._initial_observation = None
                yield DiamondOutput(main_video=to_video_frame(observation))
                continue

            action = self._next_action()
            observation, _reward, ended, truncated, _info = self._world.step(action)
            if (
                bool(ended.item())
                or bool(truncated.item())
                or self._replay_trajectory_finished()
            ):
                self._reset_requested = True
                self._clear_controls()
            yield DiamondOutput(main_video=to_video_frame(observation))

    def _reset_world(self) -> None:
        """Reset the shared world and retain its initial frame for emission."""
        if self._world is None:
            raise RuntimeError("DIAMOND model was not loaded")
        observation, _info = self._world.reset()
        if self._apply_pending_scene():
            observation = self._current_observation()
        self._initial_observation = observation
        self._replay_step = 0
        self._reset_requested = False
        self._clear_controls()

    def _current_observation(self) -> Any:
        """Return the latest full-resolution observation in the shared world."""
        buffer = self._world.obs_full_res_buffer
        if buffer is None:
            buffer = self._world.obs_buffer
        return buffer[:, -1]

    def _prepare_uploaded_scene(
        self,
        full_res: np.ndarray,
        low_res: np.ndarray,
    ) -> PreparedScene:
        """Build a device-ready repeated condition from one uploaded image."""
        if self._encode_action is None or self._agent is None:
            raise RuntimeError("DIAMOND model was not loaded")
        full_frames = np.repeat(full_res[None], self._sequence_length, axis=0)
        low_frames = np.repeat(low_res[None], self._sequence_length, axis=0)
        neutral = self._encode_action(
            self._action_type([], 0.0, 0.0, False, False),
            device=self._agent.device,
        )
        actions = neutral.reshape(1, 1, -1).repeat(1, self._sequence_length, 1)
        return PreparedScene(
            obs=self._observation_tensor(low_frames),
            obs_full_res=self._observation_tensor(full_frames),
            act=actions,
            next_act=None,
        )

    def _prepare_dataset_scene(self, scene_dir: Path) -> PreparedScene:
        """Load one official spawn with its full recorded action trajectory."""
        if self._torch is None or self._agent is None:
            raise RuntimeError("DIAMOND model was not loaded")
        device = self._agent.device
        return PreparedScene(
            obs=self._observation_tensor(np.load(scene_dir / "low_res.npy")),
            obs_full_res=self._observation_tensor(np.load(scene_dir / "full_res.npy")),
            act=self._torch.tensor(
                np.load(scene_dir / "act.npy"),
                dtype=self._torch.long,
                device=device,
            ).unsqueeze(0),
            next_act=self._torch.tensor(
                np.load(scene_dir / "next_act.npy"),
                dtype=self._torch.long,
                device=device,
            ),
        )

    def _observation_tensor(self, frames: np.ndarray) -> Any:
        """Normalize uint8 TCHW frames into a batched tensor on the model device."""
        if self._torch is None or self._agent is None:
            raise RuntimeError("DIAMOND model was not loaded")
        return (
            self._torch.tensor(frames, device=self._agent.device)
            .div(255)
            .mul(2)
            .sub(1)
            .unsqueeze(0)
        )

    def _queue_scene_reset(self) -> None:
        """Reset controls and request application of the queued scene."""
        self.output.flush()
        self._reset_requested = True
        self._replay_step = 0
        self._clear_controls()

    def _apply_pending_scene(self) -> bool:
        """Replace freshly reset buffers with a queued scene when present."""
        scene = self._pending_scene
        if scene is None:
            return False
        self._world.obs_buffer = scene.obs
        self._world.obs_full_res_buffer = scene.obs_full_res
        self._world.act_buffer = scene.act
        if scene.next_act is not None:
            self._world.next_act = scene.next_act
        self._pending_scene = None
        return True

    def _next_action(self) -> Any:
        """Return the next human or recorded replay action."""
        if self._controller == "replay":
            self._clear_controls()
            if self._replay_step == 0:
                action = self._world.act_buffer[0, -1].clone()
            else:
                action = self._world.next_act[self._replay_step - 1].clone()
            self._replay_step += 1
            return action

        assert self._encode_action is not None
        keys = [self._key_codes[key] for key in KEYS if key in self.state._pressed_keys]
        delta_x = self.state._delta_x
        delta_y = self.state._delta_y
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0
        action = self._action_type(
            keys,
            delta_x,
            delta_y,
            "left" in self.state._pressed_mouse_buttons,
            "right" in self.state._pressed_mouse_buttons,
        )
        return self._encode_action(action, device=self._agent.device)

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

    def _replay_trajectory_finished(self) -> bool:
        """Return whether the recorded spawn actions were all consumed."""
        return self._controller == "replay" and self._replay_step > int(
            self._world.next_act.size(0)
        )

    def _clear_controls(self) -> None:
        """Release held controls and discard pending mouse movement."""
        self.state._pressed_keys = frozenset()
        self.state._pressed_mouse_buttons = frozenset()
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0
