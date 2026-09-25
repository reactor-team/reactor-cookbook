"""Own DIAMOND weights and native world state independently of Reactor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from diamond_assets import (
    load_adapter_dependencies,
    load_upstream_modules,
    read_config,
    resolve_upstream_eval,
    select_device,
    to_video_frame,
)

KEYS = ("w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r")


@dataclass(frozen=True)
class DiamondSetup:
    """Report loaded spawn choices and native image sizes to the application."""

    spawn_dirs: tuple[Path, ...]
    seed: int
    full_resolution: tuple[int, int]
    low_resolution: tuple[int, int]


@dataclass(frozen=True)
class DiamondAnchor:
    """Request a native built-in spawn, a named scene, or decoded upload."""

    scene: Path | None = None
    full_res: np.ndarray | None = None
    low_res: np.ndarray | None = None


@dataclass(frozen=True)
class DiamondInput:
    """Snapshot the requested world and one frame's native human inputs."""

    world_id: int
    anchor: DiamondAnchor | None
    controller: str
    pressed_keys: frozenset[str]
    pressed_mouse_buttons: frozenset[str]
    delta_x: float
    delta_y: float


@dataclass(frozen=True)
class DiamondResult:
    """Report successful progress and native terminal/control effects."""

    frame: np.ndarray
    world_id: int
    index: int
    terminal: bool
    clear_controls: bool
    consumed_mouse: bool


@dataclass(frozen=True)
class PreparedScene:
    """Hold device tensors internal to a native world initialization."""

    obs: Any
    obs_full_res: Any
    act: Any
    next_act: Any | None


class NoAnchor(Exception):
    """A requested new world needs an explicit spawn choice."""


class WorldComplete(Exception):
    """The native terminal world needs an application-requested new identity."""


class DiamondModel:
    """Load weights once and own the native autoregressive buffers."""

    def __init__(self) -> None:
        self._agent: Any = None
        self._world: Any = None
        self._torch: Any = None
        self._action_type: Any = None
        self._encode_action: Callable[..., Any] | None = None
        self._key_codes: dict[str, int] = {}
        self._spawn_dirs: tuple[Path, ...] = ()
        self._seed = 0
        self._sequence_length = 0
        self._full_resolution = (150, 280)
        self._low_resolution = (30, 56)
        self._world_id: int | None = None
        self._index = 0
        self._replay_step = 0
        self._terminal = False

    def load(
        self, config_path: Path | None, weights_root: Path, upstream: Path
    ) -> DiamondSetup:
        """Load native weights and return immutable application setup metadata."""
        config = read_config(config_path)
        dependencies = load_adapter_dependencies()
        torch = dependencies["torch"]
        snapshot_download = dependencies["snapshot_download"]
        compose = dependencies["compose"]
        initialize_config_dir = dependencies["initialize_config_dir"]
        instantiate = dependencies["instantiate"]
        omega_conf = dependencies["omega_conf"]
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
                cache_dir=weights_root / "diamond-csgo-world-model" / "huggingface",
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
        return DiamondSetup(
            self._spawn_dirs, config.seed, self._full_resolution, self._low_resolution
        )

    def generate(self, input: DiamondInput) -> DiamondResult:
        """Emit the spawn frame or advance one unchanged native world step."""
        if self._world is None or self._agent is None or self._encode_action is None:
            raise RuntimeError("DIAMOND model was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            observation, _info = self._world.reset()
            anchor = input.anchor
            scene = None
            if anchor.scene is not None:
                scene = self._prepare_dataset_scene(anchor.scene)
            elif anchor.full_res is not None:
                if anchor.low_res is None:
                    raise NoAnchor("Uploaded spawn needs both native resolutions")
                scene = self._prepare_uploaded_scene(anchor.full_res, anchor.low_res)
            if scene is not None:
                self._world.obs_buffer = scene.obs
                self._world.obs_full_res_buffer = scene.obs_full_res
                self._world.act_buffer = scene.act
                if scene.next_act is not None:
                    self._world.next_act = scene.next_act
                observation = self._current_observation()
            frame = to_video_frame(observation)
            self._world_id = input.world_id
            self._index = 0
            self._replay_step = 0
            self._terminal = False
            return DiamondResult(frame, input.world_id, 0, False, True, False)
        if self._terminal:
            raise WorldComplete(input.world_id)
        action = self._next_action(input)
        observation, _reward, ended, truncated, _info = self._world.step(action)
        next_replay_step = self._replay_step + (input.controller == "replay")
        terminal = (
            bool(ended.item())
            or bool(truncated.item())
            or (
                input.controller == "replay"
                and next_replay_step > int(self._world.next_act.size(0))
            )
        )
        frame = to_video_frame(observation)
        self._replay_step = next_replay_step
        self._index += 1
        self._terminal = terminal
        return DiamondResult(
            frame,
            input.world_id,
            self._index,
            terminal,
            terminal or input.controller == "replay",
            True,
        )

    def reset(self) -> None:
        """Drop session buffers while retaining loaded weights and spawn data."""
        if self._world is not None:
            self._world.obs_buffer = None
            self._world.obs_full_res_buffer = None
            self._world.act_buffer = None
            self._world.next_act = None
            self._world.ep_len = None
            self._world.hx_rew_end = None
            self._world.cx_rew_end = None
        self._world_id = None
        self._index = 0
        self._replay_step = 0
        self._terminal = False

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

    def _next_action(self, input: DiamondInput) -> Any:
        """Return the next human or recorded replay action."""
        if input.controller == "replay":
            if self._replay_step == 0:
                action = self._world.act_buffer[0, -1].clone()
            else:
                action = self._world.next_act[self._replay_step - 1].clone()
            return action

        assert self._encode_action is not None
        keys = [self._key_codes[key] for key in KEYS if key in input.pressed_keys]
        action = self._action_type(
            keys,
            input.delta_x,
            input.delta_y,
            "left" in input.pressed_mouse_buttons,
            "right" in input.pressed_mouse_buttons,
        )
        return self._encode_action(action, device=self._agent.device)
