"""Serve OpenDreamer as an interactive Minecraft world model.

The adapter loads the public OpenDreamer tokenizer and EMA dynamics checkpoint,
seeds their KV caches from consecutive Minecraft frames with aligned VPT
actions, and turns Reactor input events into the action representation used
during training. It emits one RGB frame for every autoregressive model step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from opendreamer_types import (
    DEMO_CHOICES,
    ActionChanged,
    ConditioningChanged,
    OpenDreamerConfig,
    OpenDreamerOutput,
    OpenDreamerState,
    RolloutConditioning,
    RolloutReset,
    StateUpdate,
)
from opendreamer_utils import (
    decode_conditioning_image,
    ensure_demo_assets,
    load_dependencies,
    mesh_context,
    prepare_process_environment,
    read_conditioning_sequence,
    read_config,
    upstream_asset,
    upstream_root,
    verify_source_revision,
)
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorPipeline,
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
_KEY_TO_VPT_NAME = {
    "w": "key.keyboard.w",
    "a": "key.keyboard.a",
    "s": "key.keyboard.s",
    "d": "key.keyboard.d",
    "space": "key.keyboard.space",
    "shift": "key.keyboard.left.shift",
    "ctrl": "key.keyboard.left.control",
    "e": "key.keyboard.e",
    "q": "key.keyboard.q",
    "escape": "key.keyboard.escape",
    "f": "key.keyboard.f",
    "1": "key.keyboard.1",
    "2": "key.keyboard.2",
    "3": "key.keyboard.3",
    "4": "key.keyboard.4",
    "5": "key.keyboard.5",
    "6": "key.keyboard.6",
    "7": "key.keyboard.7",
    "8": "key.keyboard.8",
    "9": "key.keyboard.9",
    "f3": "key.keyboard.f3",
}
_BUTTON_TO_VPT_NAME = {
    "left": "mouse.0",
    "right": "mouse.1",
    "middle": "mouse.2",
}
_CAMERA_DELTA_MIN = -200.0
_CAMERA_DELTA_MAX = 200.0
FRAMES_PER_CHUNK = 1


class OpenDreamer(ReactorPipeline):
    """Stream an interactive Minecraft rollout from a dataset demo or uploaded image."""

    state: OpenDreamerState
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: OpenDreamerConfig | None = None
        self._deps: dict[str, Any] = {}
        self._mesh: Any = None
        self._tokenizer: Any = None
        self._dynamics: Any = None
        self._latent_shape: tuple[int, int, int, int] | None = None
        self._model_frame_shape: tuple[int, int, int] | None = None
        self._empty_dynamics_cache: Any = None
        self._empty_tokenizer_cache: Any = None
        self._next_frame_jit: Callable[..., Any] | None = None
        self._observe_frame_jit: Callable[..., Any] | None = None
        self._key_to_index: dict[str, int] = {}
        self._demos: dict[str, RolloutConditioning] = {}
        self._conditioning_source = "random"
        self._uploaded_conditioning: RolloutConditioning | None = None
        self._demo_rng = np.random.default_rng()

    def load(self, config_path: Path | None) -> None:
        """Load the public OpenDreamer source and checkpoint once.

        Args:
            config_path: Path to the model YAML named by ``reactor.yaml``.
        """
        config = read_config(config_path)
        source_root = upstream_root()
        verify_source_revision(source_root, config.source_revision)
        ensure_demo_assets(source_root, config.demos)
        prepare_process_environment(config)
        dependencies = load_dependencies(source_root)
        self._config = config
        self._deps = dependencies

        jax = dependencies["jax"]
        nnx = dependencies["nnx"]
        snapshot_download = dependencies["snapshot_download"]
        bundle_type = dependencies["bundle_type"]
        build_parallel = dependencies["build_parallel"]

        checkpoint_cache = get_weights_path() / "open-dreamer" / "huggingface"
        checkpoint_cache.mkdir(parents=True, exist_ok=True)
        checkpoint_path = snapshot_download(
            repo_id=config.checkpoint_repo_id,
            revision=config.checkpoint_revision,
            cache_dir=checkpoint_cache,
        )
        if jax.default_backend() == "cpu":
            raise RuntimeError("OpenDreamer requires a CUDA accelerator")

        mesh, _data_sharding, mesh_rules = build_parallel("data")
        self._mesh = mesh
        with mesh_context(jax, mesh):
            bundle = bundle_type.from_pretrained(
                checkpoint_path,
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(config.seed),
                model_names={"dynamics_ema", "tokenizer"},
            )
            if bundle.dynamics_ema is None or bundle.tokenizer is None:
                raise RuntimeError(
                    "checkpoint does not contain dynamics_ema and tokenizer"
                )
            self._dynamics = bundle.dynamics_ema
            self._tokenizer = bundle.tokenizer
            self._configure_inference(config)
            self._warm_inference(config)
            self._validate_action_space()

        assert self._model_frame_shape is not None
        self._demos = {
            demo.name: read_conditioning_sequence(
                upstream_asset(source_root, demo.video),
                upstream_asset(source_root, demo.actions),
                self._model_frame_shape,
                start_frame=demo.start_frame,
                required_frames=config.conditioning_frames,
                dependencies=self._deps,
            )
            for demo in config.demos
        }
        logger.info(
            "OpenDreamer model ready",
            backend=jax.default_backend(),
            devices=len(jax.devices()),
            checkpoint_revision=config.checkpoint_revision,
            source_revision=config.source_revision,
            demos=len(self._demos),
            conditioning_frames=config.conditioning_frames,
        )

    def _configure_inference(self, config: OpenDreamerConfig) -> None:
        """Create schedules, empty caches, and compiled inference callables."""
        jnp = self._deps["jnp"]
        nnx = self._deps["nnx"]
        schedule_type = self._deps["schedule_type"]
        next_frame = self._deps["next_frame"]
        tokenizer_caches_type = self._deps["tokenizer_caches_type"]
        normalize_latents = self._deps["normalize_latents"]

        dynamics_config = self._dynamics.cfg
        tokenizer_config = self._tokenizer.cfg
        schedule = schedule_type.init(
            num_steps=config.num_steps,
            k_max=dynamics_config.k_max,
            tau_ctx_target=config.tau_ctx_target,
        )

        n_latents = int(tokenizer_config.decoder.n_latents)
        d_bottleneck = int(tokenizer_config.encoder.d_bottleneck)
        height = int(tokenizer_config.decoder.H)
        width = int(tokenizer_config.decoder.W)
        self._latent_shape = (1, 1, n_latents, d_bottleneck)
        self._model_frame_shape = (height, width, 3)

        self._empty_dynamics_cache = self._dynamics.create_static_caches(
            batch_size=1,
            n_latents=n_latents,
            window_size=int(dynamics_config.context_length),
            n_agent=0,
            dtype=dynamics_config.dtype,
        )
        self._empty_tokenizer_cache = self._tokenizer.create_static_caches(
            batch_size=1,
            H=height,
            W=width,
            window_size=int(tokenizer_config.decoder.context_length),
            dtype=tokenizer_config.decoder.dtype,
        )

        def compiled_next_frame(
            tokenizer: Any,
            dynamics: Any,
            action: Any,
            latent_shape: tuple[int, int, int, int],
            dynamics_cache: Any,
            tokenizer_cache: Any,
            rng: Any,
        ) -> tuple[Any, Any, Any, Any]:
            frame, _hidden, new_dynamics_cache, decoder_cache, new_rng = next_frame(
                tokenizer,
                dynamics,
                schedule,
                action,
                latent_shape,
                dynamics_cache,
                tokenizer_cache.decoder,
                rng,
            )
            new_tokenizer_cache = tokenizer_caches_type(
                encoder=tokenizer_cache.encoder,
                decoder=decoder_cache,
            )
            return frame, new_dynamics_cache, new_tokenizer_cache, new_rng

        def compiled_observe_frame(
            tokenizer: Any,
            dynamics: Any,
            frame: Any,
            action: Any,
            dynamics_cache: Any,
            tokenizer_cache: Any,
        ) -> tuple[Any, Any]:
            video = jnp.asarray(frame, dtype=jnp.float32)[None, None, ...]
            latent, _, encoder_cache = tokenizer.encode(
                video,
                deterministic=True,
                caches=tokenizer_cache.encoder,
            )
            normalized = normalize_latents(
                latent,
                dynamics.cfg.latent_mean,
                dynamics.cfg.latent_std,
            )
            action_with_time = action[:, None, ...]
            step_indices = jnp.full((1, 1), schedule.emax, dtype=jnp.int32)
            tau_indices = jnp.full((1, 1), schedule.k_max, dtype=jnp.int32)
            _, (_, new_dynamics_cache) = dynamics(
                action_with_time,
                step_indices,
                tau_indices,
                normalized,
                deterministic=True,
                caches=dynamics_cache,
            )
            _, decoder_cache = tokenizer.decode(
                latent,
                caches=tokenizer_cache.decoder,
                deterministic=True,
            )
            new_tokenizer_cache = tokenizer_caches_type(
                encoder=encoder_cache,
                decoder=decoder_cache,
            )
            return new_dynamics_cache, new_tokenizer_cache

        self._next_frame_jit = nnx.jit(
            compiled_next_frame,
            static_argnames=("latent_shape",),
        )
        self._observe_frame_jit = nnx.jit(compiled_observe_frame)

    def _warm_inference(self, config: OpenDreamerConfig) -> None:
        """Compile the generation and conditioning paths before serving."""
        if config.warmup_steps == 0:
            return
        assert self._latent_shape is not None
        assert self._model_frame_shape is not None
        assert self._next_frame_jit is not None
        assert self._observe_frame_jit is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]
        rng = jax.random.PRNGKey(config.seed)
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache
        noop = self._noop_action()
        for _ in range(config.warmup_steps):
            rng, step_rng = jax.random.split(rng)
            frame, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                self._tokenizer,
                self._dynamics,
                noop,
                self._latent_shape,
                dynamics_cache,
                tokenizer_cache,
                step_rng,
            )
            jax.block_until_ready((frame, dynamics_cache, tokenizer_cache, rng))
        zero_frame = jnp.zeros(self._model_frame_shape, dtype=jnp.uint8)
        observed = self._observe_frame_jit(
            self._tokenizer,
            self._dynamics,
            zero_frame,
            noop,
            self._empty_dynamics_cache,
            self._empty_tokenizer_cache,
        )
        jax.block_until_ready(observed)

    @session_started
    def on_session_started(self) -> None:
        """Initialize one playable world before its first viewer connects."""
        if self._config is None:
            raise RuntimeError("OpenDreamer was not loaded")
        self.state._seed = self._config.seed
        self.state._reset_requested = True
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
            actions=self._repeated_noop_actions(self._config.conditioning_frames),
        )
        self._conditioning_source = "uploaded"
        self._queue_rollout_reset()
        await self._send_state_update()
        return ConditioningChanged(source="uploaded", selection=image.name)

    def _queue_rollout_reset(self) -> None:
        """Queue fresh autoregressive state and discard pending media."""
        self.output.flush()
        self.state._reset_requested = True
        self._clear_controls()

    def inference(self) -> Iterator[OpenDreamerOutput | None]:
        """Generate Minecraft frames from the current starting scene and player controls."""
        if (
            self._config is None
            or self._next_frame_jit is None
            or self._observe_frame_jit is None
        ):
            raise RuntimeError("OpenDreamer was not loaded")
        assert self._latent_shape is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]

        rng = jax.random.PRNGKey(self.state._seed)
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache
        conditioning: RolloutConditioning | None = None
        observation_index = 0
        self.state._reset_requested = True

        with mesh_context(jax, self._mesh):
            while True:
                if self.state._reset_requested:
                    self.state._reset_requested = False
                    rng = jax.random.PRNGKey(self.state._seed)
                    dynamics_cache = self._empty_dynamics_cache
                    tokenizer_cache = self._empty_tokenizer_cache
                    conditioning = self._select_conditioning()
                    observation_index = 0

                if conditioning is None:
                    yield None
                    continue

                if observation_index < conditioning.frames.shape[0]:
                    dynamics_cache, tokenizer_cache = self._observe_frame_jit(
                        self._tokenizer,
                        self._dynamics,
                        jnp.asarray(conditioning.frames[observation_index]),
                        self._action_at(conditioning.actions, observation_index),
                        dynamics_cache,
                        tokenizer_cache,
                    )
                    jax.block_until_ready((dynamics_cache, tokenizer_cache))
                    observation_index += 1
                    yield None
                    continue

                action = self._build_action()
                rng, step_rng = jax.random.split(rng)
                frame, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                    self._tokenizer,
                    self._dynamics,
                    action,
                    self._latent_shape,
                    dynamics_cache,
                    tokenizer_cache,
                    step_rng,
                )
                jax.block_until_ready(frame)
                self._consume_transient_controls()
                output = np.asarray(frame[0, 0])
                if output.dtype != np.uint8:
                    output = np.clip(output, 0, 255).astype(np.uint8)
                yield OpenDreamerOutput(main_video=np.ascontiguousarray(output))

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

    def _action_at(self, actions: Any, index: int) -> Any:
        """Remove the time dimension from one batched conditioning action."""
        action_type = self._deps["action_type"]

        def take(value: Any) -> Any:
            return None if value is None else value[:, index]

        return action_type(
            binary=take(actions.binary),
            categorical=take(actions.categorical),
            continuous=take(actions.continuous),
        )

    def _build_action(self) -> Any:
        """Build one upstream ``Actions`` value from the current Reactor state."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        mouse_to_categorical = self._deps["mouse_to_categorical"]
        binary = np.zeros((1, len(self._key_to_index)), dtype=np.int32)
        for key in self.state._pressed_keys:
            binary[0, self._key_to_index[_KEY_TO_VPT_NAME[key]]] = 1
        for button in self.state._pressed_mouse_buttons:
            binary[0, self._key_to_index[_BUTTON_TO_VPT_NAME[button]]] = 1
        if self.state._wheel_delta < 0:
            binary[0, self._key_to_index["mouse.wheel_neg"]] = 1
        elif self.state._wheel_delta > 0:
            binary[0, self._key_to_index["mouse.wheel_pos"]] = 1
        categorical = mouse_to_categorical(
            np.asarray([self.state._delta_x], dtype=np.float32),
            np.asarray([self.state._delta_y], dtype=np.float32),
        )
        return action_type(
            binary=jnp.asarray(binary, dtype=jnp.int32),
            categorical=jnp.asarray(categorical, dtype=jnp.int32),
            continuous=None,
        )

    def _noop_action(self) -> Any:
        """Return one neutral upstream ``Actions`` value."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        camera_classes = int(self._deps["camera_classes"])
        return action_type(
            binary=jnp.zeros((1, len(self._key_to_index) or 27), dtype=jnp.int32),
            categorical=jnp.full((1,), camera_classes // 2, dtype=jnp.int32),
            continuous=None,
        )

    def _repeated_noop_actions(self, frames: int) -> Any:
        """Return a batched neutral action history for static image conditioning."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        noop = self._noop_action()

        def repeat(value: Any) -> Any:
            return (
                None
                if value is None
                else jnp.repeat(value[:, None, ...], frames, axis=1)
            )

        return action_type(
            binary=repeat(noop.binary),
            categorical=repeat(noop.categorical),
            continuous=repeat(noop.continuous),
        )

    def _validate_action_space(self) -> None:
        """Verify the loaded source and checkpoint use the expected VPT action space."""
        source_mapping = dict(self._deps["key_to_index"])
        if len(source_mapping) != int(self._deps["binary_actions"]):
            raise RuntimeError(
                "OpenDreamer source has an inconsistent binary action space"
            )
        missing = set(_KEY_TO_VPT_NAME.values()) | set(_BUTTON_TO_VPT_NAME.values())
        missing |= {"mouse.wheel_neg", "mouse.wheel_pos", "unknown"}
        if missing.difference(source_mapping):
            raise RuntimeError("OpenDreamer source is missing required VPT actions")
        if int(self._dynamics.cfg.num_binary_actions) != len(source_mapping):
            raise RuntimeError(
                "checkpoint binary action count does not match the source"
            )
        if int(self._dynamics.cfg.categorical_action_dim) != int(
            self._deps["camera_classes"]
        ):
            raise RuntimeError(
                "checkpoint camera action count does not match the source"
            )
        self._key_to_index = source_mapping

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
