"""Own OpenDreamer's native JAX inference, RNG and incremental caches."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from opendreamer_utils import (
    OpenDreamerConfig,
    RolloutConditioning,
    ensure_demo_assets,
    load_dependencies,
    mesh_context,
    read_conditioning_sequence,
    upstream_asset,
    verify_source_revision,
)

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class OpenDreamerAnchor:
    """Initialize a new world with CPU conditioning and its sampling seed."""

    conditioning: RolloutConditioning
    seed: int


@dataclass(frozen=True)
class OpenDreamerStepState:
    """Snapshot controls without retaining application or runtime objects."""

    world_id: str
    anchor: OpenDreamerAnchor | None
    pressed_keys: frozenset[str]
    pressed_mouse_buttons: frozenset[str]
    delta_x: float
    delta_y: float
    wheel_delta: int


@dataclass(frozen=True)
class OpenDreamerResult:
    """Report model-owned conditioning progress and generated frames."""

    frame: np.ndarray | None
    world_id: str
    observation_index: int
    generated_frames: int


@dataclass(frozen=True)
class OpenDreamerSetup:
    """Return CPU-only checkpoint metadata and prepared demo sequences."""

    frame_shape: tuple[int, int, int]
    demos: dict[str, RolloutConditioning]


class MissingConditioningError(ValueError):
    """A new model world requires a conditioning anchor."""


class OpenDreamerModel:
    """Run the upstream one-frame autoregressive loop independently of Reactor."""

    def __init__(self) -> None:
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
        self._rollout_rng: Any = None
        self._dynamics_cache: Any = None
        self._tokenizer_cache: Any = None
        self._conditioning: RolloutConditioning | None = None
        self._observation_index = 0
        self._world_id: str | None = None
        self._generated_frames = 0

    def load(
        self, config: OpenDreamerConfig, weights_root: Path, source_root: Path
    ) -> OpenDreamerSetup:
        """Load the public OpenDreamer source and checkpoint once.

        Args:
            config: Validated native inference settings.
            weights_root: Checkpoint cache directory provided by the caller.
            source_root: Pinned upstream checkout provided by the caller.
        """
        verify_source_revision(source_root, config.source_revision)
        ensure_demo_assets(source_root, config.demos)
        dependencies = load_dependencies(source_root)
        self._config = config
        self._deps = dependencies

        jax = dependencies["jax"]
        nnx = dependencies["nnx"]
        snapshot_download = dependencies["snapshot_download"]
        bundle_type = dependencies["bundle_type"]
        build_parallel = dependencies["build_parallel"]

        checkpoint_cache = weights_root / "open-dreamer" / "huggingface"
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
        demos = {
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
            "OpenDreamer model ready: backend=%s, devices=%s",
            jax.default_backend(),
            len(jax.devices()),
        )
        return OpenDreamerSetup(self._model_frame_shape, demos)

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

    def generate(self, input: OpenDreamerStepState) -> OpenDreamerResult:
        """Observe one conditioning frame or generate one native Minecraft frame."""
        if (
            self._config is None
            or self._next_frame_jit is None
            or self._observe_frame_jit is None
        ):
            raise RuntimeError("OpenDreamer was not loaded")
        assert self._latent_shape is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]

        with mesh_context(jax, self._mesh):
            if input.world_id != self._world_id:
                if input.anchor is None:
                    raise MissingConditioningError("A new world requires conditioning.")
                self.reset()
                self._world_id = input.world_id
                self._rollout_rng = jax.random.PRNGKey(input.anchor.seed)
                self._dynamics_cache = self._empty_dynamics_cache
                self._tokenizer_cache = self._empty_tokenizer_cache
                self._conditioning = input.anchor.conditioning
                self._observation_index = 0
            conditioning = self._conditioning
            if conditioning is None:
                raise RuntimeError("OpenDreamer has no rollout conditioning")
            if self._observation_index < conditioning.frames.shape[0]:
                self._dynamics_cache, self._tokenizer_cache = self._observe_frame_jit(
                    self._tokenizer,
                    self._dynamics,
                    jnp.asarray(conditioning.frames[self._observation_index]),
                    self._action_at(conditioning.actions, self._observation_index)
                    if conditioning.actions is not None
                    else self._noop_action(),
                    self._dynamics_cache,
                    self._tokenizer_cache,
                )
                jax.block_until_ready((self._dynamics_cache, self._tokenizer_cache))
                self._observation_index += 1
                return OpenDreamerResult(
                    None,
                    self._world_id,
                    self._observation_index,
                    self._generated_frames,
                )
            action = self._build_action(input)
            self._rollout_rng, step_rng = jax.random.split(self._rollout_rng)
            frame, self._dynamics_cache, self._tokenizer_cache, self._rollout_rng = (
                self._next_frame_jit(
                    self._tokenizer,
                    self._dynamics,
                    action,
                    self._latent_shape,
                    self._dynamics_cache,
                    self._tokenizer_cache,
                    step_rng,
                )
            )
            jax.block_until_ready(frame)
            output = np.asarray(frame[0, 0])
            if output.dtype != np.uint8:
                output = np.clip(output, 0, 255).astype(np.uint8)
            if output.ndim != 3 or output.shape[-1] != 3:
                raise ValueError(f"Expected an RGB frame, got {output.shape}")
            self._generated_frames += 1
            return OpenDreamerResult(
                np.ascontiguousarray(output),
                self._world_id,
                self._observation_index,
                self._generated_frames,
            )

    def reset(self) -> None:
        """Drop world caches and RNG while retaining loaded weights and JITs."""
        self._rollout_rng = None
        self._dynamics_cache = None
        self._tokenizer_cache = None
        self._conditioning = None
        self._observation_index = 0
        self._generated_frames = 0
        self._world_id = None

    def _action_at(self, actions: Any, index: int) -> Any:
        """Remove the time dimension from one batched conditioning action."""
        action_type = self._deps["action_type"]

        def take(value: Any) -> Any:
            return None if value is None else self._deps["jnp"].asarray(value[:, index])

        return action_type(
            binary=take(actions.binary),
            categorical=take(actions.categorical),
            continuous=take(actions.continuous),
        )

    def _build_action(self, input: OpenDreamerStepState) -> Any:
        """Build one native ``Actions`` value from the immutable control snapshot."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        mouse_to_categorical = self._deps["mouse_to_categorical"]
        binary = np.zeros((1, len(self._key_to_index)), dtype=np.int32)
        for key in input.pressed_keys:
            binary[0, self._key_to_index[_KEY_TO_VPT_NAME[key]]] = 1
        for button in input.pressed_mouse_buttons:
            binary[0, self._key_to_index[_BUTTON_TO_VPT_NAME[button]]] = 1
        if input.wheel_delta < 0:
            binary[0, self._key_to_index["mouse.wheel_neg"]] = 1
        elif input.wheel_delta > 0:
            binary[0, self._key_to_index["mouse.wheel_pos"]] = 1
        categorical = mouse_to_categorical(
            np.asarray([input.delta_x], dtype=np.float32),
            np.asarray([input.delta_y], dtype=np.float32),
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
