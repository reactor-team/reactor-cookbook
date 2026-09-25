"""Run native AlayaWorld inference independently of Reactor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from alayaworld_assets import (
    AlayaWorldConfig,
    load_scene_metadata,
    prepare_runtime_assets,
    scene_prompt_path,
    validate_runtime_paths,
)
from alayaworld_utils import (
    camera_frames,
    compact_rollout_cache,
    ensure_camera_capacity,
    load_upstream_modules,
    resolve_attention_backend,
    set_attention_backend,
    uploaded_image_video,
)

logger = logging.getLogger(__name__)
FPS = 24
FRAMES_PER_CHUNK = 32
_UPLOAD_DEFAULT_PROMPT = "Continue the visual scene shown in the reference image."


@dataclass(frozen=True)
class AlayaInput:
    """Supply an anchor once per world, then already-planned camera poses."""

    world_id: int
    prompt: str
    seed: int
    image: Path | bytes | None = None
    trajectory: np.ndarray | None = None


@dataclass(frozen=True)
class AlayaResult:
    """Report native setup facts or a completed chunk without echoing input."""

    world_id: int
    frames: np.ndarray | None
    completed_chunks: int
    active_prompt: str
    frames_wanted: int
    initial_pose: np.ndarray | None = None


class NoAnchorError(ValueError):
    """A new world has no image from which to initialize its cache."""


class InvalidTrajectoryError(ValueError):
    """Camera poses are missing or do not fit the native chunk window."""


class RolloutExhaustedError(RuntimeError):
    """The current world has reached its configured chunk limit."""


class AlayaWorldModel:
    """Own the upstream engine, rollout cache, and successful chunk count."""

    def __init__(self) -> None:
        self._config: AlayaWorldConfig | None = None
        self._torch: Any = None
        self._engine: Any = None
        self._alaya_pipeline: Any = None
        self._upstream_config: Any = None
        self._load_input_sample: Any = None
        self._check_input_resolution: Any = None
        self._plan_rollout: Any = None
        self._cache: Any = None
        self._chunk_latents = 0
        self._history_latents = 0
        self._gap_steps = 0
        self._condition_latents = 0
        self._ar_index = 0
        self._active_prompt = ""
        self._world_id: int | None = None

    def load(self, config: AlayaWorldConfig) -> None:
        """Load the public engine from resolved configuration and warm its kernels."""
        prepare_runtime_assets(config)
        validate_runtime_paths(config)
        modules = load_upstream_modules(config.source_path)
        torch = modules["torch"]

        upstream_config = modules["load_config"](str(config.upstream_config))
        upstream_config.paths.model = str(config.model.path)
        upstream_config.paths.gemma = str(config.gemma.path)
        upstream_config.paths.da3_repo = str(config.da3_source_path)
        upstream_config.paths.da3_model = config.da3_model.repo_id
        upstream_config.paths.da3_cache = str(config.da3_cache)
        upstream_config.paths.taehv = (
            str(config.taehv_path) if config.taehv_path else ""
        )

        mode_config = next(iter(upstream_config.validation.modes.values()))
        chunk_latents = int(mode_config.layout.output_latent_frames)
        history_latents = int(
            upstream_config.layout.history_latent_frames
            if mode_config.layout.history_latent_frames is None
            else mode_config.layout.history_latent_frames
        )
        gap_steps = int(
            float(mode_config.layout.max_gap_sec or 0.0)
            * float(upstream_config.sample.fps)
            / int(upstream_config.sample.temporal_stride)
        )
        condition_latents = int(mode_config.layout.condition_latent_frames)
        configured_fps = float(upstream_config.sample.fps)
        configured_chunk_frames = chunk_latents * int(
            upstream_config.sample.temporal_stride
        )
        if configured_fps != float(FPS):
            raise ValueError(
                f"AlayaWorld sample FPS must be {FPS}, got {configured_fps}"
            )
        if configured_chunk_frames != FRAMES_PER_CHUNK:
            raise ValueError(
                f"AlayaWorld chunks must contain {FRAMES_PER_CHUNK} frames, "
                f"got {configured_chunk_frames}"
            )

        flex_attention = config.flex_attention and config.compile_mode != "none"
        engine = modules["build_engine"](
            upstream_config,
            compile_mode=config.compile_mode,
            compile_aux=False,
            bank_taehv=config.bank_taehv,
            verbose=True,
        )
        attention = resolve_attention_backend(
            config.attention_backend,
            pytorch_attention=modules["pytorch_attention"],
            torch_module=torch,
        )
        if attention is not None:
            logger.info(
                "AlayaWorld attention backend=%s modules=%s",
                config.attention_backend,
                set_attention_backend(engine, attention),
            )
        if modules["apply_da3_robust_scale"]():
            logger.info("AlayaWorld DA3 colinear camera fallback enabled")
        alaya_pipeline = modules["pipeline_type"](
            engine,
            control_modes=list(mode_config.control),
            use_memory=bool(mode_config.use_memory),
            action_cfg_scale=float(mode_config.action_cfg_scale),
            flex_attn=flex_attention,
            seed=config.seed,
            ttc=config.ttc,
            ttc_levels=tuple(
                int(value) for value in upstream_config.validation.ttc.levels
            ),
            ttc_strength=float(upstream_config.validation.ttc.strength),
            ttc_ref_action=bool(upstream_config.validation.ttc.ref_action),
        )

        self._config = config
        self._torch = torch
        self._engine = engine
        self._alaya_pipeline = alaya_pipeline
        self._upstream_config = upstream_config
        self._load_input_sample = modules["load_input_sample"]
        self._check_input_resolution = modules["check_input_resolution"]
        self._plan_rollout = modules["plan_rollout"]
        self._chunk_latents = chunk_latents
        self._history_latents = history_latents
        self._gap_steps = gap_steps
        self._condition_latents = condition_latents
        self._warmup()
        logger.info("AlayaWorld model ready: chunk_frames=%s", configured_chunk_frames)

    def reset(self) -> None:
        """Discard a session's cache while keeping the loaded weights."""
        self._cache = None
        self._world_id = None
        self._ar_index = 0
        self._active_prompt = ""

    def generate(self, input: AlayaInput) -> AlayaResult:
        """Initialize a new world or advance exactly one native chunk."""
        if input.world_id != self._world_id:
            if input.image is None:
                raise NoAnchorError("a new AlayaWorld requires an image")
            initial_pose = self._reset_rollout(input.prompt, input.seed, input.image)
            self._world_id = input.world_id
            return AlayaResult(
                world_id=self._world_id,
                frames=None,
                completed_chunks=0,
                active_prompt=self._active_prompt,
                frames_wanted=self._frames_wanted(),
                initial_pose=initial_pose,
            )
        if self._ar_index >= self._config.max_chunks_per_rollout:
            raise RolloutExhaustedError(
                "the AlayaWorld rollout reached its chunk limit"
            )
        if input.trajectory is None:
            raise InvalidTrajectoryError(
                "camera poses are required after world initialization"
            )
        frames = self._generate_chunk(input.prompt, input.trajectory)
        return AlayaResult(
            world_id=self._world_id,
            frames=frames,
            completed_chunks=self._ar_index,
            active_prompt=self._active_prompt,
            frames_wanted=self._frames_wanted(),
        )

    def _frames_wanted(self) -> int:
        """Expose the exact next window computed from the native cache."""
        stride = int(self._alaya_pipeline.cfg.sample.temporal_stride)
        start = int(self._cache.target_start(self._ar_index)) * stride
        end = start + int(self._cache.K) * stride
        if self._ar_index == 0:
            start = max(0, start - stride + 1)
        return end - start

    def _warmup(self) -> None:
        """Compile on the configured warmup scene, then discard its rollout."""
        config = self._config
        if config is None:
            raise RuntimeError("AlayaWorld was not loaded")
        if config.warmup_chunks == 0:
            return
        scene = config.random_inputs[0]
        prompt = scene_prompt_path(scene).read_text(encoding="utf-8").strip()
        try:
            pose = self._reset_rollout(
                prompt or _UPLOAD_DEFAULT_PROMPT, config.seed, scene
            )
            for _ in range(config.warmup_chunks):
                trajectory = np.repeat(pose[None], self._frames_wanted(), axis=0)
                self._generate_chunk(prompt, trajectory)
        finally:
            self.reset()
        logger.info("AlayaWorld warmup complete: chunks=%s", config.warmup_chunks)

    def _reset_rollout(
        self,
        prompt: str,
        seed: int,
        selected_input: Path | bytes,
    ) -> np.ndarray:
        """Build a fresh upstream cache without reloading model weights."""
        config = self._config
        pipeline = self._alaya_pipeline
        if config is None or pipeline is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        self._cache = None
        video, metadata, needed_latents = self._prepare_scene(selected_input)
        pipeline.seed = seed
        cache = pipeline.initialize_cache(
            video,
            prompt,
            metadata,
            rounds=1,
            K=self._chunk_latents,
            cond_end=self._condition_latents,
            needed_latents=needed_latents,
        )
        stride = int(pipeline.cfg.sample.temporal_stride)
        anchor_index = max(0, int(cache.target_base_start) * stride - stride)
        camera = camera_frames(metadata["cam_c2w"])
        initial_pose = (
            camera[anchor_index].detach().cpu().to(self._torch.float32).numpy()
        )
        self._cache = cache
        self._ar_index = 0
        self._active_prompt = prompt
        return initial_pose

    def _prepare_scene(
        self,
        selected_input: Path | bytes,
    ) -> tuple[Any, dict[str, Any], int]:
        """Prepare one built-in or uploaded image for upstream cache initialization."""
        config = self._config
        upstream_config = self._upstream_config
        if config is None or upstream_config is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        target_hw = (
            int(upstream_config.sample.height),
            int(upstream_config.sample.width),
        )
        if isinstance(selected_input, bytes):
            metadata = load_scene_metadata(config.upload_template, self._torch)
            video = uploaded_image_video(
                selected_input,
                metadata,
                target_hw=target_hw,
                torch_module=self._torch,
            )
        else:
            video, _caption, metadata = self._load_input_sample(
                str(selected_input),
                image_target_hw=target_hw,
            )
        self._check_input_resolution(video, upstream_config)
        video, metadata, rounds, _max_rounds, needed_latents = self._plan_rollout(
            upstream_config,
            video,
            metadata,
            rounds_cap=1,
            K=self._chunk_latents,
            N=self._history_latents,
            gap_steps=self._gap_steps,
            cond_end=self._condition_latents,
        )
        if rounds != 1:
            raise RuntimeError("the selected AlayaWorld image cannot seed one chunk")
        return video, metadata, int(needed_latents)

    def _generate_chunk(
        self,
        prompt: str,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """Run one native AlayaWorld generate/finalize/decode turn."""
        pipeline = self._alaya_pipeline
        cache = self._cache
        engine = self._engine
        config = self._config
        if pipeline is None or cache is None or engine is None or config is None:
            raise RuntimeError("AlayaWorld rollout was not initialized")
        if prompt != self._active_prompt:
            cache.context = engine.encode_caption(prompt)
            self._active_prompt = prompt

        self._write_camera_trajectory(cache, trajectory)
        history = cache.history
        if history is None:
            raise RuntimeError("AlayaWorld interactive decode requires history latents")
        pred = pipeline.generate(self._ar_index, cache)
        pipeline.finalize(self._ar_index, cache, pred)
        compact_rollout_cache(
            cache,
            max_spatial_frames=config.max_spatial_frames,
            recent_spatial_frames=config.recent_spatial_frames,
        )
        frames = self._decode_new_frames(history, pred)
        if (
            frames.ndim != 4
            or frames.shape[0] != FRAMES_PER_CHUNK
            or frames.shape[-1] != 3
        ):
            raise RuntimeError("AlayaWorld must return one 32-frame RGB chunk")
        self._ar_index += 1
        return frames

    def _write_camera_trajectory(
        self,
        cache: Any,
        trajectory: np.ndarray,
    ) -> None:
        """Replace the next chunk's camera slots with frontend-controlled poses."""
        stride = int(self._alaya_pipeline.cfg.sample.temporal_stride)
        target_pixel_start = int(cache.target_start(self._ar_index)) * stride
        target_pixel_end = target_pixel_start + int(cache.K) * stride
        write_start = target_pixel_start
        if self._ar_index == 0:
            write_start = max(0, target_pixel_start - stride + 1)
        if trajectory.shape != (target_pixel_end - write_start, 4, 4):
            raise InvalidTrajectoryError(
                "camera trajectory does not match the native chunk window"
            )
        metadata = cast(dict[str, Any], cache.metadata)
        camera = metadata["cam_c2w"]
        camera = ensure_camera_capacity(camera, target_pixel_end, self._torch)
        values = self._torch.from_numpy(trajectory).to(
            device=camera.device, dtype=camera.dtype
        )
        if camera.dim() == 3:
            camera[write_start:target_pixel_end] = values
        else:
            camera[:, write_start:target_pixel_end] = values.unsqueeze(0).expand(
                camera.shape[0], -1, -1, -1
            )
        metadata["cam_c2w"] = camera
        if "cam_c2w_raw" in metadata:
            metadata["cam_c2w_raw"] = camera.clone()
        metadata["frame_end"] = int(camera_frames(camera).shape[0])

    def _decode_new_frames(self, history: Any, pred: Any) -> np.ndarray:
        """Decode one chunk with bounded left context and return its new frames."""
        config = self._config
        engine = self._engine
        if config is None or engine is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        overlap = min(config.decode_overlap_latents, int(history.shape[2]))
        latent = self._torch.cat(
            [history[:, :, -overlap:].contiguous(), pred.to(history.dtype)],
            dim=2,
        ).contiguous()
        decoded = engine.decode_latent_to_video_frames(latent)
        stride = int(self._alaya_pipeline.cfg.sample.temporal_stride)
        prefix_frames = (overlap - 1) * stride + 1
        frames = decoded[prefix_frames:]
        expected = int(pred.shape[2]) * stride
        if int(frames.shape[0]) != expected:
            raise RuntimeError(
                f"AlayaWorld decoded {int(frames.shape[0])} new frames; expected {expected}"
            )
        return np.ascontiguousarray(frames.numpy(), dtype=np.uint8)
