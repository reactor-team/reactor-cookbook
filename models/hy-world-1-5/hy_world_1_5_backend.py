"""Run stateful HY-World 1.5 autoregressive inference in the serving process."""

from __future__ import annotations

import atexit
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
from reactor_runtime.log import get_logger

from hy_world_1_5_assets import HYWorld15Config, assemble_base_model
from hy_world_1_5_camera import CameraChunk

logger = get_logger(__name__)

_HEIGHT = 480
_WIDTH = 832
_LATENTS_PER_CHUNK = 4
_FLOW_SHIFT = 5.0


class HYWorld15Backend:
    """Own upstream model weights, causal history, KV cache, and VAE cache."""

    def __init__(self, config: HYWorld15Config) -> None:
        self._config = config
        self._torch: Any = None
        self._pipe: Any = None
        self._select_memory: Any = None
        self._generate_points: Any = None
        self._device: Any = None
        self._generator: Any = None
        self._kv_cache: Any = None
        self._vision_states: Any = None
        self._initial_cond_chunk: Any = None
        self._latent_history: Any = None
        self._viewmat_history = np.empty((0, 4, 4), dtype=np.float32)
        self._intrinsic_history = np.empty((0, 3, 3), dtype=np.float32)
        self._action_history = np.empty((0,), dtype=np.int64)
        self._points_local: Any = None
        self._active_prompt = ""
        self._decoded_latents = 0

    def load(self) -> None:
        """Load the pinned upstream distilled model once and keep it resident."""
        assemble_base_model(self._config)
        source = self._config.source.path
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("HY-World 1.5 requires a CUDA device")
        torch.cuda.set_device(0)
        self._initialize_upstream_globals(torch)

        pipeline_module = importlib.import_module(
            "hyvideo.pipelines.worldplay_video_pipeline"
        )
        retrieval_module = importlib.import_module("hyvideo.utils.retrieval_context")
        pipeline_type = pipeline_module.HunyuanVideo_1_5_Pipeline
        action_checkpoint = (
            self._config.action_model.path
            / "ar_distilled_action_model/model.safetensors"
        )
        pipe = pipeline_type.create_pipeline(
            pretrained_model_name_or_path=str(self._config.base_model.path),
            transformer_version="480p_i2v",
            create_sr_pipeline=False,
            force_sparse_attn=False,
            transformer_dtype=torch.bfloat16,
            enable_offloading=False,
            enable_group_offloading=False,
            device=torch.device("cuda"),
            action_ckpt=str(action_checkpoint),
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.transformer.eval().requires_grad_(False)
        pipe.vae.eval().requires_grad_(False)
        pipe.text_encoder.model.eval().requires_grad_(False)
        pipe.byt5_model.eval().requires_grad_(False)
        pipe.vision_encoder.eval().requires_grad_(False)
        pipe.vae.disable_tiling()

        self._torch = torch
        self._pipe = pipe
        self._device = torch.device("cuda")
        self._select_memory = retrieval_module.select_aligned_memory_frames
        self._generate_points = retrieval_module.generate_points_in_sphere
        logger.info(
            "HY-World 1.5 backend ready",
            source_revision=self._config.source.revision,
            action_revision=self._config.action_model.revision,
            gpu=torch.cuda.get_device_name(0),
        )

    def reset(self, *, image: Image.Image, prompt: str, seed: int) -> None:
        """Initialize a fresh image-conditioned world and its text KV cache."""
        torch = self._require_torch()
        pipe = self._require_pipe()
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("HY-World 1.5 requires a non-empty prompt")

        pipe.scheduler = pipe._create_scheduler(_FLOW_SHIFT)
        pipe._guidance_scale = 1.0
        pipe._guidance_rescale = 0.0
        pipe._clip_skip = None
        self._generator = torch.Generator(device=self._device).manual_seed(seed)
        template = torch.zeros(
            (
                1,
                int(pipe.transformer.config.in_channels),
                _LATENTS_PER_CHUNK,
                _HEIGHT // int(pipe.vae_spatial_compression_ratio),
                _WIDTH // int(pipe.vae_spatial_compression_ratio),
            ),
            device=self._device,
            dtype=pipe.target_dtype,
        )

        with torch.inference_mode():
            image_cond = pipe.get_image_condition_latents("i2v", image, _HEIGHT, _WIDTH)
            task_mask = pipe.get_task_mask("i2v", _LATENTS_PER_CHUNK)
            self._initial_cond_chunk = pipe._prepare_cond_latents(
                "i2v", image_cond, template, task_mask
            )
            self._vision_states = pipe._prepare_vision_states(
                np.asarray(image), "480p", template, self._device
            )
            pipe.init_kv_cache()
            self._kv_cache = pipe._kv_cache
            self._encode_text_cache(normalized_prompt)
            self._points_local = self._generate_points(
                self._config.points_in_sphere, 8.0
            ).to(self._device)
            pipe.vae.clear_cache()

        self._latent_history = None
        self._viewmat_history = np.empty((0, 4, 4), dtype=np.float32)
        self._intrinsic_history = np.empty((0, 3, 3), dtype=np.float32)
        self._action_history = np.empty((0,), dtype=np.int64)
        self._active_prompt = normalized_prompt
        self._decoded_latents = 0

    def generate_chunk(self, camera: CameraChunk, prompt: str) -> np.ndarray:
        """Generate and decode exactly one four-latent causal chunk."""
        torch = self._require_torch()
        pipe = self._require_pipe()
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("HY-World 1.5 requires a non-empty prompt")
        if self._generator is None or self._initial_cond_chunk is None:
            raise RuntimeError("Reset HY-World 1.5 before generating a chunk")
        _validate_camera_chunk(camera)

        with torch.inference_mode():
            if normalized_prompt != self._active_prompt:
                self._encode_text_cache(normalized_prompt)

            current_viewmats = np.concatenate(
                [self._viewmat_history, camera.viewmats], axis=0
            )
            current_intrinsics = np.concatenate(
                [self._intrinsic_history, camera.intrinsics], axis=0
            )
            current_actions = np.concatenate(
                [self._action_history, camera.actions], axis=0
            )
            current_start = len(self._action_history)
            current_latents = pipe.prepare_latents(
                batch_size=1,
                num_channels_latents=int(pipe.transformer.config.in_channels),
                latent_height=_HEIGHT // int(pipe.vae_spatial_compression_ratio),
                latent_width=_WIDTH // int(pipe.vae_spatial_compression_ratio),
                video_length=_LATENTS_PER_CHUNK,
                dtype=pipe.target_dtype,
                device=self._device,
                generator=self._generator,
            )
            current_cond = self._current_condition(current_start)
            selected = self._selected_context_indices(current_viewmats, current_start)
            if selected:
                self._cache_visual_context(
                    selected,
                    current_viewmats,
                    current_intrinsics,
                    current_actions,
                )
            denoised = self._denoise_chunk(
                current_latents,
                current_cond,
                camera,
                context_length=len(selected),
            )
            frames = self._decode_chunk(denoised)

        cpu_latents = denoised.detach().to(device="cpu", dtype=pipe.target_dtype)
        if self._latent_history is None:
            self._latent_history = cpu_latents
        else:
            self._latent_history = torch.cat([self._latent_history, cpu_latents], dim=2)
        self._viewmat_history = current_viewmats
        self._intrinsic_history = current_intrinsics
        self._action_history = current_actions
        self._active_prompt = normalized_prompt
        self._decoded_latents += _LATENTS_PER_CHUNK
        return frames

    def end_session(self) -> None:
        """Release per-world history while retaining the loaded model weights."""
        pipe = self._pipe
        if pipe is not None:
            pipe.vae.clear_cache()
            pipe.init_kv_cache()
        self._generator = None
        self._kv_cache = None
        self._vision_states = None
        self._initial_cond_chunk = None
        self._latent_history = None
        self._points_local = None
        self._viewmat_history = np.empty((0, 4, 4), dtype=np.float32)
        self._intrinsic_history = np.empty((0, 3, 3), dtype=np.float32)
        self._action_history = np.empty((0,), dtype=np.int64)
        self._active_prompt = ""
        self._decoded_latents = 0

    def _initialize_upstream_globals(self, torch: Any) -> None:
        """Initialize the single-rank public inference state expected upstream."""
        dist = torch.distributed
        if not dist.is_initialized():
            rendezvous = self._config.cache_path / f"torch-dist-{os.getpid()}"
            rendezvous.unlink(missing_ok=True)
            dist.init_process_group(
                backend="nccl",
                init_method=f"file://{rendezvous}",
                rank=0,
                world_size=1,
            )
            atexit.register(_destroy_process_group, dist)
        if dist.get_world_size() != 1:
            raise RuntimeError("The Reactor adapter expects one visible inference GPU")

        parallel = importlib.import_module("hyvideo.commons.parallel_states")
        infer_state = importlib.import_module("hyvideo.commons.infer_state")
        parallel.initialize_parallel_state(sp=1)
        infer_state.initialize_infer_state(
            SimpleNamespace(
                sage_blocks_range="0",
                use_sageattn=False,
                enable_torch_compile=False,
                use_fp8_gemm=False,
                quant_type="fp8-per-block",
                include_patterns="double_blocks",
                use_vae_parallel=False,
            )
        )

    def _encode_text_cache(self, prompt: str) -> None:
        """Replace text KV entries while leaving visual context to be rebuilt."""
        torch = self._require_torch()
        pipe = self._require_pipe()
        (
            prompt_embeds,
            _,
            prompt_mask,
            _,
        ) = pipe.encode_prompt(
            prompt,
            self._device,
            1,
            False,
            data_type="video",
        )
        extra_kwargs = pipe._prepare_byt5_embeddings(prompt, self._device)
        t_text = torch.zeros((1,), device=self._device, dtype=pipe.target_dtype)
        with torch.autocast(device_type="cuda", dtype=pipe.target_dtype):
            self._kv_cache = pipe.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=t_text,
                text_states=prompt_embeds[0, None, ...],
                encoder_attention_mask=prompt_mask[0, None, ...],
                vision_states=self._vision_states[0, None, ...],
                mask_type="i2v",
                extra_kwargs={
                    "byt5_text_states": extra_kwargs["byt5_text_states"][0, None, ...],
                    "byt5_text_mask": extra_kwargs["byt5_text_mask"][0, None, ...],
                },
                kv_cache=self._kv_cache,
                cache_txt=True,
            )
        pipe._kv_cache = self._kv_cache

    def _current_condition(self, current_start: int) -> Any:
        """Return the image/mask condition for the next four latent positions."""
        torch = self._require_torch()
        if current_start == 0:
            return self._initial_cond_chunk.clone()
        return torch.zeros_like(self._initial_cond_chunk)

    def _selected_context_indices(
        self, viewmats: np.ndarray, current_start: int
    ) -> list[int]:
        """Select official temporal and geometry-aligned memory for this chunk."""
        if current_start == 0:
            return []
        selected = self._select_memory(
            viewmats,
            current_start,
            memory_frames=self._config.memory_frames,
            temporal_context_size=self._config.temporal_context_size,
            pred_latent_size=_LATENTS_PER_CHUNK,
            points_local=self._points_local,
            device=self._device,
        )
        current = set(range(current_start, current_start + _LATENTS_PER_CHUNK))
        return sorted({int(index) for index in selected if int(index) not in current})

    def _cache_visual_context(
        self,
        selected: list[int],
        viewmats: np.ndarray,
        intrinsics: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Reconstitute visual KV entries from the selected latent history."""
        torch = self._require_torch()
        pipe = self._require_pipe()
        if self._latent_history is None:
            raise RuntimeError(
                "Visual context was requested before any latent history exists"
            )
        index_cpu = torch.as_tensor(selected, dtype=torch.long)
        context_latents = self._latent_history.index_select(2, index_cpu).to(
            self._device
        )
        context_cond = torch.zeros(
            (
                1,
                int(self._initial_cond_chunk.shape[1]),
                len(selected),
                int(self._initial_cond_chunk.shape[3]),
                int(self._initial_cond_chunk.shape[4]),
            ),
            device=self._device,
            dtype=self._initial_cond_chunk.dtype,
        )
        if 0 in selected:
            context_cond[:, :, selected.index(0)] = self._initial_cond_chunk[:, :, 0]
        hidden_states = torch.cat([context_latents, context_cond], dim=1)
        context_viewmats = torch.as_tensor(
            viewmats[selected], device=self._device, dtype=pipe.target_dtype
        ).unsqueeze(0)
        context_intrinsics = torch.as_tensor(
            intrinsics[selected], device=self._device, dtype=pipe.target_dtype
        ).unsqueeze(0)
        context_actions = torch.as_tensor(
            actions[selected], device=self._device, dtype=pipe.target_dtype
        ).unsqueeze(0)
        timestep = torch.full(
            (len(selected),),
            self._config.stabilization_level - 1,
            device=self._device,
            dtype=pipe.target_dtype,
        )
        with torch.autocast(device_type="cuda", dtype=pipe.target_dtype):
            self._kv_cache = pipe.transformer(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                hidden_states=hidden_states,
                timestep=timestep,
                timestep_r=None,
                mask_type="i2v",
                return_dict=False,
                viewmats=context_viewmats,
                Ks=context_intrinsics,
                action=context_actions,
                kv_cache=self._kv_cache,
                cache_vision=True,
                rope_temporal_size=len(selected),
                start_rope_start_idx=0,
            )
        pipe._kv_cache = self._kv_cache

    def _denoise_chunk(
        self,
        latents: Any,
        cond: Any,
        camera: CameraChunk,
        *,
        context_length: int,
    ) -> Any:
        """Apply the distilled four-step scheduler to only the current chunk."""
        torch = self._require_torch()
        pipe = self._require_pipe()
        pipe.scheduler.set_timesteps(self._config.inference_steps, device=self._device)
        viewmats = torch.as_tensor(
            camera.viewmats, device=self._device, dtype=pipe.target_dtype
        ).unsqueeze(0)
        intrinsics = torch.as_tensor(
            camera.intrinsics, device=self._device, dtype=pipe.target_dtype
        ).unsqueeze(0)
        actions = torch.as_tensor(
            camera.actions, device=self._device, dtype=pipe.target_dtype
        ).unsqueeze(0)

        for timestep in pipe.scheduler.timesteps:
            timestep_input = torch.full(
                (_LATENTS_PER_CHUNK,),
                timestep,
                device=self._device,
                dtype=pipe.scheduler.timesteps.dtype,
            )
            model_input = torch.cat([latents, cond], dim=1)
            model_input = pipe.scheduler.scale_model_input(model_input, timestep)
            with torch.autocast(device_type="cuda", dtype=pipe.target_dtype):
                noise_prediction = pipe.transformer(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=True,
                    hidden_states=model_input,
                    timestep=timestep_input,
                    timestep_r=None,
                    mask_type="i2v",
                    return_dict=False,
                    viewmats=viewmats,
                    Ks=intrinsics,
                    action=actions,
                    kv_cache=self._kv_cache,
                    cache_vision=False,
                    rope_temporal_size=_LATENTS_PER_CHUNK + context_length,
                    start_rope_start_idx=context_length,
                )[0]
            updated = pipe.scheduler.step(
                noise_prediction, timestep, latents, return_dict=False
            )[0]
            latents.copy_(updated[:, :, -_LATENTS_PER_CHUNK:])
        return latents

    def _decode_chunk(self, latents: Any) -> np.ndarray:
        """Decode new latents through the persistent causal VAE feature cache."""
        torch = self._require_torch()
        pipe = self._require_pipe()
        vae = pipe.vae
        if getattr(vae.config, "shift_factor", None):
            scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
        else:
            scaled = latents / vae.config.scaling_factor

        decoded_parts = []
        with torch.autocast(device_type="cuda", dtype=pipe.vae_dtype):
            for offset in range(_LATENTS_PER_CHUNK):
                vae._conv_idx = [0]
                decoded_parts.append(
                    vae.decoder(
                        scaled[:, :, offset : offset + 1],
                        feat_cache=vae._feat_map,
                        feat_idx=vae._conv_idx,
                        first_chunk=self._decoded_latents + offset == 0,
                    )
                )
        decoded = torch.cat(decoded_parts, dim=2)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).round().to(torch.uint8).cpu()
        return np.ascontiguousarray(decoded[0].permute(1, 2, 3, 0).numpy())

    def _require_torch(self) -> Any:
        """Return the loaded torch module."""
        if self._torch is None:
            raise RuntimeError("HY-World 1.5 backend is not loaded")
        return self._torch

    def _require_pipe(self) -> Any:
        """Return the loaded upstream pipeline."""
        if self._pipe is None:
            raise RuntimeError("HY-World 1.5 backend is not loaded")
        return self._pipe


def _validate_camera_chunk(camera: CameraChunk) -> None:
    """Require four aligned native camera conditions."""
    expected = _LATENTS_PER_CHUNK
    if camera.viewmats.shape != (expected, 4, 4):
        raise ValueError(f"viewmats must have shape ({expected}, 4, 4)")
    if camera.intrinsics.shape != (expected, 3, 3):
        raise ValueError(f"intrinsics must have shape ({expected}, 3, 3)")
    if camera.actions.shape != (expected,):
        raise ValueError(f"actions must have shape ({expected},)")


def _destroy_process_group(distributed: Any) -> None:
    """Release the process-local communication group during interpreter shutdown."""
    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()
