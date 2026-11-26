# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Expose LingBot-World-Fast's native causal loop one chunk at a time.

Imported by the backend after the pinned LingBot source is on ``sys.path``.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange

from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings


class InteractiveFastRollout:
    """Keep LingBot's native KV and VAE caches alive across chunk requests."""

    def __init__(
        self,
        pipe,
        *,
        max_chunks: int,
        context_latents: int = 21,
        chunk_size: int = 3,
        max_area: int = 480 * 832,
        shift: float = 10.0,
        timesteps_index: tuple[int, ...] = (0, 179, 358, 679),
        max_sequence_length: int = 512,
    ) -> None:
        if chunk_size != 3:
            raise ValueError("LingBot-World-Fast expects three latent frames per chunk")
        if context_latents < chunk_size or context_latents % chunk_size:
            raise ValueError(
                "context_latents must be a positive multiple of chunk_size"
            )
        if max_chunks <= 0 or max_chunks * chunk_size > 1024:
            raise ValueError(
                "max_chunks must keep the latent RoPE timeline within 1024 frames"
            )
        if pipe.local_attn_size != context_latents:
            raise ValueError(
                "the pipeline local attention size must match context_latents "
                f"({pipe.local_attn_size} != {context_latents})"
            )
        self.pipe = pipe
        self.max_chunks = max_chunks
        self.context_latents = context_latents
        self.chunk_size = chunk_size
        self.max_area = max_area
        self.shift = shift
        self.timesteps_index = timesteps_index
        self.max_sequence_length = max_sequence_length
        self._active = False

    def reset(self, prompt: str, image, intrinsics: np.ndarray, seed: int) -> None:
        """Start a fresh causal rollout without reloading model weights."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("LingBot-World-Fast requires a non-empty prompt")
        self.end()
        pipe = self.pipe
        image_tensor = TF.to_tensor(image).sub_(0.5).div_(0.5).to(pipe.device)
        image_h, image_w = image_tensor.shape[1:]
        aspect_ratio = image_h / image_w
        self.lat_h = round(
            np.sqrt(self.max_area * aspect_ratio)
            // pipe.vae_stride[1]
            // pipe.patch_size[1]
            * pipe.patch_size[1]
        )
        self.lat_w = round(
            np.sqrt(self.max_area / aspect_ratio)
            // pipe.vae_stride[2]
            // pipe.patch_size[2]
            * pipe.patch_size[2]
        )
        self.height = self.lat_h * pipe.vae_stride[1]
        self.width = self.lat_w * pipe.vae_stride[2]
        self.frame_seqlen = self.lat_h * self.lat_w // 4
        self.max_seq_len = int(
            math.ceil(self.chunk_size * self.frame_seqlen / pipe.sp_size) * pipe.sp_size
        )

        packed_intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
        if packed_intrinsics.ndim == 2:
            packed_intrinsics = packed_intrinsics[0]
        if tuple(packed_intrinsics.shape) != (4,):
            raise ValueError(
                "camera intrinsics must have shape (4,) or (N, 4); "
                f"got {tuple(packed_intrinsics.shape)}"
            )
        self.intrinsics = get_Ks_transformed(
            packed_intrinsics[None],
            height_org=480,
            width_org=832,
            height_resize=self.height,
            width_resize=self.width,
            height_final=self.height,
            width_final=self.width,
        )[0].to(pipe.device)

        seed = int(seed)
        if seed < 0:
            raise ValueError("interactive seeds must be non-negative")
        self.seed_generator = torch.Generator(device=pipe.device)
        self.seed_generator.manual_seed(seed)
        self.noise = torch.randn(
            16,
            self.max_chunks * self.chunk_size,
            self.lat_h,
            self.lat_w,
            dtype=torch.float32,
            generator=self.seed_generator,
            device=pipe.device,
        )

        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=self.shift)
        self.timesteps = pipe.scheduler.timesteps[list(self.timesteps_index)]
        self.prompt = prompt
        self.context = self._encode_prompt(prompt, clear_cache=True)
        self.cross_kv_cache = self._new_cross_attention_cache()
        self.cross_attn_initialized = False

        self.anchor_image = (
            torch.nn.functional.interpolate(
                image_tensor[None].cpu(),
                size=(self.height, self.width),
                mode="bicubic",
            )
            .transpose(0, 1)
            .to(pipe.device)
        )

        model_args = pipe.model.config
        kv_size = self.frame_seqlen * self.context_latents
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // pipe.sp_size
        self.self_kv_cache = pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[1, kv_size, local_num_heads, head_dim],
            dtype=pipe.pipe_dtype,
            device=pipe.device,
        )
        pipe.vae.model.clear_cache()
        self.chunk_index = 0
        self._active = True

    def generate_chunk(self, relative_c2ws: np.ndarray, prompt: str) -> torch.Tensor:
        """Generate and decode exactly one native three-latent causal chunk."""
        if not self._active:
            raise RuntimeError("start an interactive rollout before generating a chunk")
        if self.chunk_index >= self.max_chunks:
            raise RuntimeError("interactive rollout reached its configured chunk limit")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("LingBot-World-Fast requires a non-empty prompt")
        if prompt != self.prompt:
            self.prompt = prompt
            self.context = self._encode_prompt(prompt, clear_cache=True)
            self.cross_kv_cache = self._new_cross_attention_cache()
            self.cross_attn_initialized = False

        c2ws = torch.as_tensor(
            relative_c2ws, dtype=torch.float32, device=self.pipe.device
        )
        expected_pose_shape = (self.chunk_size, 4, 4)
        if tuple(c2ws.shape) != expected_pose_shape:
            raise ValueError(
                f"relative camera poses must have shape {expected_pose_shape}; got {tuple(c2ws.shape)}"
            )
        intrinsics = self.intrinsics.repeat(self.chunk_size, 1)
        plucker = get_plucker_embeddings(c2ws, intrinsics, self.height, self.width)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=self.height // self.lat_h,
            c2=self.width // self.lat_w,
        )
        plucker = rearrange(
            plucker[None],
            "b (f h w) c -> b c f h w",
            f=self.chunk_size,
            h=self.lat_h,
            w=self.lat_w,
        ).to(self.pipe.param_dtype)

        latent_start = self.chunk_index * self.chunk_size
        latent_end = latent_start + self.chunk_size
        current_latent = self.noise[:, latent_start:latent_end]
        current_condition = self._encode_condition_chunk()

        current_start = latent_start * self.frame_seqlen
        kv_size = self.frame_seqlen * self.context_latents
        kwargs = {
            "context": [self.context[0]],
            "seq_len": self.max_seq_len,
            "y": [current_condition],
            "dit_cond_dict": {"c2ws_plucker_emb": plucker.chunk(1, dim=0)},
            "kv_cache": self.self_kv_cache,
            "crossattn_cache": self.cross_kv_cache,
            "current_start": current_start,
            "max_attention_size": kv_size,
            "frame_seqlen": self.frame_seqlen,
        }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.pipe.model, "no_sync", noop_no_sync)
        with (
            torch.amp.autocast("cuda", dtype=self.pipe.param_dtype),
            torch.no_grad(),
            no_sync_model(),
        ):
            for timestep_index, timestep_value in enumerate(self.timesteps):
                timestep = timestep_value[None].to(self.pipe.device)
                noise_pred = self.pipe.model(
                    x=[current_latent],
                    t=timestep,
                    cross_attn_first_call=not self.cross_attn_initialized,
                    **kwargs,
                )[0]
                self.cross_attn_initialized = True
                x0 = self.pipe._convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=current_latent,
                    timestep=timestep_value,
                    scheduler=self.pipe.scheduler,
                )
                if timestep_index < len(self.timesteps) - 1:
                    current_latent = self.pipe.scheduler.add_noise(
                        x0,
                        torch.randn(
                            x0.shape,
                            generator=self.seed_generator,
                            device=x0.device,
                            dtype=x0.dtype,
                        ),
                        self.timesteps[timestep_index + 1],
                    )
            zero_timestep = (self.timesteps[-1] * 0.0)[None].to(self.pipe.device)
            self.pipe.model(
                x=[x0],
                t=zero_timestep,
                cross_attn_first_call=False,
                **kwargs,
            )

        video = self._decode_chunk(x0)
        self.chunk_index += 1
        return video

    def end(self) -> None:
        """Release rollout caches while preserving loaded model weights."""
        if not self._active:
            return
        self._active = False
        for name in (
            "noise",
            "anchor_image",
            "context",
            "self_kv_cache",
            "cross_kv_cache",
            "seed_generator",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self.pipe.vae.model.clear_cache()
        torch.cuda.empty_cache()

    def _encode_prompt(self, prompt: str, *, clear_cache: bool) -> list[torch.Tensor]:
        """Encode one prompt with the upstream T5 path and bounded cache."""
        pipe = self.pipe
        if clear_cache:
            pipe.clear_text_cache()
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if cache_key not in pipe._t5_cache:
            if not pipe.t5_cpu:
                pipe.text_encoder.model.to(pipe.device)
                context = pipe.text_encoder([prompt], pipe.device)
            else:
                context = pipe.text_encoder([prompt], torch.device("cpu"))
                context = [tensor.to(pipe.device) for tensor in context]
            pipe._t5_cache[cache_key] = context
        return pipe._t5_cache[cache_key]

    def _new_cross_attention_cache(self):
        """Allocate cross-attention storage for the active prompt."""
        model_args = self.pipe.model.config
        head_dim = model_args.dim // model_args.num_heads
        return self.pipe._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[1, self.max_sequence_length, model_args.num_heads, head_dim],
            dtype=self.pipe.pipe_dtype,
            device=self.pipe.device,
        )

    def _encode_condition_chunk(self) -> torch.Tensor:
        """Encode the anchor-plus-zero I2V condition with causal VAE state."""
        vae = self.pipe.vae
        model = vae.model
        encoded = []
        with self._vae_autocast():
            for latent_offset in range(self.chunk_size):
                if self.chunk_index == 0 and latent_offset == 0:
                    video_input = self.anchor_image[None]
                else:
                    video_input = torch.zeros(
                        1,
                        3,
                        self.pipe.vae_stride[0],
                        self.height,
                        self.width,
                        dtype=self.anchor_image.dtype,
                        device=self.pipe.device,
                    )
                model._enc_conv_idx = [0]
                features = model.encoder(
                    video_input,
                    feat_cache=model._enc_feat_map,
                    feat_idx=model._enc_conv_idx,
                )
                mean, _ = model.conv1(features).chunk(2, dim=1)
                mean = mean - vae.scale[0].view(1, model.z_dim, 1, 1, 1)
                mean = mean * vae.scale[1].view(1, model.z_dim, 1, 1, 1)
                encoded.append(mean.float().squeeze(0))
        mask = torch.zeros(
            4,
            self.chunk_size,
            self.lat_h,
            self.lat_w,
            dtype=encoded[0].dtype,
            device=self.pipe.device,
        )
        if self.chunk_index == 0:
            mask[:, 0] = 1
        return torch.cat([mask, torch.cat(encoded, dim=1)])

    def _decode_chunk(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent chunk while retaining the upstream causal VAE cache."""
        vae = self.pipe.vae
        model = vae.model
        z = latent[None]
        with self._vae_autocast():
            z = z / vae.scale[1].view(1, model.z_dim, 1, 1, 1)
            z = z + vae.scale[0].view(1, model.z_dim, 1, 1, 1)
            decoded_input = model.conv2(z)
            frames = []
            for latent_index in range(decoded_input.shape[2]):
                model._conv_idx = [0]
                frames.append(
                    model.decoder(
                        decoded_input[:, :, latent_index : latent_index + 1],
                        feat_cache=model._feat_map,
                        feat_idx=model._conv_idx,
                    )
                )
        return torch.cat(frames, dim=2).float().clamp_(-1, 1).squeeze(0)

    def _vae_autocast(self):
        """Match the upstream VAE precision without deprecated AMP APIs."""
        dtype = self.pipe.vae.dtype
        if dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast("cuda", dtype=dtype)
        return nullcontext()
