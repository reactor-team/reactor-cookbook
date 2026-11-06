"""Faithful single-GPU extraction of YUME-5B's rolling-latent sampler."""

from __future__ import annotations

import sys

import numpy as np
import torch
from yume_assets import YumeConfig
from yume_controls import conditioned_prompt
from yume_model import Movement, View


class YumeBackend:
    """Own YUME weights and its clean rolling latent/pixel continuation state."""

    def __init__(self, config: YumeConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("YUME-5B requires CUDA")
        sys.path.insert(0, str(config.source_path))
        from wan23 import Yume
        from wan23.configs import WAN_CONFIGS

        self.config = config
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.wan = Yume(
            config=WAN_CONFIGS["ti2v-5B"],
            checkpoint_dir=str(config.checkpoint_path),
            device_id=torch.cuda.current_device(),
        )
        # sample_5b.py main() enables TF32 matmuls for its float32 regions.
        torch.backends.cuda.matmul.allow_tf32 = True
        self.transformer = _wrap_like_sample_5b(
            self.wan.model.eval().requires_grad_(False).to(self.dtype)
        )
        self.wan.vae.model.to(self.device)
        self.model_input_latent: torch.Tensor | None = None
        self.model_input_pixels: torch.Tensor | None = None
        self.mode: str | None = None
        self.chunk_index = 0
        self.generator = torch.Generator(device=self.device)

    @torch.inference_mode()
    def reset(
        self,
        *,
        image: np.ndarray | None,
        video: np.ndarray | None = None,
        seed: int,
    ) -> None:
        """Initialize image or text mode while preserving upstream 32/8 context semantics."""
        self.generator.manual_seed(seed)
        self.model_input_latent = None
        self.model_input_pixels = None
        self.mode = (
            "video_to_video"
            if video is not None
            else ("image_to_video" if image is not None else "text_to_video")
        )
        self.chunk_index = 0
        if image is None and video is None:
            return
        if video is not None:
            visible = (
                _resize_conditioning(
                    video,
                    video=True,
                    height=self.config.height,
                    width=self.config.width,
                )
                .permute(1, 0, 2, 3)
                .to(self.device)
            )
        else:
            assert image is not None
            pixels = _resize_conditioning(
                image, video=False, height=self.config.height, width=self.config.width
            )[0].to(self.device)
            visible = torch.zeros(
                (3, 33, self.config.height, self.config.width), device=self.device
            )
            visible[:, 0] = pixels
        model_pixels = torch.cat(
            [visible[:, :1].repeat(1, 16, 1, 1), visible[:, :33]], dim=1
        )
        # sample_one encodes outside its bf16 autocast block.
        first = self.wan.vae.encode(
            [model_pixels[:, : -self.config.frames_per_chunk]]
        )[0]
        second = self.wan.vae.encode(
            [model_pixels[:, -self.config.frames_per_chunk :]]
        )[0]
        self.model_input_pixels = model_pixels
        self.model_input_latent = torch.cat([first, second], dim=1)

    @torch.inference_mode()
    def generate_chunk(
        self, *, prompt: str, movement: Movement, view: View
    ) -> np.ndarray:
        """Denoise eight new latents, decode 29 RGB frames, and retain clean context."""
        from wan23.utils.utils import masks_like

        text = conditioned_prompt(prompt, movement, view)
        latent_frames = self.config.latent_frames_per_chunk
        if self.mode == "text_to_video" and self.model_input_latent is None:
            arg_c, _arg_null, noise = self.wan.generate(
                text,
                frame_num=self.config.frames_per_chunk,
                max_area=self.config.width * self.config.height,
                latent_frame_zero=latent_frames,
                sampling_steps=self.config.sample_steps,
                shift=self.config.shift,
                seed=self.generator.initial_seed(),
                offload_model=False,
            )
            latent = noise
            mask2 = None
        else:
            if self.model_input_latent is None:
                raise RuntimeError("YUME rollout was not initialized")
            continuing = self.chunk_index > 0
            frame_num = (
                (self.model_input_latent.shape[1] - 1) * 4
                + 1
                + self.config.frames_per_chunk
                if continuing
                else self.model_input_pixels.shape[1]
            )
            img = (
                self.model_input_latent
                if continuing
                else self.model_input_latent[:, :-latent_frames]
            )
            arg_c, _arg_null, noise, mask2, _img = self.wan.generate(
                text,
                img=img,
                frame_num=frame_num,
                max_area=self.config.width * self.config.height,
                latent_frame_zero=latent_frames,
                sampling_steps=self.config.sample_steps,
                shift=self.config.shift,
                seed=self.generator.initial_seed(),
                offload_model=False,
            )
            if continuing:
                fresh = torch.randn(
                    (
                        self.wan.vae.model.z_dim,
                        self.model_input_latent.shape[1] + latent_frames,
                        self.model_input_latent.shape[2],
                        self.model_input_latent.shape[3],
                    ),
                    generator=self.generator,
                    device=self.device,
                    dtype=self.model_input_latent.dtype,
                )
                latent = torch.cat(
                    [self.model_input_latent, fresh[:, -latent_frames:]], dim=1
                )
                _mask1, mask2 = masks_like(
                    [latent], zero=True, latent_frame_zero=latent_frames
                )
            else:
                latent = torch.cat(
                    [
                        self.model_input_latent[:, :-latent_frames],
                        noise[:, -latent_frames:],
                    ],
                    dim=1,
                )

        sigmas = _sampling_sigmas(self.config.sample_steps, self.config.shift)
        for index, sigma in enumerate(sigmas):
            timestep = torch.tensor([sigma * 1000], device=self.device)
            if mask2 is None:
                tvec = timestep
            else:
                clean_ts = mask2[0][0][:-latent_frames, ::2, ::2].flatten()
                tvec = torch.cat(
                    [
                        clean_ts,
                        clean_ts.new_ones(arg_c["seq_len"] - clean_ts.numel())
                        * timestep,
                    ]
                ).unsqueeze(0)
            with torch.autocast("cuda", dtype=self.dtype):
                kwargs = {"flag": False} if mask2 is None else {}
                prediction = self.transformer(
                    [latent], t=tvec, latent_frame_zero=latent_frames, **kwargs, **arg_c
                )[0]
            next_sigma = 0.0 if index + 1 == len(sigmas) else sigmas[index + 1]
            tail = (
                latent[:, -latent_frames:]
                + (next_sigma - sigma) * prediction[:, -latent_frames:]
            )
            latent = torch.cat([latent[:, :-latent_frames], tail], dim=1)

        with torch.autocast("cuda", dtype=self.dtype):
            video = self.wan.vae.decode([latent[:, -latent_frames:].float()])[0][
                :, -self.config.frames_per_chunk :
            ]
        if self.model_input_latent is None:
            self.model_input_latent = latent[:, -latent_frames:]
        elif self.chunk_index == 0:
            self.model_input_latent = torch.cat(
                [
                    self.model_input_latent[:, :-latent_frames],
                    latent[:, -latent_frames:],
                ],
                dim=1,
            )
        else:
            self.model_input_latent = torch.cat(
                [self.model_input_latent, latent[:, -latent_frames:]], dim=1
            )
        # Pixel conditioning is read only while preparing the first chunk.
        self.model_input_pixels = None
        self.chunk_index += 1
        return _to_uint8(video)

    def end_session(self) -> None:
        self.model_input_latent = None
        self.model_input_pixels = None
        self.mode = None
        self.chunk_index = 0
        torch.cuda.empty_cache()


def _wrap_like_sample_5b(transformer: torch.nn.Module) -> torch.nn.Module:
    """Wrap the bf16 DiT in single-rank FSDP exactly as sample_5b.py main() does.

    get_dit_fsdp_kwargs(master_weight_type="fp32") computes on float32 copies of
    the bf16 weights; the FSDP-managed parameters also change kernel numerics,
    so a plain float32 module is not bit-identical. One local rank needs no
    collectives; an in-memory store avoids opening a port.
    """
    import functools

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from wan23.modules.model import WanAttentionBlock

    if dist.is_initialized():
        raise RuntimeError("YUME expects to own a single-rank process group")
    dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
    wrapped = FSDP(
        transformer,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=(WanAttentionBlock,)
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return wrapped.eval().requires_grad_(False)


def _resize_conditioning(
    pixels: np.ndarray, *, video: bool, height: int, width: int
) -> torch.Tensor:
    """Match sample_5b's floating-point image/video interpolation and ordering."""
    batch = pixels if video else pixels[None]
    tensor = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).float().div(255.0)
    if video:
        tensor = tensor.sub(0.5).div(0.5)
        return torch.nn.functional.interpolate(
            tensor, size=(height, width), mode="bicubic", align_corners=False
        ).clamp(-1, 1)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )
    return tensor.sub(0.5).mul(2)


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    """Quantize like sample_5b's save_video (diffusers VideoProcessor, PIL output).

    VaeImageProcessor denormalizes on the tensor's device, moves to NumPy and
    rounds `images * 255` half-to-even; truncation would bias every value down.
    """
    frames = (video.float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()
    return (frames * 255).round().astype(np.uint8)


def _sampling_sigmas(steps: int, shift: float) -> np.ndarray:
    sigma = np.linspace(1, 0, steps + 1)[:steps]
    return shift * sigma / (1 + (shift - 1) * sigma)
