"""Incremental execution of the released DMD + MemRoPE inference path.

The upstream modules own the attention, memory compression, sampling and VAE.
This wrapper retains their loop state between three-latent-frame calls.
"""

import random
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from history_forcing_model import Settings

if TYPE_CHECKING:
    import torch


@dataclass
class Run:
    """Transient sampling state, discarded together when a video ends."""

    noise: "torch.Tensor"
    noise_offset: int = 0
    position: int = 0
    prompt: str | None = None
    conditioning: "dict[str, torch.Tensor] | None" = None


class NativeBackend:
    def __init__(self, settings: Settings) -> None:
        import torch

        sys.path.insert(0, str(settings.source / "dmd"))
        from evaluation.generator import load_generator
        from long_video.config import load_settings, validate_model_config
        from long_video.memrope import apply_memrope
        from pipeline.causal_inference import CausalInferencePipeline
        from utils.paths import load_config
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0")
        config = load_config(
            settings.source / "configs/long_video/dmd_ar10s_bi1p3b.yaml"
        )
        validate_model_config(config)
        self.native_settings = load_settings(
            settings.source / "configs/long_video/inference.yaml"
        )
        # Explicit asset paths keep upstream source read-only and weights resident.
        self.pipe = CausalInferencePipeline(
            config,
            device=self.device,
            generator=WanDiffusionWrapper(
                **config.model_kwargs, is_causal=True, model_root=settings.model_root
            ),
            text_encoder=WanTextEncoder(model_root=settings.model_root),
            vae=WanVAEWrapper(model_root=settings.model_root),
        )
        load_generator(self.pipe, settings.checkpoint, use_ema=True)
        self.pipe.to(dtype=torch.bfloat16, device=self.device).eval()
        apply_memrope(self.pipe, self.native_settings["memrope"])
        self.reset()

    def reset(self) -> None:
        self.pipe.kv_cache1 = None
        self.pipe.crossattn_cache = None
        self.pipe.vae.model.clear_cache()
        self.run: Run | None = None

    def begin(self, seed: int) -> None:
        import torch
        from long_video.config import validate_seed

        validate_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.run = Run(noise=self._noise_page())
        self.pipe._initialize_kv_cache(1, torch.bfloat16, self.device)
        self.pipe._initialize_crossattn_cache(1, torch.bfloat16, self.device)

    def _noise_page(self):
        import torch

        # The native launcher draws all 252 initial latents before transition
        # noise. The first page preserves that RNG order exactly. Further pages
        # extend the same random stream without resetting visual/decoder memory.
        return torch.randn(
            (1, self.native_settings["generation"]["latent_frames"], 16, 60, 104),
            device=self.device,
            dtype=torch.bfloat16,
        )

    def denoise(self, prompt: str):
        """One native block; also used by numerical validation before decoding."""
        import torch

        p = self.pipe
        run = self.run
        if run is None:
            raise RuntimeError("begin() must supply the video's seed")
        self._update_prompt(prompt)
        if run.noise_offset == run.noise.shape[1]:
            run.noise = self._noise_page()
            run.noise_offset = 0
        noisy_input = run.noise[:, run.noise_offset : run.noise_offset + 3]
        for index, current_timestep in enumerate(p.denoising_step_list):
            timestep = (
                torch.ones((1, 3), device=self.device, dtype=torch.int64)
                * current_timestep
            )
            _, prediction = p.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=run.conditioning,
                timestep=timestep,
                kv_cache=p.kv_cache1,
                crossattn_cache=p.crossattn_cache,
                current_start=run.position * p.frame_seq_length,
            )
            if index < len(p.denoising_step_list) - 1:
                noisy_input = p.scheduler.add_noise(
                    prediction.flatten(0, 1),
                    torch.randn_like(prediction.flatten(0, 1)),
                    p.denoising_step_list[index + 1]
                    * torch.ones((3,), device=self.device, dtype=torch.long),
                ).unflatten(0, prediction.shape[:2])
        p.generator(
            noisy_image_or_video=prediction,
            conditional_dict=run.conditioning,
            timestep=torch.ones_like(timestep) * p.args.context_noise,
            kv_cache=p.kv_cache1,
            crossattn_cache=p.crossattn_cache,
            current_start=run.position * p.frame_seq_length,
        )
        run.position += 3
        run.noise_offset += 3
        return prediction

    def _update_prompt(self, prompt: str) -> None:
        run = self.run
        if run is None:
            raise RuntimeError("begin() must supply the video's seed")
        if prompt != run.prompt:
            run.conditioning = self.pipe.text_encoder(text_prompts=[prompt])
            for cache in self.pipe.crossattn_cache:
                cache["is_init"] = False
            run.prompt = prompt

    def decode(self, latents):
        import torch

        video = self.pipe.vae.decode_to_pixel(latents, use_cache=True)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if not torch.isfinite(video).all():
            raise RuntimeError("The native decoder produced non-finite video")
        # Match the native launcher's torchvision uint8 conversion before H.264.
        return (video[0].permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()

    def step(self, prompt: str) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.decode(self.denoise(prompt))
