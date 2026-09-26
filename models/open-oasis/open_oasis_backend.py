"""Faithful one-frame sampler built from the pinned Open-Oasis implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from open_oasis_types import OpenOasisConfig

SCALING_FACTOR = 0.07843137255


class OpenOasisBackend:
    def __init__(
        self, config: OpenOasisConfig, model_path: Path, vae_path: Path
    ) -> None:
        import torch
        from dit import DiT_models
        from safetensors.torch import load_model
        from utils import sigmoid_beta_schedule
        from vae import VAE_models

        if not torch.cuda.is_available():
            raise RuntimeError("Open-Oasis requires CUDA")
        self.torch = torch
        self.config = config
        self.device = torch.device("cuda:0")
        self.model = DiT_models["DiT-S/2"]()
        load_model(self.model, str(model_path))
        self.model.to(self.device).eval()
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        load_model(self.vae, str(vae_path))
        self.vae.to(self.device).eval()
        betas = sigmoid_beta_schedule(1000).float().to(self.device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).reshape(1000, 1, 1, 1)
        self.noise_range = torch.linspace(
            -1, 999, config.ddim_steps + 1, device=self.device
        )
        self.latents: Any = None
        self.actions: Any = None
        self.generator: Any = None

    def reset(self, frames: np.ndarray, seed: int) -> None:
        torch = self.torch
        tensor = (
            torch.from_numpy(np.array(frames, copy=True))
            .to(self.device)
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            latent = self.vae.encode(tensor * 2 - 1).mean * SCALING_FACTOR
        h, w = tensor.shape[-2:]
        self.latents = (
            latent.reshape(
                1, len(frames), h // self.vae.patch_size, w // self.vae.patch_size, -1
            )
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        self.actions = torch.zeros((1, len(frames), 25), device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def generate_one(self, action: np.ndarray) -> np.ndarray:
        torch = self.torch
        if self.latents is None:
            raise RuntimeError("Open-Oasis context is not initialized")
        next_action = torch.from_numpy(action).to(self.device).reshape(1, 1, 25)
        self.actions = torch.cat((self.actions, next_action), dim=1)
        noise = torch.randn(
            (1, 1, *self.latents.shape[-3:]),
            generator=self.generator,
            device=self.device,
        ).clamp(-20, 20)
        self.latents = torch.cat((self.latents, noise), dim=1)

        for noise_idx in reversed(range(1, self.config.ddim_steps + 1)):
            length = self.latents.shape[1]
            t_ctx = torch.full(
                (1, length - 1), 14, dtype=torch.long, device=self.device
            )
            t_last = torch.full(
                (1, 1),
                self.noise_range[noise_idx],
                dtype=torch.long,
                device=self.device,
            )
            t_next_last = torch.full(
                (1, 1),
                self.noise_range[noise_idx - 1],
                dtype=torch.long,
                device=self.device,
            )
            t_next_last = torch.where(t_next_last < 0, t_last, t_next_last)
            t = torch.cat((t_ctx, t_last), dim=1)[:, -self.config.context_frames :]
            t_next = torch.cat((t_ctx, t_next_last), dim=1)[
                :, -self.config.context_frames :
            ]
            current = self.latents[:, -self.config.context_frames :].clone()
            actions = self.actions[:, -self.config.context_frames :]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                velocity = self.model(current, t, actions)
            alpha = self.alphas_cumprod[t]
            start = alpha.sqrt() * current - (1 - alpha).sqrt() * velocity
            predicted_noise = ((1 / alpha).sqrt() * current - start) / (
                1 / alpha - 1
            ).sqrt()
            alpha_next = self.alphas_cumprod[t_next].clone()
            alpha_next[:, :-1] = 1
            if noise_idx == 1:
                alpha_next[:, -1:] = 1
            prediction = (
                alpha_next.sqrt() * start + predicted_noise * (1 - alpha_next).sqrt()
            )
            self.latents[:, -1:] = prediction[:, -1:]

        latent = (
            self.latents[:, -1]
            .permute(0, 2, 3, 1)
            .reshape(1, -1, self.latents.shape[2])
        )
        # Upstream decodes in float32 (unlike its half-precision encoder and DiT).
        with torch.inference_mode():
            frame = (self.vae.decode(latent / SCALING_FACTOR) + 1) / 2
        self.latents = self.latents[:, -self.config.context_frames :]
        self.actions = self.actions[:, -self.config.context_frames :]
        return frame[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
