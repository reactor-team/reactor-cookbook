"""Run Zing 0.5 incrementally while preserving its released cache semantics."""

from __future__ import annotations

import sys
from typing import Iterable

import numpy as np
import random
import torch

from zing_assets import ZingAdapterConfig
from zing_model import LOCAL_ATTN_SIZE, SINK_SIZE, action_values


class ZingBackend:
    """Own model weights and one session-scoped autoregressive rollout."""

    def __init__(self, adapter: ZingAdapterConfig) -> None:
        sys.path.insert(0, str(adapter.source_path / "src"))
        from zing_v0_5.config import load_config, with_cache_window
        from zing_v0_5.pipeline import InferencePipeline

        upstream_config = load_config(adapter.source_path / "config" / "zing.yaml")
        upstream_config = with_cache_window(upstream_config, LOCAL_ATTN_SIZE, SINK_SIZE)
        self.config = upstream_config
        self.adapter = adapter
        self.pipeline = InferencePipeline(
            upstream_config,
            adapter.asset_path / "pretrained",
            adapter.asset_path / "generator" / "model.pt",
        )
        self.device = self.pipeline.device
        # The declared B200 recipe keeps both encoders resident between steps.
        self.pipeline.text_encoder.to(self.device)
        self.pipeline.vae.to(self.device)
        self.cache = None
        self.context = None
        self.prompt = ""
        self.latents: list[torch.Tensor] = []
        self.pixel_frames_emitted = 0
        self.generator: torch.Generator | None = None

    @torch.inference_mode()
    def reset(self, *, image: np.ndarray | None, prompt: str, seed: int) -> None:
        self.end_session()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.cache = self.pipeline.generator.make_kv_cache()
        self.context = self._encode_prompt(prompt)
        self.prompt = prompt
        self.pixel_frames_emitted = 0
        if image is not None:
            latent = self._encode_image(image)
            self.latents = [latent]
            self._commit(latent, self._zero_action(1), prompt_switch=False)
            self.pixel_frames_emitted = 1
        else:
            self.latents = []

    @torch.inference_mode()
    def generate_chunk(self, *, prompt: str, pressed_keys: Iterable[str]) -> np.ndarray:
        from zing_v0_5.scheduler import DmdScheduler

        if self.cache is None or self.context is None or self.generator is None:
            raise RuntimeError("Zing rollout has not been initialized")
        prompt_switch = prompt != self.prompt
        if prompt_switch:
            self.context = self._encode_prompt(prompt)
            self.prompt = prompt
        latent_frames = (
            1 if not self.latents else self.config.inference.frames_per_block
        )
        shape = (
            1,
            latent_frames,
            self.config.vae.z_dim,
            self.adapter.height // self.config.vae.spatial_scale,
            self.adapter.width // self.config.vae.spatial_scale,
        )
        current = torch.randn(
            shape, generator=self.generator, device=self.device, dtype=torch.bfloat16
        )
        action = self._action(pressed_keys, latent_frames)
        token_count = (
            latent_frames
            * (shape[3] // self.config.generator.patch_size[1])
            * (shape[4] // self.config.generator.patch_size[2])
        )
        self.cache.reserve(token_count)
        scheduler = DmdScheduler(self.config.inference).to(self.device)
        for step_index, timestep in enumerate(scheduler.timesteps):
            step_time = timestep * torch.ones(
                (1, latent_frames), device=self.device, dtype=torch.float32
            )
            prediction = self.pipeline._model_flow(
                current,
                step_time,
                self.context,
                self.cache,
                "active",
                action,
                prompt_switch and step_index == 0,
            )
            current, _ = scheduler.step(prediction, current)
        self._commit(current, action, prompt_switch=False)
        self.latents.append(current.detach())
        return self._decode_new_frames()

    def end_session(self) -> None:
        self.cache = None
        self.context = None
        self.latents = []
        self.generator = None
        self.pixel_frames_emitted = 0
        torch.cuda.empty_cache()

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        context, lengths = self.pipeline.text_encoder.encode([prompt])
        return context.to(self.device, dtype=torch.bfloat16), lengths.to(self.device)

    def _encode_image(self, pixels: np.ndarray) -> torch.Tensor:
        frames = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        return self.pipeline.vae.encode(frames).to(self.device, dtype=torch.bfloat16)

    def _action(self, pressed: Iterable[str], frames: int) -> torch.Tensor:
        values = torch.tensor(
            action_values(pressed), device=self.device, dtype=torch.float32
        )
        return values.view(1, 1, 1, 8).expand(1, frames, 4, 8).clone()

    def _zero_action(self, frames: int) -> torch.Tensor:
        return torch.zeros((1, frames, 4, 8), device=self.device, dtype=torch.float32)

    def _commit(
        self, latent: torch.Tensor, action: torch.Tensor, prompt_switch: bool
    ) -> None:
        zero = torch.zeros(
            (1, latent.shape[1]), device=self.device, dtype=torch.float32
        )
        self.pipeline._model_flow(
            latent, zero, self.context, self.cache, "final", action, prompt_switch
        )

    def _decode_new_frames(self) -> np.ndarray:
        # Pinned WanVAE.decode clears its temporal cache on every call. Decode
        # the bounded rollout history to retain the exact native boundary context;
        # decoding only the new latents would change the first frames of a chunk.
        all_latents = torch.cat(self.latents, dim=1)
        video = self.pipeline.vae.decode(all_latents)
        new = (video[:, self.pixel_frames_emitted :] * 0.5 + 0.5).clamp(0, 1)
        frames = new[0].permute(0, 2, 3, 1).mul(255).to(torch.uint8).cpu().numpy()
        self.pixel_frames_emitted = int(video.shape[1])
        return frames
