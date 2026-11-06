"""Run SolarWM's native Stage2 NFE4 sampler one causal chunk at a time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_LATENT = (48, 30, 54)


@dataclass(frozen=True)
class BackendSettings:
    """Locate pinned SolarWM source, weights, and runtime scratch space."""

    upstream_config: Path
    base_path: Path
    checkpoint_path: Path
    runtime_root: Path


@dataclass
class _Run:
    """One world's encoded conditions, native caches and completed chunk count."""

    first_latent: Any
    condition: Any
    generator: Any
    initial_noise: Any
    kv_cache: Any
    crossattn_cache: Any
    chunk_index: int = 0


class SolarWMBackend:
    """Preserve SolarWM self-KV, cross-attention, and VAE caches across chunks."""

    def __init__(self, settings: BackendSettings) -> None:
        import torch
        from solarwm.backends.wan22.runtime.stage2 import (
            build_stage2_generation_provider,
        )
        from solarwm.config.loader import load_config

        overrides = (
            f"model.base_path={settings.base_path}",
            f"checkpoint.path={settings.checkpoint_path}",
            f"runtime.output_dir={settings.runtime_root}",
            f"data.index_root={settings.runtime_root}",
            f"data.transport.root={settings.runtime_root}",
        )
        config = load_config(settings.upstream_config, overrides).values
        self.provider = build_stage2_generation_provider(config)
        self.provider._load_role("model")
        self.torch = torch
        self.config = config
        self.device = self.provider.device
        self._run: _Run | None = None

    def reset(self, seed: int, image: np.ndarray, prompt: str, chunks: int) -> None:
        """Encode a fresh anchor, draw the horizon's noise and allocate native caches.

        ``image`` is the (3,480,864) float32 anchor in [-1,1]. As upstream
        _stage2_self_forcing_latents does, the initial noise of all ``chunks``
        is drawn once, before any chunk's re-noising draws.
        """
        torch = self.torch
        self.end_session()
        pixel_tensor = torch.from_numpy(image)[None, :, None].to(self.device)
        with torch.no_grad():
            first_latent = self.provider.vae.encode(pixel_tensor).to(torch.bfloat16)
            condition = self.provider.text_encoder([prompt])
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        initial_noise = self.provider._noise((1, 3 * int(chunks), *_LATENT), generator)
        kv_cache = self.provider.allocate_kv_cache(
            1, dtype=torch.bfloat16, device=self.device
        )
        crossattn_cache = self.provider.allocate_crossattn_cache(
            1, dtype=torch.bfloat16, device=self.device
        )
        self.provider.vae.module.clear_cache()
        self._run = _Run(
            first_latent=first_latent,
            condition=condition,
            generator=generator,
            initial_noise=initial_noise,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
        )

    def generate_chunk(self, relative_c2ws: np.ndarray) -> tuple[np.ndarray, int]:
        """Generate and causally decode one native three-latent SolarWM chunk."""
        torch = self.torch
        run = self._run
        if run is None:
            raise RuntimeError("SolarWM rollout has not been reset")
        from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens
        from solarwm.backends.wan22.runtime.stage2 import _generation_steps

        chunk, frame_tokens = 3, int(self.config["model"]["frame_sequence_length"])
        start = run.chunk_index * chunk
        if start + chunk > run.initial_noise.shape[1]:
            raise RuntimeError("SolarWM rollout horizon is exhausted")
        latents = run.initial_noise[:, start : start + chunk].clone()
        if start == 0:
            latents[:, :1] = run.first_latent
        camera = _camera_tokens(relative_c2ws, frame_tokens, self.device)
        steps = _generation_steps(self.provider)
        if len(steps) != 4:
            raise RuntimeError("SolarWM Stage2 requires its native NFE4 schedule")
        for index, step in enumerate(steps):
            timestep = torch.full((1, chunk), float(step.item()), device=self.device)
            if start == 0:
                timestep[:, 0] = 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                flow = self.provider.diffusion(
                    latents,
                    run.condition,
                    camera,
                    expand_timesteps_to_tokens(timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=run.kv_cache,
                    crossattn_cache=run.crossattn_cache,
                    current_start=start * frame_tokens,
                    cache_start=0,
                    cache_update_policy="none",
                )
                x0 = self.provider.diffusion.flow_to_x0(latents, flow, timestep)
            if start == 0:
                x0[:, :1] = run.first_latent
            if index + 1 < len(steps):
                next_t = torch.full(
                    (1, chunk), float(steps[index + 1].item()), device=self.device
                )
                if start == 0:
                    next_t[:, 0] = 0.0
                noise = self.provider._noise(tuple(x0.shape), run.generator)
                latents = (
                    self.provider.diffusion.scheduler.add_noise(
                        x0.flatten(0, 1).float(),
                        noise.flatten(0, 1).float(),
                        next_t.flatten(),
                    )
                    .unflatten(0, (1, chunk))
                    .to(torch.bfloat16)
                )
                if start == 0:
                    latents[:, :1] = run.first_latent
            else:
                latents = x0
        if not bool(torch.isfinite(latents).all().item()):
            raise RuntimeError(
                f"SolarWM chunk {run.chunk_index + 1} contains non-finite latents"
            )
        zeros = torch.zeros((1, chunk), device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.provider.diffusion(
                latents,
                run.condition,
                camera,
                expand_timesteps_to_tokens(zeros, frame_tokens),
                sequence_length=chunk * frame_tokens,
                kv_cache=run.kv_cache,
                crossattn_cache=run.crossattn_cache,
                current_start=start * frame_tokens,
                cache_start=0,
                cache_update_policy="commit_detached",
            )
            decoded = self.provider.vae.decode(latents, use_cache=True)
        # Upstream _encode_mp4 rounds to uint8 before encoding.
        frames = ((decoded[0].float().clamp(-1, 1) + 1.0) * 127.5).round()
        output = frames.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        run.chunk_index += 1
        return output, run.chunk_index

    def end_session(self) -> None:
        """Drop one world's caches without unloading shared weights."""
        self.provider.vae.module.clear_cache()
        self._run = None


def _camera_tokens(
    c2ws: np.ndarray, frame_tokens: int, device: object
) -> dict[str, object]:
    """Convert first-pose-relative C2W matrices to SolarWM W2C and intrinsic tokens.

    Follows upstream build_camera_tokens: analytic FP32 SE3 inversion on the
    CPU, each latent's view matrix repeated over its tokens, fixed intrinsics.
    """
    import torch
    from solarwm.backends.wan22.runtime.data import (
        FX_NORM,
        FY_NORM,
        _torch_invert_se3_fp32,
    )

    relative = torch.as_tensor(np.ascontiguousarray(c2ws, dtype=np.float32))
    viewmats = _torch_invert_se3_fp32(relative)
    viewmats = (
        viewmats.unsqueeze(1)
        .expand(-1, int(frame_tokens), -1, -1)
        .reshape(-1, 4, 4)
        .contiguous()
    )
    intrinsics = torch.zeros((1, 3, 3), dtype=torch.float32)
    intrinsics[:, 0, 0] = FX_NORM
    intrinsics[:, 1, 1] = FY_NORM
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0
    intrinsics = intrinsics.expand(viewmats.shape[0], -1, -1).contiguous()
    return {
        "viewmats": viewmats.unsqueeze(0).to(device),
        "K": intrinsics.unsqueeze(0).to(device),
    }
