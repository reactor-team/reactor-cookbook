"""Run SolarWM's native Stage2 NFE4 sampler one causal chunk at a time."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from reactor_runtime import UploadedFile


@dataclass(frozen=True)
class BackendSettings:
    """Locate pinned SolarWM source, weights, and runtime scratch space."""

    source_path: Path
    upstream_config: Path
    base_path: Path
    checkpoint_path: Path
    runtime_root: Path


def _expose_upstream_package(source_path: Path) -> None:
    """Let the ``solarwm.py`` entry module resolve upstream SolarWM subpackages."""
    package_path = source_path / "src" / "solarwm"
    if not package_path.is_dir():
        raise RuntimeError(f"SolarWM package is missing: {package_path}")
    entry_module = sys.modules.get("solarwm")
    if entry_module is None:
        raise RuntimeError("SolarWM Reactor entry module is not loaded")
    search_locations = [str(package_path)]
    entry_module.__path__ = search_locations
    if entry_module.__spec__ is not None:
        entry_module.__spec__.submodule_search_locations = search_locations


class SolarWMBackend:
    """Preserve SolarWM self-KV, cross-attention, and VAE caches across chunks."""

    def __init__(self, settings: BackendSettings) -> None:
        _expose_upstream_package(settings.source_path)
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
        self.first_latent = None
        self.condition = None
        self.generator = None
        self.kv_cache = None
        self.crossattn_cache = None
        self.chunk_index = 0

    def reset(self, seed: int, image: UploadedFile, prompt: str) -> None:
        """Encode a fresh uploaded anchor and allocate native rolling caches."""
        torch = self.torch
        self.end_session()
        pixels = _prepare_image(image, width=864, height=480)
        pixel_tensor = torch.from_numpy(pixels).to(self.device, dtype=torch.float32)
        pixel_tensor = (pixel_tensor.permute(2, 0, 1)[None, :, None] / 127.5) - 1.0
        with torch.no_grad():
            self.first_latent = self.provider.vae.encode(pixel_tensor).to(
                torch.bfloat16
            )
            self.condition = self.provider.text_encoder([prompt])
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.kv_cache = self.provider.allocate_kv_cache(
            1, dtype=torch.bfloat16, device=self.device
        )
        self.crossattn_cache = self.provider.allocate_crossattn_cache(
            1, dtype=torch.bfloat16, device=self.device
        )
        clear = getattr(self.provider.vae.module, "clear_cache", None)
        if callable(clear):
            clear()
        self.chunk_index = 0

    def generate_chunk(self, relative_c2ws: np.ndarray) -> np.ndarray:
        """Generate and causally decode one native three-latent SolarWM chunk."""
        torch = self.torch
        if any(
            value is None
            for value in (
                self.first_latent,
                self.condition,
                self.generator,
                self.kv_cache,
                self.crossattn_cache,
            )
        ):
            raise RuntimeError("SolarWM rollout has not been reset")
        from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens
        from solarwm.backends.wan22.runtime.stage2 import _generation_steps

        chunk, frame_tokens = 3, int(self.config["model"]["frame_sequence_length"])
        start = self.chunk_index * chunk
        shape = (1, chunk, 48, 30, 54)
        latents = self.provider._noise(shape, self.generator)
        if start == 0:
            latents[:, :1] = self.first_latent
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
                    self.condition,
                    camera,
                    expand_timesteps_to_tokens(timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start * frame_tokens,
                    cache_start=0,
                    cache_update_policy="none",
                )
                x0 = self.provider.diffusion.flow_to_x0(latents, flow, timestep)
            if start == 0:
                x0[:, :1] = self.first_latent
            if index + 1 < len(steps):
                next_t = torch.full(
                    (1, chunk), float(steps[index + 1].item()), device=self.device
                )
                if start == 0:
                    next_t[:, 0] = 0.0
                noise = self.provider._noise(tuple(x0.shape), self.generator)
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
                    latents[:, :1] = self.first_latent
            else:
                latents = x0
        if not bool(torch.isfinite(latents).all().item()):
            raise RuntimeError(
                f"SolarWM chunk {self.chunk_index + 1} contains non-finite latents"
            )
        zeros = torch.zeros((1, chunk), device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.provider.diffusion(
                latents,
                self.condition,
                camera,
                expand_timesteps_to_tokens(zeros, frame_tokens),
                sequence_length=chunk * frame_tokens,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=start * frame_tokens,
                cache_start=0,
                cache_update_policy="commit_detached",
            )
            decoded = self.provider.vae.decode(latents, use_cache=True)
        self.chunk_index += 1
        frames = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1)
        return frames.byte().cpu().numpy()

    def end_session(self) -> None:
        """Drop rollout caches without unloading shared model weights."""
        clear = getattr(self.provider.vae.module, "clear_cache", None)
        if callable(clear):
            clear()
        self.first_latent = self.condition = self.generator = None
        self.kv_cache = self.crossattn_cache = None
        self.chunk_index = 0


def _prepare_image(upload: UploadedFile, *, width: int, height: int) -> np.ndarray:
    """Apply SolarWM's bilinear resize and center-crop image path."""
    with Image.open(io.BytesIO(upload.data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        scale = max(height / image.height, width / image.width)
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.BILINEAR)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        return np.array(
            image.crop((left, top, left + width, top + height)),
            dtype=np.uint8,
            copy=True,
        )


def _camera_tokens(
    c2ws: np.ndarray, frame_tokens: int, device: object
) -> dict[str, object]:
    """Convert first-pose-relative C2W matrices to SolarWM W2C and intrinsic tokens."""
    import torch

    c2w = torch.as_tensor(c2ws, dtype=torch.float32, device=device)
    viewmats = torch.linalg.inv(c2w)[None].repeat_interleave(frame_tokens, dim=1)
    intrinsic = torch.tensor(
        [
            [969.6969696969696 / 1920.0, 0.0, 0.5],
            [0.0, 969.6969696969696 / 1080.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return {
        "viewmats": viewmats,
        "K": intrinsic[None, None].expand(1, 3 * frame_tokens, 3, 3),
    }
