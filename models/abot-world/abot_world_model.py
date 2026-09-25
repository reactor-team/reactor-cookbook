"""ABot-World weights, causal caches and native chunk generation without Reactor."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from abot_world_assets import (
    ABotWorldConfig,
    build_upstream_config,
    load_upstream_modules,
)

logger = logging.getLogger(__name__)
_EXPECTED_LATENTS_PER_CHUNK = 3
_EXPECTED_LOCAL_CACHE_LATENTS = 21


@contextmanager
def _materialized_image(
    source: Path | bytes, suffix: str, weights_root: Path
) -> Iterator[Path]:
    """Keep encoded bytes on the weights volume only while the native encoder needs them."""
    if isinstance(source, Path):
        yield source
        return
    temporary_root = weights_root / "temporary-images"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="abot-world-", suffix=suffix, dir=temporary_root
    ) as temporary:
        temporary.write(source)
        temporary.flush()
        yield Path(temporary.name)


class _FrameCapture:
    """Collect frames from the upstream decoder's writer interface."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def append_data(self, frame: np.ndarray) -> None:
        """Append one decoded RGB frame."""
        self.frames.append(np.asarray(frame))


@dataclass(frozen=True)
class ABotAnchor:
    """Image and seed supplied until the new world is acknowledged."""

    image: Path | bytes
    suffix: str
    seed: int


@dataclass(frozen=True)
class ABotInput:
    """One world's optional anchor, current prompt and native action channels."""

    world_id: int
    anchor: ABotAnchor | None
    prompt: str
    action: dict[str, bool]


@dataclass(frozen=True)
class ABotResult:
    """Native frames and the completed step's own world and conditioning facts."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    sampled_keys: frozenset[str]
    prompt: str
    complete: bool


class NoAnchor(Exception):
    """A new world was requested without its initial image."""


class RolloutExhausted(Exception):
    """The world reached its configured chunk capacity."""


class ABotWorldModel:
    """Own the upstream pipeline, latent shape, caches and successful chunk count."""

    def __init__(self) -> None:
        self._config: ABotWorldConfig | None = None
        self._weights_root: Path | None = None
        self._modules: dict[str, Any] = {}
        self._pipeline: Any = None
        self._device: Any = None
        self._latent_shape: tuple[int, int, int, int, int] | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._active_prompt = ""

    def load(self, config: ABotWorldConfig, weights_root: Path) -> None:
        """Load native weights with application-resolved paths."""
        self._weights_root = weights_root
        modules = load_upstream_modules(config)
        torch = modules["torch"]
        if not torch.cuda.is_available():
            raise RuntimeError("ABot-World requires a CUDA accelerator")
        device = torch.device("cuda")
        upstream_config = build_upstream_config(config, modules)
        num_frame_per_block = int(upstream_config.num_frame_per_block)
        local_attn_size = int(upstream_config.model_kwargs.local_attn_size)
        if num_frame_per_block != _EXPECTED_LATENTS_PER_CHUNK:
            raise ValueError(
                "ABot-World must retain its native three-latent autoregressive chunk size"
            )
        if local_attn_size != _EXPECTED_LOCAL_CACHE_LATENTS:
            raise ValueError(
                "ABot-World must retain its native 21-latent KV cache window"
            )

        modules["set_seed"](config.seed)
        torch.set_grad_enabled(False)
        vae = modules["create_vae"](upstream_config)
        pipeline = modules["pipeline_type"](upstream_config, device=device, vae=vae)
        try:
            modules["replace_norms"](pipeline.generator.model)
            modules["replace_rope"]()
            logger.info("ABot-World upstream Helios kernels enabled")
        except Exception as error:  # noqa: BLE001 - optional upstream kernels may be unavailable.
            logger.warning("ABot-World Helios kernels unavailable: %s", error)
        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        if pipeline.encoder is not None:
            pipeline.encoder.to(device=device)
        pipeline.torch_dtype = torch.bfloat16

        vae_for_shape = (
            pipeline.encoder if pipeline.encoder is not None else pipeline.vae
        )
        upsampling = int(getattr(vae_for_shape, "upsampling_factor", 16))
        latent_channels = int(vae_for_shape.z_dim)
        self._latent_shape = (
            1,
            num_frame_per_block,
            latent_channels,
            config.height // upsampling,
            config.width // upsampling,
        )
        self._config = config
        self._modules = modules
        self._pipeline = pipeline
        self._device = device
        logger.info(
            "ABot-World model ready: three latent frames, 21-latent cache, max_chunks=%s",
            config.max_chunks,
        )

    def generate(self, input: ABotInput) -> ABotResult:
        """Apply an explicit new world or step the native rolling caches once."""
        config = self._require_config()
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor("A fresh ABot world requires an anchor image")
            self._reset_rollout(
                input.anchor.image, input.anchor.suffix, input.prompt, input.anchor.seed
            )
            self._world_id = input.world_id
        if self._chunk_index >= config.max_chunks:
            raise RolloutExhausted("Reset ABot-World before requesting another chunk")
        frames = self._generate_chunk(input.prompt, input.action)
        self._active_prompt = input.prompt
        self._chunk_index += 1
        return ABotResult(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            sampled_keys=frozenset(
                key for key, active in input.action.items() if active
            ),
            prompt=self._active_prompt,
            complete=self._chunk_index >= config.max_chunks,
        )

    def reset(self) -> None:
        """Reset reusable native caches and forget the session's world."""
        self._reset_upstream_stream()
        self._world_id = None
        self._chunk_index = 0
        self._active_prompt = ""

    def _reset_rollout(
        self,
        selected_input: Path | bytes,
        suffix: str,
        prompt: str,
        seed: int,
    ) -> None:
        """Initialize upstream conditions and rolling caches for a fresh world."""
        pipeline = self._require_pipeline()
        config = self._require_config()
        torch = self._modules["torch"]
        self._modules["set_seed"](seed)
        pipeline.set_prompts([prompt], device=self._device)
        empty_ref_dir = config.checkpoint.path / "unused-reference-slots"
        pipeline.set_ref_latent_mask_from_exists_paths(
            ref_dir=str(empty_ref_dir),
            device=self._device,
        )
        pipeline.reset_stream(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self._device,
            initial_latent=None,
        )
        with _materialized_image(
            selected_input, suffix, self._weights_root
        ) as image_path:
            pipeline.set_first_frame_latent(
                str(image_path),
                height=config.height,
                width=config.width,
                device=self._device,
            )
        self._active_prompt = prompt
        self._chunk_index = 0

    def _generate_chunk(self, prompt: str, action: dict[str, bool]) -> np.ndarray:
        """Run one upstream autoregressive block and cached VAE decode."""
        pipeline = self._require_pipeline()
        config = self._require_config()
        latent_shape = self._latent_shape
        if latent_shape is None:
            raise RuntimeError("ABot-World latent shape was not initialized")
        torch = self._modules["torch"]
        if prompt != self._active_prompt:
            pipeline.set_prompts([prompt], device=self._device)
        pipeline.set_act(
            action,
            height=config.height,
            width=config.width,
            num_frames=latent_shape[1],
            device=self._device,
        )
        noise = torch.randn(latent_shape, device=self._device, dtype=torch.bfloat16)
        latent_block = pipeline.generate_next_block(noise)
        capture = _FrameCapture()
        pipeline.decode_block_and_write(latent_block, capture)
        return self._normalize_frames(capture.frames)

    def _normalize_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return a contiguous native-resolution uint8 RGB frame batch."""
        config = self._require_config()
        if not frames:
            raise RuntimeError("ABot-World decoded an empty chunk")
        normalized: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.shape != (config.height, config.width, 3):
                raise RuntimeError(
                    f"ABot-World frame {index} has shape {array.shape}; expected "
                    f"{(config.height, config.width, 3)}"
                )
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            normalized.append(np.ascontiguousarray(array))
        return np.ascontiguousarray(np.stack(normalized))

    def _reset_upstream_stream(self) -> None:
        """Reset reusable upstream caches and decoder state after a session."""
        pipeline = self._pipeline
        if pipeline is None or self._device is None:
            return
        if pipeline.kv_cache1 is None:
            return
        torch = self._modules["torch"]
        pipeline.reset_stream(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self._device,
            initial_latent=None,
        )
        vae_model = getattr(pipeline.vae, "model", None)
        if vae_model is not None and hasattr(vae_model, "clear_cache"):
            vae_model.clear_cache()
        taehv = getattr(pipeline.vae, "taehv", None)
        if taehv is not None and hasattr(taehv, "reset"):
            taehv.reset()

    def _require_config(self) -> ABotWorldConfig:
        """Return loaded configuration or report an invalid lifecycle call."""
        if self._config is None:
            raise RuntimeError("ABot-World was not loaded")
        return self._config

    def _require_pipeline(self) -> Any:
        """Return the loaded upstream pipeline or report an invalid lifecycle call."""
        if self._pipeline is None:
            raise RuntimeError("ABot-World was not loaded")
        return self._pipeline
