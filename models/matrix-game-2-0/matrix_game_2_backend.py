"""Run the upstream Matrix-Game-2.0 causal rollout one native chunk at a time."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

LATENTS_PER_CHUNK = 3
LATENT_HEIGHT = 44
LATENT_WIDTH = 80
FRAME_HEIGHT = 352
FRAME_WIDTH = 640
FIRST_CHUNK_FRAMES = 9
FRAMES_PER_CHUNK = 12

_KEYBOARD_VECTOR_ORDER = ("w", "s", "a", "d")
_NATIVE_CAMERA_SCALE = 0.1


def _keyboard_vector(pressed_keys: tuple[str, ...]) -> tuple[float, ...]:
    unsupported_keys = set(pressed_keys).difference(_KEYBOARD_VECTOR_ORDER)
    if unsupported_keys:
        raise ValueError(
            f"unsupported Matrix keyboard keys: {sorted(unsupported_keys)}"
        )
    return tuple(float(key in pressed_keys) for key in _KEYBOARD_VECTOR_ORDER)


def _mouse_vector(pitch: float, yaw: float) -> tuple[float, float]:
    if not -1.0 <= pitch <= 1.0 or not -1.0 <= yaw <= 1.0:
        raise ValueError("Matrix pitch and yaw must remain in [-1, 1]")
    return pitch * _NATIVE_CAMERA_SCALE, yaw * _NATIVE_CAMERA_SCALE


@dataclass(frozen=True)
class ChunkAction:
    """Hold the universal checkpoint controls sampled for one chunk."""

    pressed_keys: tuple[str, ...]
    pitch: float
    yaw: float


class MatrixGame2Backend:
    """Own model weights and the active upstream autoregressive cache state."""

    def __init__(
        self,
        *,
        source_path: Path,
        model_path: Path,
        checkpoint_file: str,
        max_latent_frames: int,
    ) -> None:
        self._source_path = source_path
        self._model_path = model_path
        self._checkpoint_file = checkpoint_file
        self._max_latent_frames = max_latent_frames
        self._current_start_frame = 0
        self._conditional_dict: dict[str, Any] | None = None
        self._sampled_noise: Any = None
        self._vae_cache: list[Any] | None = None

        modules = self._load_upstream_modules(source_path)
        self._torch = modules["torch"]
        self._rearrange = modules["rearrange"]
        self._cond_current = modules["cond_current"]
        self._zero_vae_cache = modules["zero_vae_cache"]
        self._set_seed = modules["set_seed"]
        self._transforms = modules["transforms"]

        torch = self._torch
        if not torch.cuda.is_available():
            raise RuntimeError("Matrix-Game-2.0 requires a CUDA accelerator")
        self._device = torch.device("cuda")
        self._weight_dtype = torch.bfloat16

        inference_config = modules["OmegaConf"].load(
            source_path / "configs/inference_yaml/inference_universal.yaml"
        )
        inference_config.model_kwargs.model_config = str(
            source_path / "configs/distilled_model/universal"
        )
        if str(inference_config.mode) != "universal":
            raise ValueError("Matrix adapter requires the universal inference mode")
        if int(inference_config.num_frame_per_block) != LATENTS_PER_CHUNK:
            raise ValueError(
                f"Matrix universal inference must use {LATENTS_PER_CHUNK} latents per chunk"
            )

        generator = modules["WanDiffusionWrapper"](
            **getattr(inference_config, "model_kwargs", {}),
            is_causal=True,
        )
        vae_decoder = modules["VAEDecoderWrapper"]()
        vae_weights = torch.load(
            model_path / "Wan2.1_VAE.pth",
            map_location="cpu",
            weights_only=True,
        )
        decoder_weights = {
            key: value
            for key, value in vae_weights.items()
            if "decoder." in key or "conv2" in key
        }
        vae_decoder.load_state_dict(decoder_weights)
        vae_decoder.to(self._device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()

        pipeline = modules["CausalInferenceStreamingPipeline"](
            inference_config,
            generator=generator,
            vae_decoder=vae_decoder,
        )
        checkpoint = model_path / checkpoint_file
        pipeline.generator.load_state_dict(modules["load_file"](checkpoint))
        pipeline = pipeline.to(device=self._device, dtype=self._weight_dtype)
        pipeline.vae_decoder.to(torch.float16)
        pipeline.requires_grad_(False)
        pipeline.eval()
        if int(pipeline.local_attn_size) != 6:
            raise ValueError(
                "the universal distilled checkpoint must retain local_attn_size=6"
            )

        vae = modules["get_wanx_vae_wrapper"](model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(self._device, self._weight_dtype)

        self._config = inference_config
        self._pipeline = pipeline
        self._vae = vae
        self._frame_process = self._transforms.Compose(
            [
                self._transforms.Resize(
                    size=(FRAME_HEIGHT, FRAME_WIDTH), antialias=True
                ),
                self._transforms.ToTensor(),
                self._transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )
        logger.info(
            "Matrix-Game-2.0 upstream backend ready",
            checkpoint=str(checkpoint),
            local_attention_frames=int(pipeline.local_attn_size),
            latent_frames_per_chunk=LATENTS_PER_CHUNK,
        )

    @property
    def completed_chunks(self) -> int:
        """Return the number of chunks already committed to the active caches."""
        return self._current_start_frame // LATENTS_PER_CHUNK

    def reset(self, image: Image.Image, seed: int) -> None:
        """Initialize official image conditioning, noise, and empty causal caches."""
        torch = self._torch
        self._set_seed(seed)
        image = self._resize_crop(image, FRAME_HEIGHT, FRAME_WIDTH)
        image_tensor = self._frame_process(image)[None, :, None, :, :].to(
            dtype=self._weight_dtype,
            device=self._device,
        )

        padding_frames = 4 * (self._max_latent_frames - 1)
        padding_video = torch.zeros_like(image_tensor).repeat(
            1,
            1,
            padding_frames,
            1,
            1,
        )
        image_condition = torch.concat([image_tensor, padding_video], dim=2)
        image_condition = self._vae.encode(
            image_condition,
            device=self._device,
            tiled=True,
            tile_size=[44, 80],
            tile_stride=[23, 38],
        ).to(self._device)
        mask_condition = torch.ones_like(image_condition)
        mask_condition[:, :, 1:] = 0
        condition_concat = torch.cat(
            [mask_condition[:, :4], image_condition],
            dim=1,
        )
        visual_context = self._vae.clip.encode_video(image_tensor)

        rgb_action_frames = (self._max_latent_frames - 1) * 4 + 1
        self._conditional_dict = {
            "cond_concat": condition_concat.to(
                device=self._device,
                dtype=self._weight_dtype,
            ),
            "visual_context": visual_context.to(
                device=self._device,
                dtype=self._weight_dtype,
            ),
            "mouse_cond": torch.zeros(
                (1, rgb_action_frames, 2),
                device=self._device,
                dtype=self._weight_dtype,
            ),
            "keyboard_cond": torch.zeros(
                (1, rgb_action_frames, 4),
                device=self._device,
                dtype=self._weight_dtype,
            ),
        }
        self._sampled_noise = torch.randn(
            (
                1,
                16,
                self._max_latent_frames,
                LATENT_HEIGHT,
                LATENT_WIDTH,
            ),
            device=self._device,
            dtype=self._weight_dtype,
        )
        self._vae_cache = [None for _ in self._zero_vae_cache]
        self._current_start_frame = 0
        self._initialize_upstream_caches()

    def generate_chunk(self, action: ChunkAction) -> np.ndarray:
        """Advance the exact upstream denoise/cache/decode sequence by one chunk."""
        torch = self._torch
        conditional_dict = self._conditional_dict
        sampled_noise = self._sampled_noise
        vae_cache = self._vae_cache
        if conditional_dict is None or sampled_noise is None or vae_cache is None:
            raise RuntimeError("select a Matrix starting image before generating")
        if self._current_start_frame + LATENTS_PER_CHUNK > self._max_latent_frames:
            raise RuntimeError("Matrix rollout has reached its official latent horizon")
        start = self._current_start_frame
        noisy_input = sampled_noise[:, :, start : start + LATENTS_PER_CHUNK]
        replacement = {
            "keyboard": torch.tensor(
                _keyboard_vector(action.pressed_keys),
                device=self._device,
                dtype=self._weight_dtype,
            ),
            "mouse": torch.tensor(
                _mouse_vector(action.pitch, action.yaw),
                device=self._device,
                dtype=self._weight_dtype,
            ),
        }
        chunk_condition, self._conditional_dict = self._cond_current(
            conditional_dict,
            start,
            LATENTS_PER_CHUNK,
            replace=replacement,
            mode="universal",
        )

        with torch.inference_mode():
            for index, current_timestep in enumerate(
                self._pipeline.denoising_step_list
            ):
                timestep = (
                    torch.ones(
                        (1, LATENTS_PER_CHUNK),
                        device=self._device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )
                _, denoised_prediction = self._pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=chunk_condition,
                    timestep=timestep,
                    kv_cache=self._pipeline.kv_cache1,
                    kv_cache_mouse=self._pipeline.kv_cache_mouse,
                    kv_cache_keyboard=self._pipeline.kv_cache_keyboard,
                    crossattn_cache=self._pipeline.crossattn_cache,
                    current_start=start * self._pipeline.frame_seq_length,
                )
                if index < len(self._pipeline.denoising_step_list) - 1:
                    next_timestep = self._pipeline.denoising_step_list[index + 1]
                    flat_prediction = self._rearrange(
                        denoised_prediction,
                        "b c f h w -> (b f) c h w",
                    )
                    noisy_input = self._pipeline.scheduler.add_noise(
                        flat_prediction,
                        torch.randn_like(flat_prediction),
                        next_timestep
                        * torch.ones(
                            (LATENTS_PER_CHUNK,),
                            device=self._device,
                            dtype=torch.long,
                        ),
                    )
                    noisy_input = self._rearrange(
                        noisy_input,
                        "(b f) c h w -> b c f h w",
                        b=denoised_prediction.shape[0],
                    )

            context_timestep = torch.ones_like(timestep) * self._config.context_noise
            self._pipeline.generator(
                noisy_image_or_video=denoised_prediction,
                conditional_dict=chunk_condition,
                timestep=context_timestep,
                kv_cache=self._pipeline.kv_cache1,
                kv_cache_mouse=self._pipeline.kv_cache_mouse,
                kv_cache_keyboard=self._pipeline.kv_cache_keyboard,
                crossattn_cache=self._pipeline.crossattn_cache,
                current_start=start * self._pipeline.frame_seq_length,
            )
            video, updated_vae_cache = self._pipeline.vae_decoder(
                denoised_prediction.transpose(1, 2).half(),
                *vae_cache,
            )

        self._vae_cache = updated_vae_cache
        self._current_start_frame += LATENTS_PER_CHUNK
        frames = self._rearrange(video, "B T C H W -> B T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).to(torch.uint8).cpu().numpy()[0]
        )
        expected_frames = (
            FIRST_CHUNK_FRAMES if self.completed_chunks == 1 else FRAMES_PER_CHUNK
        )
        if frames.shape != (expected_frames, FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                "unexpected Matrix causal decode shape: "
                f"expected {(expected_frames, FRAME_HEIGHT, FRAME_WIDTH, 3)}, "
                f"got {frames.shape}"
            )
        return np.ascontiguousarray(frames)

    def end_rollout(self) -> None:
        """Release image conditioning and all session-scoped causal caches."""
        self._conditional_dict = None
        self._sampled_noise = None
        self._vae_cache = None
        self._current_start_frame = 0
        self._pipeline.kv_cache1 = None
        self._pipeline.kv_cache_mouse = None
        self._pipeline.kv_cache_keyboard = None
        self._pipeline.crossattn_cache = None
        self._torch.cuda.empty_cache()

    def _initialize_upstream_caches(self) -> None:
        """Allocate caches with the upstream checkpoint's native local window."""
        pipeline = self._pipeline
        pipeline.kv_cache1 = None
        pipeline.kv_cache_mouse = None
        pipeline.kv_cache_keyboard = None
        pipeline.crossattn_cache = None
        pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=self._weight_dtype,
            device=self._device,
        )
        pipeline._initialize_kv_cache_mouse_and_keyboard(
            batch_size=1,
            dtype=self._weight_dtype,
            device=self._device,
        )
        pipeline._initialize_crossattn_cache(
            batch_size=1,
            dtype=self._weight_dtype,
            device=self._device,
        )

    @staticmethod
    def _resize_crop(image: Image.Image, height: int, width: int) -> Image.Image:
        """Apply the same centered aspect-ratio crop as official streaming inference."""
        source_width, source_height = image.size
        if source_height / source_width > height / width:
            new_width = source_width
            new_height = int(new_width * height / width)
        else:
            new_height = source_height
            new_width = int(new_height * width / height)
        left = (source_width - new_width) / 2
        top = (source_height - new_height) / 2
        right = (source_width + new_width) / 2
        bottom = (source_height + new_height) / 2
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _load_upstream_modules(source_path: Path) -> dict[str, Any]:
        """Import only modules exercised by the official streaming inference path."""
        source = str(source_path)
        if source not in sys.path:
            sys.path.insert(0, source)

        import torch
        from einops import rearrange
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from torchvision.transforms import v2

        causal_inference = importlib.import_module("pipeline.causal_inference")
        constants = importlib.import_module("demo_utils.constant")
        misc = importlib.import_module("utils.misc")
        vae_block = importlib.import_module("demo_utils.vae_block3")
        wan_wrapper = importlib.import_module("utils.wan_wrapper")
        wanx_vae = importlib.import_module("wan.vae.wanx_vae")

        expected_root = source_path.resolve()
        for module in (
            causal_inference,
            constants,
            misc,
            vae_block,
            wan_wrapper,
            wanx_vae,
        ):
            module_file = Path(module.__file__).resolve()
            if expected_root not in module_file.parents:
                raise RuntimeError(
                    f"upstream module {module.__name__} resolved outside {source_path}: "
                    f"{module_file}"
                )

        return {
            "torch": torch,
            "rearrange": rearrange,
            "OmegaConf": OmegaConf,
            "load_file": load_file,
            "transforms": v2,
            "CausalInferenceStreamingPipeline": causal_inference.CausalInferenceStreamingPipeline,
            "cond_current": causal_inference.cond_current,
            "zero_vae_cache": constants.ZERO_VAE_CACHE,
            "set_seed": misc.set_seed,
            "VAEDecoderWrapper": vae_block.VAEDecoderWrapper,
            "WanDiffusionWrapper": wan_wrapper.WanDiffusionWrapper,
            "get_wanx_vae_wrapper": wanx_vae.get_wanx_vae_wrapper,
        }
