"""Run stateful DreamX-World inference with upstream weights and rolling caches."""

from __future__ import annotations

import importlib
import io
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

if TYPE_CHECKING:
    from dreamx_camera import CameraChunk
    from dreamx_world_model import DreamXConfig

logger = logging.getLogger(__name__)

_LATENT_FRAMES_PER_CHUNK = 3
_LATENT_CHANNELS = 48
_LATENT_HEIGHT = 44
_LATENT_WIDTH = 80
_TOKENS_PER_LATENT_FRAME = 880
_EXPECTED_LOCAL_ATTENTION_FRAMES = 12


class DreamXBackend:
    """Keep upstream DreamX-World weights and causal state resident on one GPU."""

    def __init__(self, config: DreamXConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DreamX-World requires a CUDA GPU")
        self._config = config
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._pipeline = self._load_pipeline()
        self._initial_latent: torch.Tensor | None = None
        self._conditional: dict[str, torch.Tensor] | None = None
        self._conditional_prompt = ""
        self._chunk_index = 0
        self._current_start_frame = 0
        self._color_reference: np.ndarray | None = None

    @property
    def local_attention_frames(self) -> int:
        """Return the upstream rolling KV cache length in latent frames."""
        return int(self._pipeline.local_attn_size)

    def reset(self, seed: int, image: Path | bytes) -> None:
        """Start a fresh causal rollout from one image while retaining loaded weights."""
        set_seed = self._upstream_module("utils.misc").set_seed
        set_seed(seed)
        self._release_rollout()
        pixel = self._load_image_tensor(image)
        with torch.inference_mode():
            self._initial_latent = self._pipeline.vae.encode_to_latent(pixel).to(
                device=self._device,
                dtype=self._dtype,
            )
        self._pipeline._initialize_kv_cache(
            1,
            self._dtype,
            self._device,
            _LATENT_FRAMES_PER_CHUNK,
        )
        self._pipeline._initialize_crossattn_cache(
            1,
            self._dtype,
            self._device,
        )

    def generate_chunk(self, prompt: str, camera_chunk: CameraChunk) -> np.ndarray:
        """Generate and decode exactly one native three-latent-frame chunk."""
        if self._initial_latent is None or self._pipeline.kv_cache1 is None:
            raise RuntimeError("Reset DreamX-World with an image before generating")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("DreamX-World requires a non-empty prompt")
        first_chunk = self._chunk_index == 0
        camera_condition = self._camera_condition(camera_chunk)
        conditional = self._prompt_condition(prompt)

        with torch.inference_mode():
            noise = torch.randn(
                [
                    1,
                    _LATENT_FRAMES_PER_CHUNK,
                    _LATENT_CHANNELS,
                    _LATENT_HEIGHT,
                    _LATENT_WIDTH,
                ],
                device=self._device,
                dtype=self._dtype,
            )
            if first_chunk:
                noise[:, 0] = self._initial_latent[:, 0]
            denoised = self._denoise_one_block(
                noise,
                conditional,
                camera_condition,
                first_chunk=first_chunk,
            )
            decoded = self._pipeline.vae.decode_to_pixel(denoised, use_cache=True)
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)

        frames = decoded[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
        frames = self._color_correct_chunk(frames)
        self._chunk_index += 1
        self._current_start_frame += _LATENT_FRAMES_PER_CHUNK
        return np.ascontiguousarray(np.rint(frames).clip(0, 255), dtype=np.uint8)

    def end_session(self) -> None:
        """Release session rollout caches while keeping model weights loaded."""
        self._release_rollout()

    def _load_pipeline(self) -> Any:
        """Construct the upstream pipeline and load the pinned autoregressive checkpoint."""
        self._add_upstream_to_import_path()
        pipeline_class = self._upstream_module(
            "pipeline.pipeline_causal_camera"
        ).CausalCameraInferencePipeline
        upstream_config = OmegaConf.load(self._config.upstream_config)
        default_config = OmegaConf.load(
            self._config.upstream_config.with_name("default_config.yaml")
        )
        upstream_config = OmegaConf.merge(default_config, upstream_config)

        pipeline = pipeline_class(
            upstream_config,
            device=self._device,
            num_output_frames=_LATENT_FRAMES_PER_CHUNK,
            model_config_path=str(self._config.transformer_config),
            text_encoder_path=str(
                self._config.wan.path / "models_t5_umt5-xxl-enc-bf16.pth"
            ),
            tokenizer_path=str(self._config.wan.path / "google/umt5-xxl"),
            vae_path=str(self._config.wan.path / "Wan2.2_VAE.pth"),
        )
        checkpoint = load_file(str(self._config.dreamx.path), device="cpu")
        state_dict = {f"model.{key}": value for key, value in checkpoint.items()}
        model_parameters = {name for name, _ in pipeline.generator.named_parameters()}
        loaded_parameters = model_parameters.intersection(state_dict)
        if len(loaded_parameters) < int(len(model_parameters) * 0.95):
            raise RuntimeError(
                "DreamX checkpoint matched only "
                f"{len(loaded_parameters)}/{len(model_parameters)} generator parameters"
            )
        missing, unexpected = pipeline.generator.load_state_dict(
            state_dict, strict=False
        )
        logger.info(
            "DreamX checkpoint loaded: parameters=%s missing=%s unexpected=%s",
            len(loaded_parameters),
            len(missing),
            len(unexpected),
        )
        del checkpoint, state_dict

        pipeline = pipeline.to(dtype=self._dtype)
        pipeline.text_encoder.to(device=self._device)
        pipeline.generator.to(device=self._device)
        pipeline.vae.to(device=self._device)
        if int(pipeline.num_frame_per_block) != _LATENT_FRAMES_PER_CHUNK:
            raise RuntimeError(
                "DreamX upstream num_frame_per_block must remain "
                f"{_LATENT_FRAMES_PER_CHUNK}; got {pipeline.num_frame_per_block}"
            )
        if int(pipeline.local_attn_size) != _EXPECTED_LOCAL_ATTENTION_FRAMES:
            raise RuntimeError(
                "DreamX upstream local_attn_size must remain "
                f"{_EXPECTED_LOCAL_ATTENTION_FRAMES}; got {pipeline.local_attn_size}"
            )
        pipeline.vae.model.clear_cache()
        return pipeline

    def _denoise_one_block(
        self,
        noise: torch.Tensor,
        conditional: dict[str, torch.Tensor],
        camera: dict[str, torch.Tensor],
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        """Run the upstream spatial loop and commit one block to its rolling KV cache."""
        mask = torch.ones_like(noise)
        if first_chunk:
            mask[:, 0] = 0
        latents = noise
        timestep: torch.Tensor | None = None
        denoised: torch.Tensor | None = None
        for index, current_timestep in enumerate(self._pipeline.denoising_step_list):
            tokens = (mask[0, :, 0, ::2, ::2] * current_timestep).flatten()
            if tokens.numel() < _TOKENS_PER_LATENT_FRAME * _LATENT_FRAMES_PER_CHUNK:
                padding = tokens.new_full(
                    (
                        _TOKENS_PER_LATENT_FRAME * _LATENT_FRAMES_PER_CHUNK
                        - tokens.numel(),
                    ),
                    current_timestep,
                )
                tokens = torch.cat((tokens, padding))
            timestep = tokens.unsqueeze(0)
            _, denoised = self._pipeline.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional,
                y=None,
                y_camera=camera,
                timestep=timestep,
                kv_cache=self._pipeline.kv_cache1,
                crossattn_cache=self._pipeline.crossattn_cache,
                current_start=self._current_start_frame * _TOKENS_PER_LATENT_FRAME,
            )
            if index < len(self._pipeline.denoising_step_list) - 1:
                next_timestep = self._pipeline.denoising_step_list[
                    index + 1
                ] * torch.ones(
                    [1, _LATENT_FRAMES_PER_CHUNK],
                    device=self._device,
                    dtype=torch.long,
                )
                if first_chunk:
                    next_timestep[:, 0] = 0
                latents = self._pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    next_timestep.flatten(),
                ).unflatten(0, denoised.shape[:2])
                latents = latents * mask + noise * (1 - mask)
            else:
                denoised = denoised * mask + noise * (1 - mask)
        if denoised is None or timestep is None:
            raise RuntimeError("DreamX denoising schedule is empty")
        context_timestep = torch.ones_like(timestep) * self._pipeline.args.context_noise
        self._pipeline.generator(
            noisy_image_or_video=denoised,
            conditional_dict=conditional,
            y=None,
            y_camera=camera,
            timestep=context_timestep,
            kv_cache=self._pipeline.kv_cache1,
            crossattn_cache=self._pipeline.crossattn_cache,
            current_start=self._current_start_frame * _TOKENS_PER_LATENT_FRAME,
        )
        return denoised

    def _prompt_condition(self, prompt: str) -> dict[str, torch.Tensor]:
        """Encode a changed prompt and invalidate only its cross-attention cache."""
        if self._conditional is not None and prompt == self._conditional_prompt:
            return self._conditional
        with torch.inference_mode():
            conditional = self._pipeline.text_encoder(text_prompts=[prompt])
        for cache in self._pipeline.crossattn_cache:
            cache["is_init"] = False
        self._conditional = conditional
        self._conditional_prompt = prompt
        return conditional

    def _camera_condition(self, chunk: CameraChunk) -> dict[str, torch.Tensor]:
        """Build upstream chunk-relative PRoPE matrices for three latent poses."""
        poses = chunk.poses
        has_reference = chunk.reference_pose is not None
        if has_reference:
            poses = np.concatenate((chunk.reference_pose[None], poses), axis=0)
        world_to_camera = np.stack([_pose_world_to_camera(pose) for pose in poses])
        camera_to_world = np.linalg.inv(world_to_camera)
        relative_camera_to_world = np.stack(
            [np.eye(4, dtype=np.float32)]
            + [world_to_camera[0] @ value for value in camera_to_world[1:]],
            axis=0,
        )
        if has_reference:
            relative_camera_to_world = relative_camera_to_world[1:]
        viewmats = torch.as_tensor(
            np.linalg.inv(relative_camera_to_world),
            device=self._device,
            dtype=self._dtype,
        )
        viewmats = (
            viewmats.unsqueeze(1)
            .expand(-1, _TOKENS_PER_LATENT_FRAME, -1, -1)
            .reshape(1, -1, 4, 4)
        )
        intrinsic = torch.zeros((1, 3, 3), device=self._device, dtype=self._dtype)
        intrinsic[:, 0, 0] = 969.6969696969696 / (960.0 * 2)
        intrinsic[:, 1, 1] = 969.6969696969696 / (540.0 * 2)
        intrinsic[:, 0, 2] = 0.5
        intrinsic[:, 1, 2] = 0.5
        intrinsic[:, 2, 2] = 1.0
        intrinsic = (
            intrinsic.unsqueeze(1).expand(-1, viewmats.shape[1], -1, -1).contiguous()
        )
        return {"viewmats": viewmats, "K": intrinsic}

    def _load_image_tensor(self, image: Path | bytes) -> torch.Tensor:
        """Apply the upstream fixed 704x1280 RGB image transform."""
        source: Path | io.BytesIO
        source = io.BytesIO(image) if isinstance(image, bytes) else image
        with Image.open(source) as opened:
            resized = opened.convert("RGB").resize(
                (1280, 704), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = tensor.mul(2.0).sub(1.0).unsqueeze(0).unsqueeze(2)
        return tensor.to(device=self._device, dtype=self._dtype)

    def _color_correct_chunk(self, frames: np.ndarray) -> np.ndarray:
        """Apply upstream Lab correction against the previous chunk boundary."""
        strength = self._config.color_correction_strength
        if strength <= 0.0 or len(frames) <= 1:
            if len(frames):
                self._color_reference = frames[-1].copy()
            return frames
        reference = (
            frames[0] if self._color_reference is None else self._color_reference
        )
        reference_lab = cv2.cvtColor(
            (reference / 255.0).astype(np.float32),
            cv2.COLOR_RGB2LAB,
        )
        reference_mean = reference_lab.mean(axis=(0, 1))
        reference_std = reference_lab.std(axis=(0, 1))
        processed = frames.copy()
        for index, frame in enumerate(processed):
            rgb = (frame / 255.0).astype(np.float32)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            mean = lab.mean(axis=(0, 1))
            std = lab.std(axis=(0, 1))
            corrected = lab.copy()
            for channel in range(3):
                if std[channel] > 1e-6:
                    corrected[:, :, channel] = (
                        corrected[:, :, channel] - mean[channel]
                    ) * (reference_std[channel] / std[channel]) + reference_mean[
                        channel
                    ]
                else:
                    corrected[:, :, channel] = reference_mean[channel]
            corrected_rgb = np.clip(
                cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB),
                0.0,
                1.0,
            )
            processed[index] = (
                (1.0 - strength) * rgb + strength * corrected_rgb
            ) * 255.0
        self._color_reference = processed[-1].copy()
        return processed

    def _release_rollout(self) -> None:
        """Drop all state whose lifetime is one autoregressive rollout."""
        self._pipeline.kv_cache1 = None
        self._pipeline.crossattn_cache = None
        self._pipeline.vae.model.clear_cache()
        self._initial_latent = None
        self._conditional = None
        self._conditional_prompt = ""
        self._chunk_index = 0
        self._current_start_frame = 0
        self._color_reference = None

    def _add_upstream_to_import_path(self) -> None:
        """Place the pinned checkout first so its absolute imports remain unchanged."""
        source = str(self._config.source_path)
        if source not in sys.path:
            sys.path.insert(0, source)

    def _upstream_module(self, name: str) -> Any:
        """Import an upstream module and require it to come from the pinned checkout."""
        self._add_upstream_to_import_path()
        module = importlib.import_module(name)
        module_path = module.__file__
        if module_path is None:
            raise RuntimeError(f"{name} does not resolve to a source file")
        module_file = Path(module_path).resolve()
        if not module_file.is_relative_to(self._config.source_path):
            raise RuntimeError(
                f"{name} resolved outside the pinned DreamX-World source: {module_file}"
            )
        return module


def _pose_world_to_camera(pose: np.ndarray) -> np.ndarray:
    """Return a homogeneous world-to-camera matrix from one 19-value pose row."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3] = np.asarray(pose[7:], dtype=np.float32).reshape(3, 4)
    return matrix
