"""Advance Echo-WM Flash one native audio-video block per Reactor turn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from helpers.action_camera import default_k_pix
from helpers.action_condition import (
    _INTERNAL_TRANSLATION_CALIBRATION,
    action_config,
)
from ltx_causal import (
    CausalCacheConfig,
    CausalModelWrapper,
    causal_audio_blocks,
    causal_audio_frames,
    causal_video_blocks,
)
from ltx_causal.rollout import (
    _BlockForward,
    _denoise_av_block,
    _generate_audio_prefix,
    _RolloutBuffers,
)
from ltx_causal.scheduling import resolve_causal_sigmas
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae import decode_audio
from ltx_core.model.transformer.attention import Attention, PytorchAttention
from ltx_core.model.video_vae import decode_video
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils import ModelLedger, combined_image_conditionings
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import create_noised_state, noise_video_state
from ltx_pipelines.utils.types import PipelineComponents

from echo_wm_assets import EchoWMConfig
from echo_wm_attention import (
    AttentionBenchmark,
    benchmark_attention_backends,
    resolve_attention_backend,
    set_attention_backend,
)
from echo_wm_camera import CameraChunk


class EchoWMBackend:
    """Own upstream model components and one bounded causal rollout."""

    def __init__(self, config: EchoWMConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Echo-WM Flash requires a CUDA accelerator")
        self._config = config
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._cache_config = CausalCacheConfig(
            video_local_attn_size=config.video_local_attn_size,
            video_sink_size=config.video_sink_size,
            video_chunk_size=config.video_chunk_size,
        )
        self._cache_config.validate()
        self._total_video_latents = 1 + config.video_chunk_size * config.max_chunks
        self._total_pixel_frames = 1 + config.frames_per_chunk * config.max_chunks
        self._patches_per_frame = (config.height // 32) * (config.width // 32)
        self._video_blocks = causal_video_blocks(
            self._total_video_latents, config.video_chunk_size
        )
        self._audio_blocks = causal_audio_blocks(
            self._total_video_latents, config.video_chunk_size
        )
        self._audio_latents = causal_audio_frames(
            self._total_video_latents, config.video_chunk_size
        )
        self._ledger = ModelLedger(
            dtype=self._dtype,
            device=self._device,
            checkpoint_path=str(config.checkpoint.path),
            gemma_root_path=str(config.gemma.path),
        )
        self._components = PipelineComponents(dtype=self._dtype, device=self._device)
        self._text_encoder = self._ledger.text_encoder()
        self._embeddings_processor = self._ledger.gemma_embeddings_processor()
        self._video_encoder = self._ledger.video_encoder()
        self._x0_model = self._ledger.transformer(
            action_config=action_config(config.width, config.height)
        )
        pytorch_attention = PytorchAttention()
        attention_function = resolve_attention_backend(
            config.attention_backend,
            pytorch_attention=pytorch_attention,
            torch_module=torch,
        )
        self._attention_modules = set_attention_backend(
            self._x0_model,
            attention_function,
            Attention,
        )
        self._attention_benchmark: AttentionBenchmark | None = None
        if (
            config.attention_benchmark
            and config.attention_backend == "flash_attention_4"
        ):
            self._attention_benchmark = benchmark_attention_backends(
                flash_attention=attention_function,
                pytorch_attention=pytorch_attention,
                torch_module=torch,
                query_tokens=config.video_chunk_size * self._patches_per_frame,
                key_value_tokens=config.video_local_attn_size * self._patches_per_frame,
            )
        self._wrapper = CausalModelWrapper(
            self._x0_model.velocity_model,
            patches_per_frame=self._patches_per_frame,
            cache=self._cache_config,
        )
        self._video_decoder = self._ledger.video_decoder()
        self._audio_decoder = self._ledger.audio_decoder()
        self._vocoder = self._ledger.vocoder()
        self._audio_sample_rate = int(self._vocoder.output_sampling_rate)
        if self._audio_sample_rate != 48_000:
            raise ValueError(
                f"Echo-WM audio must be 48000 Hz, got {self._audio_sample_rate}"
            )
        self._sigmas = resolve_causal_sigmas(config.timesteps)
        self._forward: _BlockForward | None = None
        self._buffers: _RolloutBuffers | None = None
        self._action_cond: dict[str, torch.Tensor] | None = None
        self._generator: torch.Generator | None = None
        self._chunk_index = 0
        self._seed = config.seed
        self._last_profile: dict[str, float] = {}

    @property
    def attention_modules(self) -> int:
        """Return upstream attention modules using the configured callable."""
        return self._attention_modules

    @property
    def attention_benchmark(self) -> AttentionBenchmark | None:
        """Return startup SDPA/FA4 verification results when enabled."""
        return self._attention_benchmark

    @property
    def last_profile(self) -> dict[str, float]:
        """Return CUDA stage timings for the latest generated chunk."""
        return dict(self._last_profile)

    @torch.inference_mode()
    def reset(self, *, image: Path, prompt: str, seed: int, fov_degrees: float) -> None:
        """Initialize upstream conditioning, buffers, and bounded caches."""
        self.end_session(release_cuda_cache=False)
        config = self._config
        encoded_prompt = self._encode_prompt(prompt)
        output_shape = VideoPixelShape(
            1,
            self._total_pixel_frames,
            config.height,
            config.width,
            config.fps,
        )
        conditionings = combined_image_conditionings(
            [ImageConditioningInput(str(image), 0, 1.0)],
            config.height,
            config.width,
            self._video_encoder,
            self._dtype,
            self._device,
        )
        generator = torch.Generator(device=self._device).manual_seed(seed)
        noiser = GaussianNoiser(generator)
        video_state, _ = noise_video_state(
            output_shape,
            noiser,
            conditionings,
            self._components,
            self._dtype,
            self._device,
        )
        audio_shape = AudioLatentShape(
            batch=1,
            channels=8,
            frames=self._audio_latents,
            mel_bins=16,
        )
        audio_tools = AudioLatentTools(self._components.audio_patchifier, audio_shape)
        audio_state = create_noised_state(
            audio_tools,
            [],
            noiser,
            self._dtype,
            self._device,
        )
        action_cond = self._empty_action_condition(fov_degrees)
        buffers = _RolloutBuffers.create(
            video_state.clean_latent,
            audio_state.clean_latent,
            self._patches_per_frame,
            generator,
        )
        forward = _BlockForward.create(
            wrapper=self._wrapper,
            clean_video=video_state.clean_latent,
            clean_audio=audio_state.clean_latent,
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
            video_context=encoded_prompt.video_encoding,
            audio_context=encoded_prompt.audio_encoding,
            context_mask=encoded_prompt.attention_mask,
            action_cond=action_cond,
        )
        _generate_audio_prefix(
            forward,
            buffers,
            self._video_blocks[0],
            self._audio_blocks[0],
            self._sigmas,
            generator,
        )
        self._forward = forward
        self._buffers = buffers
        self._action_cond = action_cond
        self._generator = generator
        self._chunk_index = 0
        self._seed = seed
        self._last_profile = {}

    @torch.inference_mode()
    def generate_chunk(
        self,
        camera: CameraChunk,
        *,
        fov_degrees: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate, cache, decode, and return one native audio-video block."""
        forward = self._require_forward()
        buffers = self._require_buffers()
        generator = self._require_generator()
        next_index = self._chunk_index + 1
        if next_index >= len(self._video_blocks):
            raise RuntimeError("Echo-WM rollout reached its configured chunk capacity")
        video_block = self._video_blocks[next_index]
        audio_block = self._audio_blocks[next_index]
        self._update_camera(camera, video_block, fov_degrees)
        events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            if self._config.profile_cuda
            else []
        )
        if events:
            events[0].record()
        video_sample, audio_sample = _denoise_av_block(
            forward,
            buffers.initial_video,
            buffers.initial_audio,
            video_block,
            audio_block,
            self._sigmas,
            generator,
        )
        if events:
            events[1].record()
        video_start, video_end = video_block
        audio_start, audio_end = audio_block
        buffers.video_output[
            :,
            video_start * self._patches_per_frame : video_end * self._patches_per_frame,
        ] = video_sample
        buffers.audio_output[:, audio_start:audio_end] = audio_sample
        forward(video_sample, video_block, 0.0, audio_sample, audio_block, 0.0)
        if events:
            events[2].record()
        video = self._decode_video(video_end)
        if events:
            events[3].record()
        audio = self._decode_audio(audio_end, int(video.shape[0]))
        if events:
            events[4].record()
            events[4].synchronize()
            self._last_profile = {
                "denoise_seconds": events[0].elapsed_time(events[1]) / 1000.0,
                "cache_commit_seconds": events[1].elapsed_time(events[2]) / 1000.0,
                "video_decode_seconds": events[2].elapsed_time(events[3]) / 1000.0,
                "audio_decode_seconds": events[3].elapsed_time(events[4]) / 1000.0,
                "cuda_total_seconds": events[0].elapsed_time(events[4]) / 1000.0,
            }
        else:
            self._last_profile = {}
        self._chunk_index = next_index
        return video, audio

    def end_session(self, *, release_cuda_cache: bool = True) -> None:
        """Release rollout tensors while retaining loaded model components."""
        self._forward = None
        self._buffers = None
        self._action_cond = None
        self._generator = None
        self._chunk_index = 0
        self._last_profile = {}
        if release_cuda_cache:
            torch.cuda.empty_cache()

    def _encode_prompt(self, prompt: str) -> Any:
        hidden_states, mask = self._text_encoder.encode(prompt)
        result = self._embeddings_processor.process_hidden_states(hidden_states, mask)
        if result.audio_encoding is None:
            raise ValueError(
                "Echo-WM Flash checkpoint did not provide audio text embeddings"
            )
        return result

    def _empty_action_condition(self, fov_degrees: float) -> dict[str, torch.Tensor]:
        viewmats = (
            torch.eye(
                4,
                device=self._device,
                dtype=torch.float32,
            )
            .reshape(1, 1, 4, 4)
            .repeat(1, self._total_video_latents, 1, 1)
        )
        intrinsic = default_k_pix(
            self._config.width,
            self._config.height,
            fov_degrees,
        ).to(device=self._device, dtype=torch.float32)
        intrinsics = intrinsic.reshape(1, 1, 3, 3).repeat(
            1, self._total_video_latents, 1, 1
        )
        return {"ucpe_viewmats": viewmats, "ucpe_Ks": intrinsics}

    def _update_camera(
        self,
        camera: CameraChunk,
        video_block: tuple[int, int],
        fov_degrees: float,
    ) -> None:
        action = self._action_cond
        if action is None:
            raise RuntimeError("Echo-WM rollout was not initialized")
        start, end = video_block
        expected = end - start
        if camera.latent_poses.shape != (expected, 4, 4):
            raise ValueError(
                f"Echo-WM camera chunk must have shape {(expected, 4, 4)}, "
                f"got {camera.latent_poses.shape}"
            )
        normalized = torch.from_numpy(camera.latent_poses).to(
            device=self._device,
            dtype=torch.float32,
        )
        normalized = normalized.clone()
        normalized[:, :3, 3] /= _INTERNAL_TRANSLATION_CALIBRATION
        action["ucpe_viewmats"][:, start:end].copy_(normalized.unsqueeze(0))
        intrinsic = default_k_pix(
            self._config.width,
            self._config.height,
            fov_degrees,
        ).to(device=self._device, dtype=torch.float32)
        action["ucpe_Ks"][:, start:end].copy_(
            intrinsic.reshape(1, 1, 3, 3).expand(1, expected, -1, -1)
        )

    def _decode_video(self, video_end: int) -> np.ndarray:
        buffers = self._require_buffers()
        context = min(self._config.video_decode_context_latents, video_end)
        start = video_end - context
        token_start = start * self._patches_per_frame
        token_end = video_end * self._patches_per_frame
        shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=context,
            height=self._config.height // 32,
            width=self._config.width // 32,
        )
        latent = self._components.video_patchifier.unpatchify(
            buffers.video_output[:, token_start:token_end],
            output_shape=shape,
        )
        decode_generator = torch.Generator(device=self._device).manual_seed(self._seed)
        decoded = torch.cat(
            list(
                decode_video(
                    latent,
                    self._video_decoder,
                    tiling_config=(
                        TilingConfig.default()
                        if self._config.video_decode_tiling
                        else None
                    ),
                    generator=decode_generator,
                )
            ),
            dim=0,
        )
        frame_count = 25 if self._chunk_index == 0 else self._config.frames_per_chunk
        return np.ascontiguousarray(
            decoded[-frame_count:].cpu().numpy(), dtype=np.uint8
        )

    def _decode_audio(self, audio_end: int, video_frames: int) -> np.ndarray:
        buffers = self._require_buffers()
        context = min(self._cache_config.audio_local_attn_size, audio_end)
        start = audio_end - context
        shape = AudioLatentShape(
            batch=1,
            channels=8,
            frames=context,
            mel_bins=16,
        )
        latent = self._components.audio_patchifier.unpatchify(
            buffers.audio_output[:, start:audio_end],
            output_shape=shape,
        )
        decoded = decode_audio(latent, self._audio_decoder, self._vocoder)
        waveform = decoded.waveform.float().mean(dim=0)
        expected_samples = round(
            video_frames * self._audio_sample_rate / self._config.fps
        )
        if waveform.numel() < expected_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (expected_samples - waveform.numel(), 0),
            )
        waveform = waveform[-expected_samples:]
        pcm = (
            waveform.clamp(-1.0, 1.0)
            .mul(32767.0)
            .round()
            .to(torch.int16)
            .reshape(1, -1)
            .cpu()
            .numpy()
        )
        return np.ascontiguousarray(pcm)

    def _require_forward(self) -> _BlockForward:
        if self._forward is None:
            raise RuntimeError("Echo-WM rollout was not initialized")
        return self._forward

    def _require_buffers(self) -> _RolloutBuffers:
        if self._buffers is None:
            raise RuntimeError("Echo-WM rollout was not initialized")
        return self._buffers

    def _require_generator(self) -> torch.Generator:
        if self._generator is None:
            raise RuntimeError("Echo-WM rollout was not initialized")
        return self._generator
