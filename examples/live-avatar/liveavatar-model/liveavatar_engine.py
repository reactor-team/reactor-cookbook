"""LiveAvatar's denoiser, driven one block at a time on one GPU.

Upstream LiveAvatar renders a whole audio file in clips of 48 frames, each made
of four causal blocks of three latent frames (12 video frames, 0.48 s at
25 FPS). Every denoising step keeps its own key/value cache of the blocks
before it, at that step's noise level, so a block needs only its own audio and
the caches. This module runs that loop one block per call, from audio that
arrives while the session runs, and keeps the clip's state between calls:

- the four per-step caches, which roll over one clip of latent frames, and the
  reference ("cond") cache they share, filled from the portrait at bind;
- the clip's noise, drawn at the start of every fourth block as upstream draws
  it per clip;
- the streaming VAE decoder's causal cache.

It imports the pinned upstream source and nothing from ``reactor_runtime``.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Video frames per second; the audio features are sampled at this rate.
FPS = 25
#: Latent frames per causal block.
LATENT_FRAMES_PER_BLOCK = 3
#: Video frames one block decodes to.
BLOCK_FRAMES = 4 * LATENT_FRAMES_PER_BLOCK
#: Blocks in one upstream clip: the noise is drawn per clip, the caches roll over one clip.
BLOCKS_PER_CLIP = 4
#: Video frames the first block of a clip-chain drops: the decoder's warm-up frames.
FIRST_BLOCK_DROP = 3
#: Rate of the audio the wav2vec encoder reads.
ENCODER_RATE = 16_000
#: Audio samples at :data:`ENCODER_RATE` per video frame.
ENCODER_SAMPLES_PER_FRAME = ENCODER_RATE // FPS
#: The released sampler: four Euler steps with flow shift 3, no guidance.
SAMPLING_STEPS = 4
#: Latent frame positions the rotary table covers; a clip restarts before it runs out.
_ROPE_FRAMES = 45_000


@dataclass
class EngineTiming:
    """Milliseconds the last block spent in each stage, measured on the GPU's clock."""

    audio_ms: float = 0.0
    dit_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    steps_ms: list[float] = field(default_factory=list)


class BlockEngine:
    """Upstream LiveAvatar's causal sampler, one block per :meth:`render`.

    Args:
        source: The pinned upstream checkout.
        base: The Wan2.2-S2V-14B checkpoint directory.
        lora: The LiveAvatar checkpoint directory (``liveavatar.safetensors``).
        height: Output height in pixels, a multiple of 64.
        width: Output width in pixels, a multiple of 64.
        prompt: The scene and performance text the denoiser is conditioned on.
        seed: Seed of the clip noise and of upstream's rotary offsets.
        audio_context_s: Seconds of earlier audio the encoder reads before a block.
        device: The CUDA device.
    """

    def __init__(
        self,
        source: Path,
        base: Path,
        lora: Path,
        *,
        height: int = 512,
        width: int = 512,
        prompt: str = "",
        seed: int = 420,
        audio_context_s: float = 1.0,
        device: str = "cuda:0",
    ) -> None:
        if height % 64 or width % 64:
            raise ValueError("height and width must be multiples of 64")
        self.source = source
        self.base = base
        self.lora = lora
        self.height = height
        self.width = width
        self.prompt = prompt
        self.seed = seed
        self.audio_context = int(audio_context_s * ENCODER_RATE)
        self.device_name = device
        self.timing = EngineTiming()
        self._pipe: Any = None
        self._bound = False

    # -- load -------------------------------------------------------------------

    def load(self, *, compile_dit: bool = False, fp8: bool = False) -> None:
        """Load the denoiser with the LiveAvatar weights merged, the VAE, wav2vec and the text encoder.

        Encodes :attr:`prompt` once and leaves the text encoder on the CPU.

        Args:
            compile_dit: Compile the denoiser's forward and the decoder, and capture each step as a CUDA graph.
            fp8: Run the transformer blocks' projections on FP8 tensor cores (upstream's ``--fp8``),
                which changes pixels slightly.
        """
        import torch
        import yaml

        torch.cuda.set_device(self.device_name)
        self.device = torch.device(self.device_name)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if str(self.source) not in sys.path:
            sys.path.insert(0, str(self.source))
        # Upstream decorates its forward and decoder with torch.compile when this is on; the engine
        # compiles the denoiser itself.
        os.environ["ENABLE_COMPILE"] = "false"
        from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS["s2v-14B"]
        pipe = WanS2V(
            config=cfg,
            checkpoint_dir=str(self.base),
            device_id=self.device.index or 0,
            rank=0,
            single_gpu=False,
            sp_size=1,
            convert_model_dtype=True,
            init_on_cpu=False,
        )
        settings = yaml.safe_load(
            (self.source / "liveavatar/configs/s2v_causal_sft.yaml").read_text()
        )
        pipe.noise_model = pipe.add_lora_to_model(
            pipe.noise_model,
            lora_rank=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
            lora_target_modules=settings["lora_target_modules"],
            init_lora_weights=settings["init_lora_weights"],
            pretrained_lora_path=str(self.lora / "liveavatar.safetensors"),
            load_lora_weight_only=False,
        )
        pipe.noise_model.eval().requires_grad_(False)
        pipe.audio_encoder.model.to(
            device=self.device, dtype=pipe.param_dtype
        ).eval().requires_grad_(False)
        self._pipe = pipe
        self.dtype = pipe.param_dtype
        self.motion_frames = pipe.motion_frames
        self.lat_motion_frames = (self.motion_frames + 3) // 4
        self.lat_h, self.lat_w = self.height // 8, self.width // 8
        self.frame_tokens = (self.lat_h // 2) * (self.lat_w // 2)
        self.block_tokens = LATENT_FRAMES_PER_BLOCK * self.frame_tokens
        self.cache_tokens = BLOCKS_PER_CLIP * self.block_tokens

        with torch.no_grad():
            pipe.text_encoder.model.to(self.device)
            self.context = pipe.text_encoder([self.prompt], self.device)
            pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=pipe.num_train_timesteps, shift=3
        )
        scheduler.set_timesteps(SAMPLING_STEPS, device=self.device)
        self.timesteps = scheduler.timesteps.clone()
        self.sigmas = [float(s) for s in scheduler.sigmas]
        self._step_t = [
            torch.full(
                (1, LATENT_FRAMES_PER_BLOCK),
                float(t),
                device=self.device,
                dtype=torch.float32,
            )
            for t in self.timesteps
        ]
        self.cond_states = torch.zeros(
            (1, 16, LATENT_FRAMES_PER_BLOCK, self.lat_h, self.lat_w),
            device=self.device,
            dtype=self.dtype,
        )
        from liveavatar_fast import (
            FastForward,
            install_dense_attention,
            quantize_blocks_fp8,
        )

        install_dense_attention()
        self.fast = FastForward(
            pipe.noise_model,
            frame_tokens=self.frame_tokens,
            window_tokens=self.cache_tokens,
            steps=SAMPLING_STEPS,
        )
        self.fast.set_context(self.context[0])
        if fp8:
            quantize_blocks_fp8(pipe.noise_model)
        self._allocate_reference_cache()
        if compile_dit:
            self.fast.compile()
            # The decoder runs eagerly at about twice the time; compiled, its feature caches stay Python
            # lists the graph reads and writes.
            pipe.vae.model.decoder = torch.compile(
                pipe.vae.model.decoder, dynamic=False
            )

    def _allocate_reference_cache(self) -> None:
        """Allocate the reference cache upstream's prefill writes, in upstream's per-layer layout."""
        import torch

        layers = self._pipe.noise_model.num_layers
        shape = (1, 0, 40, 128)
        cond_shape = (1, 3000, 40, 128)
        self.cond_cache = [
            {
                "cond_k": torch.zeros(cond_shape, dtype=self.dtype, device=self.device),
                "cond_v": torch.zeros(cond_shape, dtype=self.dtype, device=self.device),
                "cond_end": torch.tensor([0], dtype=torch.long, device=self.device),
            }
            for _ in range(layers)
        ]
        self.prefill_cache = [
            {
                "k": torch.zeros(shape, dtype=self.dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.dtype, device=self.device),
                **self.cond_cache[layer],
            }
            for layer in range(layers)
        ]
        self.crossattn_cache = [
            {
                "k": torch.zeros((1, 0, 40, 128), dtype=self.dtype, device=self.device),
                "v": torch.zeros((1, 0, 40, 128), dtype=self.dtype, device=self.device),
                "is_init": False,
            }
            for _ in range(layers)
        ]

    # -- clips --------------------------------------------------------------------

    def bind(self, image: np.ndarray) -> None:
        """Start a clip from a portrait: encode it, fill the reference cache, and reset the decoder.

        Args:
            image: RGB uint8 ``(H, W, 3)`` at the engine's output size.
        """
        import torch

        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"the portrait must be uint8 ({self.height}, {self.width}, 3)"
            )
        pipe = self._pipe
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            pixels = (
                torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 127.5
                - 1.0
            )
            pixels = pixels[None, :, None].to(dtype=pipe.vae.dtype)
            self.ref_latents = torch.stack(pipe.vae.encode(pixels))
            motion = pixels.repeat(1, 1, self.motion_frames, 1, 1)
            self.motion_latents = torch.stack(pipe.vae.encode(motion))
            for layer in self.crossattn_cache:
                layer["is_init"] = False
                layer["k"] = layer["k"][:, :0]
                layer["v"] = layer["v"][:, :0]
            for layer in self.cond_cache:
                layer["cond_end"].zero_()
            self._sink(self.ref_latents)
            vae = pipe.vae.model
            vae.clear_cache()
            vae.first_decode = True
            vae.first_encode = True
            pipe.vae.stream_decode(self.motion_latents[:, :, :7])
        random.seed(self.seed)
        self._generator = torch.Generator(device=self.device)
        self.block = 0
        self._clip_noise: Any = None
        self._first_latent: Any = None
        self._audio_history = np.zeros(0, dtype=np.float32)
        self._bound = True
        torch.cuda.synchronize(self.device)

    def _sink(self, ref_latents: Any) -> None:
        """Fill the reference cache every step shares: the portrait and the motion frames, at t = 0."""
        import torch

        latents = torch.zeros(
            (16, LATENT_FRAMES_PER_BLOCK, self.lat_h, self.lat_w),
            device=self.device,
            dtype=self.dtype,
        )
        audio = torch.zeros(
            (1, 25, 1024, BLOCK_FRAMES), device=self.device, dtype=self.dtype
        )
        self._pipe.noise_model(
            [latents],
            t=self._step_t[0] * 0,
            context=self.context[0:1],
            seq_len=None,
            cond_states=self.cond_states,
            motion_latents=self.motion_latents,
            ref_latents=ref_latents,
            audio_input=audio,
            motion_frames=[self.motion_frames, self.lat_motion_frames],
            drop_motion_frames=False,
            sink_flag=True,
            kv_cache=self.prefill_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=0,
            current_end=self.block_tokens,
        )
        self.fast.set_reference(self.cond_cache)

    @property
    def bound(self) -> bool:
        return self._bound

    def reset(self) -> None:
        """Forget the clip; the next :meth:`render` needs a :meth:`bind` first."""
        self._bound = False

    # -- blocks -------------------------------------------------------------------

    def frames_for_block(self, block: int) -> int:
        """Video frames block number ``block`` of a clip-chain decodes to."""
        return BLOCK_FRAMES - FIRST_BLOCK_DROP if block == 0 else BLOCK_FRAMES

    def render(self, audio: np.ndarray) -> np.ndarray:
        """Render the next block from its audio.

        Args:
            audio: float32 mono at :data:`ENCODER_RATE`, ``BLOCK_FRAMES * ENCODER_SAMPLES_PER_FRAME``
                samples, in [-1, 1]. The engine keeps the audio of earlier blocks as the encoder's context.

        Returns:
            RGB uint8 ``(N, height, width, 3)``; ``N`` is :meth:`frames_for_block` of the block.
        """
        import torch

        if not self._bound:
            raise RuntimeError("bind a portrait before rendering")
        need = BLOCK_FRAMES * ENCODER_SAMPLES_PER_FRAME
        if audio.shape != (need,):
            raise ValueError(f"a block takes {need} samples at {ENCODER_RATE} Hz")
        if (self.block + BLOCKS_PER_CLIP) * LATENT_FRAMES_PER_BLOCK >= _ROPE_FRAMES:
            raise RuntimeError("the clip has run out of rotary positions; bind again")
        pipe = self._pipe
        events = [
            torch.cuda.Event(enable_timing=True) for _ in range(SAMPLING_STEPS + 4)
        ]
        wall = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            events[0].record()
            audio_input = self._audio_features(audio)
            events[1].record()
            clip, index = divmod(self.block, BLOCKS_PER_CLIP)
            if index == 0:
                if clip == 1 and self._first_latent is not None:
                    # Upstream's real-time pipeline anchors the clips after the first on the first
                    # latent frame it generated rather than on the encoded portrait.
                    self._sink(self._first_latent)
                self._generator.manual_seed(self.seed + clip)
                self._clip_noise = torch.randn(
                    (
                        16,
                        BLOCKS_PER_CLIP * LATENT_FRAMES_PER_BLOCK,
                        self.lat_h,
                        self.lat_w,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                    generator=self._generator,
                )
            span = slice(
                index * LATENT_FRAMES_PER_BLOCK, (index + 1) * LATENT_FRAMES_PER_BLOCK
            )
            latents = self._clip_noise[:, span]
            start = self.block * self.block_tokens
            self.fast.encode_audio(
                audio_input, (self.motion_frames, self.lat_motion_frames)
            )
            for i in range(SAMPLING_STEPS):
                velocity = self.fast(
                    latents, self._step_t[i], self.cond_states, i, start
                )
                latents = (
                    latents.float()
                    + (self.sigmas[i + 1] - self.sigmas[i]) * velocity.float()
                ).to(velocity.dtype)
                events[2 + i].record()
            if self.block == 0:
                self._first_latent = latents[None, :, 0:1].clone()
            image = torch.stack(pipe.vae.stream_decode(latents[None]))[0]
            image = image[:, -BLOCK_FRAMES:]
            if self.block == 0:
                image = image[:, FIRST_BLOCK_DROP:]
            frames = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).contiguous()
            events[2 + SAMPLING_STEPS].record()
        out = frames.cpu().numpy()
        events[3 + SAMPLING_STEPS].record()
        torch.cuda.synchronize(self.device)
        steps = [
            events[1 + i].elapsed_time(events[2 + i]) for i in range(SAMPLING_STEPS)
        ]
        self.timing = EngineTiming(
            audio_ms=events[0].elapsed_time(events[1]),
            dit_ms=sum(steps),
            decode_ms=events[1 + SAMPLING_STEPS].elapsed_time(
                events[2 + SAMPLING_STEPS]
            ),
            total_ms=(time.perf_counter() - wall) * 1000,
            steps_ms=steps,
        )
        self.block += 1
        return out

    def _audio_features(self, audio: np.ndarray) -> Any:
        """wav2vec features of the block's frames, from the block and the audio before it.

        Upstream encodes a whole file, interpolates the 50 Hz features to 30 Hz, and takes the
        30 Hz index nearest each 25 FPS frame. This reads a window of earlier audio plus the block
        and samples the same positions within it.
        """
        import torch

        encoder = self._pipe.audio_encoder
        history = self._audio_history[-self.audio_context :]
        window = np.concatenate([history, audio.astype(np.float32)])
        self._audio_history = (
            window[-self.audio_context :] if self.audio_context else window[:0]
        )
        values = encoder.processor(
            window, sampling_rate=ENCODER_RATE, return_tensors="pt"
        ).input_values
        hidden = encoder.model(
            values.to(self.device, dtype=self.dtype), output_hidden_states=True
        ).hidden_states
        features = torch.cat(hidden).float()  # (layers, T50, 1024)
        count = features.shape[1]
        first = self.block * BLOCK_FRAMES
        frames = np.arange(first, first + BLOCK_FRAMES)
        window_start = (
            self.block * BLOCK_FRAMES * ENCODER_SAMPLES_PER_FRAME - history.size
        ) / 320.0
        positions = np.round(frames * 30 / FPS) * 50 / 30 - window_start
        positions = np.clip(positions, 0, count - 1)
        low = np.floor(positions).astype(np.int64)
        high = np.minimum(low + 1, count - 1)
        weight = torch.from_numpy(positions - low).to(self.device, torch.float32)[
            None, :, None
        ]
        picked = (
            features[:, low] * (1 - weight) + features[:, high] * weight
        )  # (layers, F, 1024)
        return picked.permute(0, 2, 1)[None].to(self.dtype)  # (1, layers, 1024, F)
