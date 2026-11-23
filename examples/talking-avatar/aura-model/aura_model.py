"""AURA-1.3B, the model half: one 16-frame chunk per call, from speech that arrives while the session runs.

Implements :class:`avatar_contract.AvatarModel` over AURA's reference code, installed from its pinned repository:
the transformer, the VAE, the text encoder, the rolling key/value cache and the sampler are AURA's own; this module
feeds them one chunk at a time. AURA's reference loop reads the Mimi features of a whole file; here the features of
each chunk come from Mimi run on the chunk's speech plus the speech before it, sampled at the positions the reference
samples a file at. Imports nothing from the runtime.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np

from aura_assets import DIT, MIMI, TEXT_ENCODER, TOKENIZER, VAE
from avatar_contract import AvatarInput, AvatarResult

FPS = 16
#: Video frames one chunk advances (four latent frames; the first chunk of a clip decodes to 13).
BLOCK_FRAMES = 16
SAMPLE_RATE = 24_000
BLOCK_SAMPLES = BLOCK_FRAMES * SAMPLE_RATE // FPS
#: Mimi's frame: 80 ms at 24 kHz, 12.5 per second.
MIMI_FRAME = 1920
MIMI_RATE = 12.5
#: The rate the reference samples a file's features at before it picks one per video frame.
FEATURE_RATE = 25

DEFAULT_PROMPT = "A person is talking to the camera, with natural facial expressions and head movements."


class NoAvatar(Exception):
    """A fresh clip is asked for and the chunk's input carries no portrait to bind."""


def feature_positions(chunk: int, history_samples: int) -> np.ndarray:
    """Where the 16 frames of ``chunk`` read Mimi's 12.5 Hz features, counted from the start of the window.

    The reference picks a file's 25 Hz features at ``round(t * 25)`` for a frame at time ``t``; at 12.5 Hz that is
    half the index. The window starts ``history_samples`` before the chunk.
    """
    window_start = (chunk * BLOCK_SAMPLES - history_samples) / SAMPLE_RATE
    frames = np.arange(chunk * BLOCK_FRAMES, (chunk + 1) * BLOCK_FRAMES)
    return np.round(frames / FPS * FEATURE_RATE) / 2 - window_start * MIMI_RATE


class AuraModel:
    """AURA-1.3B behind ``load`` / ``generate`` / ``reset``."""

    fps = FPS
    block_frames = BLOCK_FRAMES
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.aura: Any = None
        self.avatar_id: int | None = None
        self.timing: dict[str, float] = {}

    def load(
        self,
        weights_root: Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        seed: int = 42,
        audio_context_s: float = 4.0,
        compile_dit: bool = True,
        warmup_avatar: bytes | None = None,
        warmup_blocks: int = 6,
    ) -> None:
        """Load AURA-1.3B on the GPU from the weights folder, encode the prompt, and render warm-up chunks.

        Builds the reference pipeline from the files under ``weights_root`` (see ``aura_assets``) as AURA's own
        ``Aura("aura-1.3b")`` builds it from the Hugging Face hub. The text encoder moves to the CPU once the prompt
        is encoded. With ``compile_dit``, each transformer block compiles; the warm-up chunks pay that cost here.
        """
        import torch
        from aura.configs import CONFIGS
        from aura.model import AuraDiT
        from aura.pipeline import Aura, FlowSchedule
        from aura.wan.t5 import T5EncoderModel
        from aura.wan.vae2_1 import Wan2_1_VAE
        from moshi.models import loaders
        from safetensors.torch import load_file

        cfg = CONFIGS["aura-1.3b"]
        device = torch.device("cuda")
        aura = Aura.__new__(Aura)
        aura.cfg, aura.device = cfg, device
        aura.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=str(weights_root / TEXT_ENCODER),
            tokenizer_path=str(weights_root / TOKENIZER),
        )
        aura.vae = Wan2_1_VAE(
            vae_pth=str(weights_root / VAE), dtype=cfg.vae_dtype, device=device
        )
        with torch.device("meta"):
            aura.model = AuraDiT(cfg)
        state = load_file(weights_root / DIT)
        state = {k: v for k, v in state.items() if not k.startswith("frame_packer.")}
        state["trainable_cond_mask.weight"] = state["trainable_cond_mask.weight"][:2]
        aura.model.load_state_dict(state, assign=True)
        aura.model.eval().requires_grad_(False).to(device)
        if compile_dit:
            for i, block in enumerate(aura.model.blocks):
                aura.model.blocks[i] = torch.compile(block, dynamic=True)
        aura.schedule = FlowSchedule()
        self.aura, self.cfg, self.device = aura, cfg, device
        self.mimi = loaders.get_mimi(
            str(weights_root / MIMI), device=device, num_codebooks=1
        ).eval()
        self.seed = seed
        self.history_samples = (
            int(audio_context_s * SAMPLE_RATE) // MIMI_FRAME * MIMI_FRAME
        )
        with torch.no_grad():
            self.context = aura.text_encoder([prompt], device)[0]
        aura.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        if warmup_blocks:
            if warmup_avatar is None:
                raise ValueError("warm-up chunks need a portrait")
            t = np.arange(BLOCK_SAMPLES) / SAMPLE_RATE
            voiced = (
                6000
                * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
                * np.sin(2 * np.pi * 180 * t)
            ).astype(np.int16)
            silence = np.zeros(BLOCK_SAMPLES, np.int16)
            for n in range(warmup_blocks):
                speaking = n % 2 == 0
                self.generate(
                    AvatarInput(
                        audio=voiced if speaking else silence,
                        silent=not speaking,
                        listening=False,
                        avatar=warmup_avatar,
                        avatar_id=-1,
                    )
                )
            self.reset()

    def generate(self, input: AvatarInput) -> AvatarResult:
        """Render one chunk: 16 frames, 1.0 s of video.

        Binds ``input.avatar`` first when ``input.avatar_id`` is not the portrait this model last bound, which starts
        a new clip. Then renders the chunk from its speech, on AURA's four denoising steps.

        Raises:
            NoAvatar: A new clip is due and the input carries no portrait.
        """
        if input.avatar_id != self.avatar_id:
            if input.avatar is None:
                raise NoAvatar("no portrait to start a clip from")
            self._bind(input.avatar)
            self.avatar_id = input.avatar_id
        import torch

        aura, cfg, r = self.aura, self.cfg, self.chunk
        wall = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            audio_tokens = aura.model.encode_audio(self._features(input.audio))
            generator = torch.Generator(self.device).manual_seed(self.seed + r)
            shape = (cfg.in_dim, cfg.frames_per_block, self.lat_h, self.lat_w)
            x = torch.randn(
                shape, dtype=torch.bfloat16, device=self.device, generator=generator
            )
            steps = cfg.denoising_steps
            for i, t in enumerate(steps):
                timestep = torch.full((1,), t, dtype=torch.long, device=self.device)
                velocity = aura.model(
                    x,
                    timestep,
                    self.context,
                    self.ref_latent,
                    audio_tokens,
                    self.kv_cache,
                )
                x0 = (x.double() - aura.schedule.sigma(t) * velocity.double()).float()
                if i + 1 < len(steps):
                    sigma = aura.schedule.sigma(steps[i + 1])
                    noise = torch.randn(
                        x0.shape,
                        dtype=x0.dtype,
                        device=self.device,
                        generator=generator,
                    )
                    x = (1 - sigma) * x0 + sigma * noise
            self.kv_cache.advance()
            video = aura.decode(x0, first=r == 0)
            frames = (
                ((video.float() + 1) * 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 3, 0)
            )
        out = frames.cpu().numpy()
        self.timing = {"total_ms": (time.perf_counter() - wall) * 1000}
        self.chunk += 1
        return AvatarResult(
            frames=out, block=r, avatar_id=input.avatar_id, timing=self.timing
        )

    def reset(self) -> None:
        """Forget the bound portrait. The next ``generate`` binds again and starts a fresh clip."""
        self.avatar_id = None

    def _bind(self, portrait: bytes) -> None:
        """Encode the portrait at its closest training size and start a clip: a fresh cache, VAE state and history."""
        import torch
        import torchvision.transforms.functional as TF
        from aura.kv_cache import KVCache
        from PIL import Image

        cfg = self.cfg
        image = Image.open(io.BytesIO(portrait)).convert("RGB")
        ratio = image.height / image.width
        h, w = cfg.buckets[min(cfg.buckets, key=lambda k: abs(k - ratio))]
        image = TF.center_crop(TF.resize(image, min(h, w)), [h, w])
        ref = (TF.to_tensor(image) * 2 - 1)[None, :, None].to(self.device)
        with torch.no_grad():
            self.ref_latent = torch.stack(
                self.aura.vae.encode(list(ref.to(cfg.vae_dtype)))
            )
        self.lat_h, self.lat_w = h // cfg.vae_stride[1], w // cfg.vae_stride[2]
        self.kv_cache = KVCache(
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            head_dim=cfg.dim // cfg.num_heads,
            tokens_per_frame=(self.lat_h // cfg.patch_size[1])
            * (self.lat_w // cfg.patch_size[2]),
            frames_per_block=cfg.frames_per_block,
            window=cfg.kv_window,
            segment_frames=cfg.segment_frames,
            ref_rope_offset=cfg.ref_rope_offset,
            rope_max_block=cfg.segment_frames // cfg.frames_per_block - 1
            if cfg.rope_reindex
            else None,
            device=self.device,
        )
        self.aura.vae.model.clear_cache()
        self.chunk = 0
        self.audio_history = np.zeros(0, np.float32)

    def _features(self, audio: np.ndarray) -> Any:
        """Mimi features of the chunk's 16 frames, from the chunk's speech and the speech before it."""
        import torch

        block = audio.astype(np.float32) / 32768.0
        window = np.concatenate([self.audio_history, block])
        self.audio_history = (
            window[-self.history_samples :] if self.history_samples else window[:0]
        )
        pad = (-window.size) % MIMI_FRAME
        wav = torch.from_numpy(np.concatenate([window, np.zeros(pad, np.float32)])).to(
            self.device
        )[None, None]
        latent = (
            self.mimi.decode_latent(self.mimi.encode(wav)).transpose(1, 2).float()
        )  # [1, T @ 12.5 Hz, 512]
        count = latent.shape[1]
        positions = np.clip(
            feature_positions(self.chunk, window.size - block.size), 0, count - 1
        )
        low = np.floor(positions).astype(np.int64)
        high = np.minimum(low + 1, count - 1)
        weight = torch.from_numpy(positions - low).to(self.device, torch.float32)[
            :, None
        ]
        picked = latent[0, low] * (1 - weight) + latent[0, high] * weight  # [16, 512]
        return picked.T[None, None].to(torch.bfloat16)  # [1, 1, 512, 16]
