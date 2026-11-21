"""LiveAvatar, the model half.

Plain Python around :class:`liveavatar_engine.BlockEngine`. Nothing here knows
about clients, tracks, commands or rooms, and nothing here imports
``reactor_runtime``. :class:`LiveAvatarModel` has three methods: ``load`` loads
the weights and renders a few warm-up blocks, ``generate`` renders one 0.48 s
block from 0.48 s of speech, and ``reset`` forgets the bound portrait so the
next block starts a fresh clip.

It implements :class:`avatar_contract.AvatarModel`: the application half and
this one meet on :class:`avatar_contract.AvatarInput` and
:class:`avatar_contract.AvatarResult`, plain data and CPU arrays.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from avatar_contract import AvatarInput, AvatarResult
from liveavatar_engine import BLOCK_FRAMES, ENCODER_RATE, FPS, BlockEngine

#: Rate of the speech a block carries, and of the audio that plays with the frames.
SAMPLE_RATE = 24_000
#: Samples of speech one block consumes: 0.48 s at :data:`SAMPLE_RATE`.
BLOCK_SAMPLES = BLOCK_FRAMES * SAMPLE_RATE // FPS
#: Samples of earlier speech kept so each block resamples without an edge at its start.
_RESAMPLE_CONTEXT = 240

DEFAULT_PROMPT = (
    "A person faces the camera and talks naturally, with clear lip movements, lively facial expressions "
    "and small, natural head movements. The background stays still. The lighting is soft and even."
)
DEFAULT_LISTENING_PROMPT = (
    "A person faces the camera in complete silence, listening. The lips are closed and do not move, the jaw is "
    "relaxed, and the head is still. Only the eyes blink now and then. The background stays still. The lighting "
    "is soft and even."
)


class NoAvatar(Exception):
    """A fresh clip is asked for and the block's input carries no portrait to bind."""


class LiveAvatarModel:
    """LiveAvatar behind ``load`` / ``generate`` / ``reset``.

    Holds the engine, the id of the portrait it bound, and the speech it last read, which the next
    block resamples from. A block marked ``silent`` or ``listening`` renders on the listening prompt:
    on the speaking prompt the face keeps moving as if it talked.
    """

    fps = FPS
    block_frames = BLOCK_FRAMES
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.engine: BlockEngine | None = None
        self.avatar_id: int | None = None
        self._tail = np.zeros(_RESAMPLE_CONTEXT, dtype=np.float32)

    def load(
        self,
        weights_root: Path,
        *,
        source: Path,
        width: int = 512,
        height: int = 512,
        prompt: str = DEFAULT_PROMPT,
        listening_prompt: str | None = DEFAULT_LISTENING_PROMPT,
        audio_context_s: float = 4.0,
        seed: int = 420,
        compile_dit: bool = True,
        fp8: bool = False,
        warmup_avatar: bytes | None = None,
        warmup_blocks: int = 6,
    ) -> None:
        """Load the weights on the GPU and render ``warmup_blocks`` blocks from ``warmup_avatar``.

        Args:
            weights_root: The folder with ``wan2_2/`` (Wan2.2-S2V-14B) and ``liveavatar_lora/``.
            source: The pinned upstream LiveAvatar checkout.
            width: Output width, a multiple of 64.
            height: Output height, a multiple of 64.
            prompt: The scene and performance text blocks with speech are conditioned on.
            listening_prompt: The text silent and listening blocks are conditioned on, or ``None`` for
                ``prompt`` on every block.
            audio_context_s: Seconds of earlier speech the audio encoder reads with each block.
            seed: Seed of the clip noise.
            compile_dit: Compile the denoiser's forward with ``torch.compile``. The warm-up then pays the
                compilation, so the first session does not.
            fp8: Run the transformer's projections in FP8, which changes pixels slightly.
            warmup_avatar: A portrait for the warm-up blocks; required when ``warmup_blocks`` is positive.
            warmup_blocks: Blocks rendered at load, half of them speech-like and half silent.
        """
        engine = BlockEngine(
            source,
            weights_root / "wan2_2",
            weights_root / "liveavatar_lora",
            height=height,
            width=width,
            prompt=prompt,
            listening_prompt=listening_prompt,
            seed=seed,
            audio_context_s=audio_context_s,
        )
        engine.load(compile_dit=compile_dit, fp8=fp8)
        self.engine = engine
        if warmup_blocks:
            if warmup_avatar is None:
                raise ValueError("warm-up blocks need a portrait")
            self._warmup(warmup_avatar, warmup_blocks)

    def generate(self, input: AvatarInput) -> AvatarResult:
        """Render one block.

        Binds ``input.avatar`` first when ``input.avatar_id`` is not the portrait this model last bound,
        which starts a new clip. Then renders the block from the block's speech.

        Raises:
            NoAvatar: A new clip is asked for and the input carries no portrait.
            RuntimeError: ``load`` has not run.
        """
        engine = self._engine()
        if input.avatar_id != self.avatar_id or not engine.bound:
            if input.avatar is None:
                raise NoAvatar("no portrait to start a clip from")
            engine.bind(crop_portrait(input.avatar, engine.width, engine.height))
            self.avatar_id = input.avatar_id
            self._tail[:] = 0
        block = engine.block
        frames = engine.render(
            self._to_encoder_rate(input.audio),
            listening=input.silent or input.listening,
        )
        t = engine.timing
        timing = {
            "audio_ms": t.audio_ms,
            "dit_ms": t.dit_ms,
            "decode_ms": t.decode_ms,
            "total_ms": t.total_ms,
        }
        return AvatarResult(
            frames=frames, block=block, avatar_id=input.avatar_id, timing=timing
        )

    def reset(self) -> None:
        """Forget the bound portrait. The next ``generate`` binds again and starts a fresh clip."""
        self.avatar_id = None
        if self.engine is not None:
            self.engine.reset()

    @property
    def size(self) -> tuple[int, int]:
        """The output ``(width, height)``."""
        engine = self._engine()
        return engine.width, engine.height

    def _engine(self) -> BlockEngine:
        if self.engine is None:
            raise RuntimeError("load() has not run")
        return self.engine

    def _to_encoder_rate(self, audio: np.ndarray) -> np.ndarray:
        """Resample one block of 24 kHz int16 speech to the encoder's 16 kHz float, carrying the edge."""
        from scipy.signal import resample_poly

        if audio.shape != (BLOCK_SAMPLES,):
            raise ValueError(
                f"a block carries {BLOCK_SAMPLES} samples at {SAMPLE_RATE} Hz"
            )
        x = np.concatenate([self._tail, audio.astype(np.float32) / 32768.0])
        self._tail = x[-_RESAMPLE_CONTEXT:]
        y = resample_poly(x, ENCODER_RATE // 8000, SAMPLE_RATE // 8000)
        return np.clip(
            y[_RESAMPLE_CONTEXT * ENCODER_RATE // SAMPLE_RATE :], -1, 1
        ).astype(np.float32)

    def _warmup(self, avatar: bytes, blocks: int) -> None:
        voiced = speech_like(BLOCK_SAMPLES)
        silence = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
        for n in range(blocks):
            audio = voiced if n % 2 == 0 else silence
            self.generate(
                AvatarInput(
                    audio=audio,
                    silent=n % 2 == 1,
                    listening=False,
                    avatar=avatar,
                    avatar_id=-1,
                )
            )
        self.reset()


def crop_portrait(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode a portrait, scale it to cover ``width`` x ``height``, and crop the centre. RGB uint8."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        scale = max(width / image.width, height / image.height)
        size = (
            max(width, round(image.width * scale)),
            max(height, round(image.height * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
        left, top = (size[0] - width) // 2, (size[1] - height) // 2
        return np.asarray(
            image.crop((left, top, left + width, top + height)), dtype=np.uint8
        )


def read_audio(path: Path) -> np.ndarray:
    """Decode any audio file ffmpeg reads to mono float32 at the encoder's rate."""
    import subprocess

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(ENCODER_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def speech_like(n: int) -> np.ndarray:
    """A voiced-looking int16 signal for warm-up blocks; a block's cost does not depend on its words."""
    t = np.arange(n) / SAMPLE_RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    tone = np.sin(2 * np.pi * 180.0 * t) + 0.5 * np.sin(2 * np.pi * 360.0 * t)
    return (6000 * envelope * tone).astype(np.int16)
