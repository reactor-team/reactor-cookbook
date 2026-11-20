"""Native avatar take lifecycle and audiovisual chunks, independent of Reactor."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LiveAvatarSettings:
    """Resolved, picklable deployment settings; no model-side environment lookup."""

    source: Path
    base: Path
    lora: Path
    turbo: bool = True


@dataclass(frozen=True)
class TakeConditions:
    """App-prepared RGB image, mono 16 kHz WAV and optional native pose video.

    Paths remain valid until the take ends. Conditions accompany a new take ID
    until its first successful result acknowledges that ID.
    """

    image: Path
    audio: Path
    pose: Path | None
    prompt: str
    negative_prompt: str
    seed: int
    max_chunks: int


@dataclass(frozen=True)
class LiveAvatarInput:
    """One requested clip; conditions are omitted after the take is acknowledged."""

    take_id: int
    conditions: TakeConditions | None


@dataclass(frozen=True)
class LiveAvatarResult:
    """Actual progress and CPU media: uint8 (T,H,W,3), float32 (1,T*1920).

    T is 45 for the first clip and 48 thereafter. Completion carries no media.
    """

    take_id: int
    video: np.ndarray | None
    audio: np.ndarray | None
    chunks: int
    frames: int
    complete: bool


class TakeFailed(RuntimeError):
    """A recoverable failure belonging to one take, including across worker IPC."""

    def __init__(self, take_id: int, reason: str):
        super().__init__(take_id, reason)
        self.take_id = take_id
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class LiveAvatarModel:
    def __init__(self):
        self.rank: int = 0
        self.world_size: int = 1
        self._backend = None
        self._take_id = None
        self._chunks = self._frames = 0
        self._complete = False

    def load(self, settings: LiveAvatarSettings) -> None:
        from liveavatar_parallel import ParallelBackend

        self._backend = ParallelBackend(
            settings=settings, rank=self.rank, world_size=self.world_size
        )

    def generate(self, input: LiveAvatarInput) -> LiveAvatarResult:
        if self._backend is None:
            raise RuntimeError("LiveAvatar model was not loaded")
        if input.take_id != self._take_id:
            conditions = input.conditions
            if conditions is None:
                raise ValueError("A new take requires image and audio conditions")
            self.reset()
            try:
                self._backend.start(
                    image=conditions.image,
                    audio=conditions.audio,
                    pose=conditions.pose,
                    prompt=conditions.prompt,
                    negative_prompt=conditions.negative_prompt,
                    seed=conditions.seed,
                    max_chunks=conditions.max_chunks,
                )
            except Exception as error:
                raise TakeFailed(
                    input.take_id, str(error) or type(error).__name__
                ) from error
            self._take_id = input.take_id
        if self._complete:
            raise RuntimeError("The take completed; start a new take before generating")
        try:
            media = self._backend.next()
        except Exception as error:
            raise TakeFailed(
                input.take_id, str(error) or type(error).__name__
            ) from error
        if media is None:
            self._complete = True
            return LiveAvatarResult(
                take_id=input.take_id,
                video=None,
                audio=None,
                chunks=self._chunks,
                frames=self._frames,
                complete=True,
            )
        video, audio = media
        expected = 45 if self._chunks == 0 else 48
        if (
            video.dtype != np.uint8
            or video.ndim != 4
            or video.shape[0] != expected
            or video.shape[-1] != 3
        ):
            raise ValueError("Native clip must contain 45/48 RGB frames")
        if (
            audio.dtype != np.float32
            or audio.shape != (1, expected * 48000 // 25)
            or not np.isfinite(audio).all()
            or (np.abs(audio) > 1).any()
        ):
            raise ValueError(
                "Native clip requires synchronized finite mono 48 kHz audio"
            )
        self._chunks += 1
        self._frames += len(video)
        return LiveAvatarResult(
            take_id=input.take_id,
            video=video,
            audio=audio,
            chunks=self._chunks,
            frames=self._frames,
            complete=False,
        )

    def reset(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._take_id = None
        self._chunks = self._frames = 0
        self._complete = False
