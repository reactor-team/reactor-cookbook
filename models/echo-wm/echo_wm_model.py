"""Echo-WM model state and one native audiovisual chunk, independent of Reactor."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from echo_wm_assets import EchoWMConfig, load_examples
from echo_wm_camera import CameraChunk

if TYPE_CHECKING:
    from echo_wm_backend import EchoWMBackend

logger = logging.getLogger(__name__)


@contextmanager
def _materialized_image(
    value: Path | bytes, runtime_dir: Path, suffix: str
) -> Iterator[Path]:
    """Keep encoded input bytes on disk only while the native loader needs them."""
    if isinstance(value, Path):
        yield value
        return
    with tempfile.NamedTemporaryFile(
        prefix="echo-wm-upload-", suffix=suffix, dir=runtime_dir
    ) as temporary:
        temporary.write(value)
        temporary.flush()
        yield Path(temporary.name)


@dataclass(frozen=True)
class EchoAnchor:
    """Image and random seed used only when starting a new world."""

    image: Path | bytes
    suffix: str
    seed: int


@dataclass(frozen=True)
class EchoInput:
    """One world's anchor, text, and three latent-aligned camera poses."""

    world_id: int
    anchor: EchoAnchor | None
    prompt: str
    poses: np.ndarray
    fov_degrees: float


@dataclass(frozen=True)
class EchoResult:
    """Native media and the model's acknowledged world and progress."""

    video: np.ndarray
    audio: np.ndarray
    world_id: int
    chunk_index: int
    prompt: str
    complete: bool
    profile: dict[str, float]


class NoAnchor(Exception):
    """The requested world has no image to initialize its causal state."""


class RolloutExhausted(Exception):
    """The world has reached its configured native chunk capacity."""


class EchoModel:
    """Own the weights, native caches, and chunk count for one Echo-WM world."""

    def __init__(self) -> None:
        self._backend: EchoWMBackend | None = None
        self._config: EchoWMConfig | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._prompt = ""

    def load(self, config: EchoWMConfig) -> None:
        """Load the configured backend and warm it without selecting a client image."""
        from echo_wm_backend import EchoWMBackend

        self._config = config
        backend = EchoWMBackend(config)
        self._backend = backend
        benchmark = backend.attention_benchmark
        if benchmark is not None:
            logger.info("Echo-WM attention verification: %s", benchmark)
        self._warmup()
        logger.info(
            "Echo-WM Flash ready: attention=%s, modules=%s",
            config.attention_backend,
            backend.attention_modules,
        )

    def generate(self, input: EchoInput) -> EchoResult:
        """Apply an explicitly requested world, then decode one native block."""
        backend = self._backend
        config = self._config
        if backend is None or config is None:
            raise RuntimeError("Echo-WM was not loaded")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor(input.world_id)
            with _materialized_image(
                input.anchor.image, config.runtime_dir, input.anchor.suffix
            ) as image:
                backend.reset(
                    image=image,
                    prompt=input.prompt,
                    seed=input.anchor.seed,
                    fov_degrees=input.fov_degrees,
                )
            self._world_id = input.world_id
            self._chunk_index = 0
            self._prompt = input.prompt
        if self._chunk_index >= config.max_chunks:
            raise RolloutExhausted(input.world_id)
        video, audio = backend.generate_chunk(
            CameraChunk(latent_poses=input.poses), fov_degrees=input.fov_degrees
        )
        self._chunk_index += 1
        return EchoResult(
            video=video,
            audio=audio,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            prompt=self._prompt,
            complete=self._chunk_index >= config.max_chunks,
            profile=dict(backend.last_profile),
        )

    def reset(self) -> None:
        """Release the world's causal state while retaining the loaded weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
        self._prompt = ""

    def _warmup(self) -> None:
        """Run configured throwaway blocks with neutral poses before serving."""
        config = self._config
        backend = self._backend
        if config is None or backend is None:
            raise RuntimeError("Echo-WM was not loaded")
        if not config.warmup_chunks:
            return
        scene = load_examples(config)[0]
        camera = CameraChunk(
            latent_poses=np.tile(
                np.eye(4, dtype=np.float32), (config.video_chunk_size, 1, 1)
            )
        )
        try:
            backend.reset(
                image=scene.image,
                prompt=scene.prompt,
                seed=scene.seed,
                fov_degrees=scene.fov_degrees,
            )
            for _ in range(config.warmup_chunks):
                backend.generate_chunk(camera, fov_degrees=scene.fov_degrees)
        finally:
            backend.end_session(release_cuda_cache=False)
        logger.info("Echo-WM warmup complete: %s chunks", config.warmup_chunks)
