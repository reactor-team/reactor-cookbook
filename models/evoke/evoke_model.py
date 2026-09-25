"""Own EVOKE's native worker and causal world independently of Reactor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from upstream_backend import EvokeWorkerBackend, WorkerSettings


@dataclass(frozen=True)
class EvokeAnchor:
    """Conditioning needed only when starting an explicitly requested world."""

    mode: str
    media: Path | bytes | None
    media_suffix: str
    pose: Path | bytes | None
    pose_suffix: str
    seed: int
    source_fps: int
    source_height: int
    source_width: int


@dataclass(frozen=True)
class EvokeInput:
    """One world's identity, optional conditioning, prompt and camera poses."""

    world_id: int
    anchor: EvokeAnchor | None
    prompt: str
    trajectory: np.ndarray | None


@dataclass(frozen=True)
class EvokeResult:
    """Native frames and the model's own completed-world facts."""

    frames: np.ndarray
    world_id: int
    chunk_index: int
    seed: int
    complete: bool


class NoAnchor(Exception):
    """The requested world has no initial conditioning."""


class RolloutExhausted(Exception):
    """The world has reached its configured native rollout horizon."""


class EvokeModel:
    """Keep the native worker, rollout identity and successful chunk count."""

    def __init__(self) -> None:
        self._backend: EvokeWorkerBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._max_chunks = 0
        self._seed = 0
        self._mode = ""

    def load(self, settings: WorkerSettings) -> None:
        """Load the persistent inference worker from resolved settings."""
        self._backend = EvokeWorkerBackend(settings)
        self._max_chunks = settings.max_chunks

    def generate(self, input: EvokeInput) -> EvokeResult:
        """Start the requested world or advance it by one native chunk."""
        if self._backend is None:
            raise RuntimeError("EVOKE model was not loaded")
        if input.world_id != self._world_id:
            anchor = input.anchor
            if anchor is None or (anchor.mode != "t2v" and anchor.media is None):
                raise NoAnchor("A fresh EVOKE world requires explicit conditioning")
            if anchor.mode == "v2v" and anchor.pose is None:
                raise NoAnchor("A video-conditioned world requires its camera track")
            self._backend.reset(
                mode=anchor.mode,
                media=anchor.media,
                pose=anchor.pose,
                media_suffix=anchor.media_suffix,
                pose_suffix=anchor.pose_suffix,
                prompt=input.prompt,
                seed=anchor.seed,
                source_fps=anchor.source_fps,
                source_height=anchor.source_height,
                source_width=anchor.source_width,
            )
            self._world_id = input.world_id
            self._chunk_index = 0
            self._seed = anchor.seed
            self._mode = anchor.mode
        if self._chunk_index >= self._max_chunks:
            raise RolloutExhausted(
                "EVOKE requires an explicit new world at its horizon"
            )
        frames = normalize_output_frames(
            self._backend.generate_chunk(
                input.trajectory,
                seed=self._seed,
                prompt=input.prompt,
            )
        )
        expected = 33 if self._mode == "t2v" and self._chunk_index == 0 else 36
        if int(frames.shape[0]) != expected:
            raise RuntimeError(
                f"EVOKE chunk {self._chunk_index + 1} produced {frames.shape[0]} frames; expected {expected}"
            )
        self._chunk_index += 1
        return EvokeResult(
            frames,
            input.world_id,
            self._chunk_index,
            self._seed,
            self._chunk_index >= self._max_chunks,
        )

    def reset(self) -> None:
        """Release session state and uploads while retaining loaded weights."""
        if self._backend is not None:
            self._backend.end_session()
        self._world_id = None
        self._chunk_index = 0
        self._seed = 0
        self._mode = ""


def normalize_output_frames(frames: np.ndarray) -> np.ndarray:
    """Return contiguous uint8 RGB frames with shape ``(T, H, W, 3)``."""
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise RuntimeError(
            f"EVOKE output must have shape (T, H, W, 3), got {value.shape}"
        )
    if value.dtype != np.uint8:
        value = np.asarray(value, dtype=np.float32)
        if float(np.nanmin(value)) < -0.05:
            value = (value + 1.0) * 127.5
        elif float(np.nanmax(value)) <= 1.5:
            value = value * 255.0
        value = np.clip(value, 0.0, 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(value)
