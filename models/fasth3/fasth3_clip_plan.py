"""Clip geometry for the FastH3 channel.

Pure arithmetic over the checkpoint's published constraints: how long a clip may
be, how many frames that is, and what canvas an aspect ratio resolves to. No
torch, no fastvideo, no GPU — so ``python -m reactor_runtime.schema`` and the
structural tests import it on any machine.

The constants below are duplicated from FastVideo rather than imported, because
importing ``fastvideo.pipelines.basic.minimax_h3.packing`` drags in torch.
``tests/test_fasth3.py`` asserts they still match upstream whenever fastvideo
is importable, so the duplication cannot drift silently.
"""

from __future__ import annotations

import math

FPS = 24
"""The only frame rate MiniMax-H3 accepts; the pipeline rejects anything else."""

# The causal VAE consumes video in 17-frame chunks that decode to 5 latents, so
# a valid pixel length is always `17n + 5`.
_FRAMES_PER_CHUNK = 17
_LATENTS_PER_CHUNK = 5

# The checkpoint's trained duration window, in seconds.
_MIN_DURATION = 5.0
_MAX_DURATION = 15.0

# Canvas rules: the short edge is fixed, total area is capped, and both sides
# must land on a multiple of 32.
_SHORT_EDGE = 768
_MAX_PIXELS = 768 * 1344
_CANVAS_MULTIPLE = 32
_MIN_ASPECT = 1 / 4
_MAX_ASPECT = 4


def align_frames(frames: int) -> int:
    """Round up to the next valid `17n + 5` pixel length."""
    if frames < 1:
        raise ValueError(f"frames must be positive, got {frames}")
    while frames % _FRAMES_PER_CHUNK != _LATENTS_PER_CHUNK:
        frames += 1
    return frames


def _bounds() -> tuple[int, int]:
    """The shortest and longest clip that satisfies both alignment and duration.

    The ceiling is the subtle one: 15.0 s is 360 frames, which aligns *up* to
    362 (15.083 s) and is then rejected for exceeding the duration cap. So the
    longest clip this checkpoint will actually generate is 345 frames.
    """
    low = align_frames(int(_MIN_DURATION * FPS))
    high = low
    while True:
        nxt = align_frames(high + 1)
        if nxt / FPS > _MAX_DURATION:
            return low, high
        high = nxt


MIN_FRAMES, MAX_FRAMES = _bounds()
MIN_SECONDS = MIN_FRAMES / FPS
MAX_SECONDS = MAX_FRAMES / FPS

# The same bounds as the schema publishes them. Rounded *inward* to three
# decimals so a client reads "5.167", not "5.166666666666667", and so every
# value inside the published range still snaps to a generatable clip.
MIN_SECONDS_PUBLISHED = math.ceil(MIN_SECONDS * 1000) / 1000
MAX_SECONDS_PUBLISHED = math.floor(MAX_SECONDS * 1000) / 1000


def frames_for_seconds(seconds: float) -> int:
    """Snap a requested clip length to the nearest length the model can make.

    Rounds up to a valid frame count, then clamps into the generatable range, so
    every accepted value round-trips through ``seconds_for_frames``.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    frames = align_frames(max(1, round(seconds * FPS)))
    return max(MIN_FRAMES, min(MAX_FRAMES, frames))


def seconds_for_frames(frames: int) -> float:
    """Exact playout length of a clip, in seconds."""
    return frames / FPS


def canvas_for_aspect(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve an aspect ratio to a `(height, width)` the checkpoint accepts.

    Mirrors FastVideo's ``resolve_canvas_size``: pin the short edge to 768,
    shrink to the area cap if the result is too wide, then round both sides to a
    multiple of 32.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"aspect must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not _MIN_ASPECT <= ratio <= _MAX_ASPECT:
        raise ValueError(f"aspect ratios run from 1:4 to 4:1, got {aspect_width}:{aspect_height}")

    if ratio >= 1:
        width, height = _SHORT_EDGE * ratio, float(_SHORT_EDGE)
    else:
        width, height = float(_SHORT_EDGE), _SHORT_EDGE / ratio
    area = width * height
    if area > _MAX_PIXELS:
        scale = (_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    m = _CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


# The canvases `set_canvas` offers. Deliberately a short list: every entry is a
# distinct tensor shape that load() must warm, and an unwarmed shape pays a
# one-off compile stall on its first clip.
ASPECT_CHOICES: tuple[str, ...] = ("16:9", "1:1", "9:16", "4:3")

_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9),
    "1:1": (1, 1),
    "9:16": (9, 16),
    "4:3": (4, 3),
}


def canvas_for_choice(aspect: str) -> tuple[int, int]:
    """Resolve one of ``ASPECT_CHOICES`` to `(height, width)`."""
    try:
        ratio = _ASPECT_RATIOS[aspect]
    except KeyError:
        raise ValueError(f"unknown aspect {aspect!r}; choose one of {list(ASPECT_CHOICES)}") from None
    return canvas_for_aspect(*ratio)


def clip_frames(index: int, steady_frames: int, ramp_frames: tuple[int, ...]) -> int:
    """Frame count for clip ``index`` of a channel run.

    The ramp exists purely for time-to-first-frame: the channel opens with one or
    more short clips (which generate proportionally faster) and then settles on
    ``steady_frames``. An empty ramp means a uniform cadence from clip zero.
    """
    if index < 0:
        raise ValueError(f"clip index must be non-negative, got {index}")
    if index < len(ramp_frames):
        return ramp_frames[index]
    return steady_frames


def parse_ramp(seconds: object) -> tuple[int, ...]:
    """Turn a config list of clip lengths in seconds into snapped frame counts."""
    if seconds is None:
        return ()
    if not isinstance(seconds, list | tuple):
        raise TypeError(f"ramp must be a list of seconds, got {type(seconds).__name__}")
    return tuple(frames_for_seconds(float(value)) for value in seconds)


__all__ = [
    "ASPECT_CHOICES",
    "FPS",
    "MAX_FRAMES",
    "MAX_SECONDS",
    "MAX_SECONDS_PUBLISHED",
    "MIN_FRAMES",
    "MIN_SECONDS",
    "MIN_SECONDS_PUBLISHED",
    "align_frames",
    "canvas_for_aspect",
    "canvas_for_choice",
    "clip_frames",
    "frames_for_seconds",
    "parse_ramp",
    "seconds_for_frames",
]
