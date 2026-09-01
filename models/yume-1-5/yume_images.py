"""Validation and temporary materialization for uploaded YUME images."""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 100_000_000
FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}
MAX_VIDEO_BYTES = 500 * 1024 * 1024


def validate_image(image: UploadedFile) -> None:
    """Accept only bounded image formats decoded by the upstream PIL path."""
    if not image.mime_type.startswith("image/") or not image.data:
        raise CommandError("invalid_image", f"{image.name} is not a non-empty image.")
    if image.size > MAX_BYTES:
        raise CommandError("image_too_large", f"{image.name} exceeds 25 MiB.")
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            if decoded.format not in FORMATS:
                raise CommandError(
                    "unsupported_media", "Use JPEG, PNG, WebP, BMP, or TIFF."
                )
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                raise CommandError(
                    "image_too_large", f"{image.name} exceeds {MAX_PIXELS} pixels."
                )
            decoded.verify()
    except CommandError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error


def validate_video(video: UploadedFile) -> None:
    """Accept a bounded decodable video with at least 33 conditioning frames."""
    if not video.mime_type.startswith("video/") or not video.data:
        raise CommandError("invalid_video", f"{video.name} is not a non-empty video.")
    if video.size > MAX_VIDEO_BYTES:
        raise CommandError("video_too_large", f"{video.name} exceeds 500 MiB.")
    try:
        import av

        with av.open(io.BytesIO(video.data)) as container:
            for count, _frame in enumerate(container.decode(video=0), start=1):
                if count >= 33:
                    return
    except (av.AVError, IndexError, OSError, ValueError) as error:
        raise CommandError(
            "invalid_video", f"{video.name} cannot be decoded."
        ) from error
    raise CommandError(
        "video_too_short", f"{video.name} must contain at least 33 frames."
    )


@contextmanager
def materialized_image(image: UploadedFile, runtime_dir: Path) -> Iterator[Path]:
    """Yield a local path consumable by PIL and remove it afterwards."""
    suffix = Path(image.name).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(
        prefix="yume-upload-", suffix=suffix, dir=runtime_dir
    ) as temporary:
        temporary.write(image.data)
        temporary.flush()
        yield Path(temporary.name)


@contextmanager
def materialized_video(video: UploadedFile, runtime_dir: Path) -> Iterator[Path]:
    """Yield a temporary local video path accepted by PyAV."""
    suffix = Path(video.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(
        prefix="yume-video-", suffix=suffix, dir=runtime_dir
    ) as temporary:
        temporary.write(video.data)
        temporary.flush()
        yield Path(temporary.name)
