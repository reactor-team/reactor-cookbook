"""Bounded CPU preparation of uploaded YUME conditioning media."""

from __future__ import annotations

import io
import av
import numpy as np
from PIL import Image
from reactor_runtime import CommandError, UploadedFile

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 100_000_000
FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}
MAX_VIDEO_BYTES = 500 * 1024 * 1024


def prepare_image(image: UploadedFile) -> np.ndarray:
    """Decode once to native uint8 RGB (704,1280,3), preserving bilinear resize."""
    if not image.mime_type.startswith("image/") or not image.data:
        raise CommandError("invalid_image", "Supply a nonempty image.")
    if image.size > MAX_BYTES:
        raise CommandError("image_too_large", "Image exceeds 25 MiB.")
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            if decoded.format not in FORMATS:
                raise CommandError(
                    "unsupported_media", "Use JPEG, PNG, WebP, BMP, or TIFF."
                )
            if decoded.width * decoded.height > MAX_PIXELS:
                raise CommandError(
                    "image_too_large", "Image exceeds 100 million pixels."
                )
            return np.array(
                decoded.convert("RGB").resize((1280, 704), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
    except (Image.DecompressionBombError, OSError, ValueError) as error:
        raise CommandError("invalid_image", "Image cannot be decoded.") from error


def prepare_video(video: UploadedFile) -> np.ndarray:
    """Decode only the first 33 frames; preserve the native bicubic resize."""
    if not video.mime_type.startswith("video/") or not video.data:
        raise CommandError("invalid_video", "Supply a nonempty video.")
    if video.size > MAX_VIDEO_BYTES:
        raise CommandError("video_too_large", "Video exceeds 500 MiB.")
    frames = []
    try:
        with av.open(io.BytesIO(video.data)) as container:
            for frame in container.decode(video=0):
                if frame.width * frame.height > MAX_PIXELS:
                    raise CommandError(
                        "video_too_large", "Video frame exceeds 100 million pixels."
                    )
                image = (
                    frame.to_image()
                    .convert("RGB")
                    .resize((1280, 704), Image.Resampling.BICUBIC)
                )
                frames.append(np.asarray(image, dtype=np.uint8))
                if len(frames) == 33:
                    return np.ascontiguousarray(np.stack(frames))
    except (av.FFmpegError, IndexError, OSError, ValueError) as error:
        raise CommandError("invalid_video", "Video cannot be decoded.") from error
    raise CommandError("video_too_short", "Video must contain at least 33 frames.")
