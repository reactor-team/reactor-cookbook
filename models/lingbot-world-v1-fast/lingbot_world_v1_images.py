"""Validate LingBot anchor uploads and normalize generated frames."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000
_MIME_FORMATS = {
    "image/bmp": "BMP",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def validate_uploaded_image(upload: UploadedFile) -> None:
    """Validate an uploaded image before replacing the active world anchor."""
    if upload.size <= 0:
        raise CommandError("image_empty", "Upload a non-empty anchor image.")
    if upload.size > _MAX_UPLOAD_BYTES:
        raise CommandError("image_too_large", "Anchor images must be at most 25 MiB.")
    expected = _MIME_FORMATS.get(upload.mime_type.lower())
    if expected is None:
        raise CommandError(
            "image_type_unsupported",
            "Anchor images must be JPEG, PNG, WebP, or BMP.",
        )
    try:
        with Image.open(io.BytesIO(upload.data)) as image:
            actual = image.format
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise CommandError(
            "image_invalid", "The uploaded anchor image cannot be decoded."
        ) from error
    if actual != expected:
        raise CommandError(
            "image_type_mismatch",
            f"The upload declares {expected} but contains {actual or 'unknown'} data.",
        )
    if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
        raise CommandError(
            "image_dimensions_invalid",
            "Anchor images must contain at most 100 million pixels.",
        )


def normalize_output_frames(frames: np.ndarray) -> np.ndarray:
    """Return contiguous uint8 RGB frames suitable for Reactor's video track."""
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise RuntimeError(
            f"LingBot output must have shape (T, H, W, 3); got {value.shape}"
        )
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)
