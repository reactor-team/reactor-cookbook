"""Validate ABot-World starting images and preserve their loader suffixes."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
SUPPORTED_MIME_TYPES = frozenset(("image/jpeg", "image/png", "image/webp", "image/bmp"))


def validate_uploaded_image(upload: UploadedFile) -> None:
    """Require a bounded JPEG, PNG, WebP, or BMP that Pillow can decode."""
    if upload.size > MAX_UPLOAD_BYTES:
        raise CommandError(
            "image_too_large", "The starting image must be at most 25 MiB."
        )
    if upload.mime_type.lower() not in SUPPORTED_MIME_TYPES:
        raise CommandError(
            "unsupported_media",
            "The starting image must be a JPEG, PNG, WebP, or BMP upload.",
        )
    try:
        with Image.open(io.BytesIO(upload.data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise CommandError(
                    "invalid_image_dimensions",
                    "The starting image must contain at most 100 million pixels.",
                )
            image.verify()
    except CommandError:
        raise
    except (OSError, RuntimeError, ValueError, UnidentifiedImageError) as error:
        raise CommandError(
            "invalid_image", "The uploaded starting image could not be decoded."
        ) from error


def upload_suffix(source: UploadedFile) -> str:
    """Return a supported filename suffix for the native first-frame loader."""
    suffix = Path(source.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".img"
    return suffix
