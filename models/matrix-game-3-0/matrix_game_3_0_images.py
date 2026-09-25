"""Validate uploaded Matrix-Game 3.0 input images."""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_FORMATS = {"BMP", "JPEG", "PNG", "WEBP"}


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    if not image.mime_type.startswith("image/"):
        raise CommandError("unsupported_media", f"{image.name} is not an image.")
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            width, height = decoded.size
            image_format = decoded.format
            if image_format not in _UPLOAD_FORMATS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.load()
    except CommandError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error
