"""Validate uploaded Echo-WM first-frame images."""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_MAX_PIXELS = 100_000_000
_FORMATS = {"BMP", "JPEG", "PNG", "WEBP"}


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject uploads that cannot serve as an Echo-WM first frame."""
    if not image.mime_type.startswith("image/"):
        raise CommandError("unsupported_media", f"{image.name} is not an image.")
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _MAX_UPLOAD_BYTES:
        raise CommandError("image_too_large", f"{image.name} exceeds 25 MiB.")
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            if decoded.format not in _FORMATS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > _MAX_PIXELS:
                raise CommandError(
                    "image_too_large", f"{image.name} exceeds {_MAX_PIXELS} pixels."
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
