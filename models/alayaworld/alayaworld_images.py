"""Validate client image uploads before model inference."""

import io

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_MIME_FORMATS = {
    "image/bmp": "BMP",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    expected_format = _UPLOAD_MIME_FORMATS.get(image.mime_type.lower())
    if expected_format is None:
        raise CommandError(
            "unsupported_media",
            f"{image.name} must declare image/jpeg, image/png, image/webp, or image/bmp.",
        )
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            image_format = decoded.format or ""
            width, height = decoded.size
            if image_format != expected_format:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} contains {image_format or 'unknown'} data but declares "
                    f"{image.mime_type}.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.verify()
    except CommandError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error
