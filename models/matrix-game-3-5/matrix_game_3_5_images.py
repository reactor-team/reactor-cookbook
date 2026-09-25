"""Validate uploaded Matrix input images."""

from __future__ import annotations

import io

import av
from reactor_runtime import CommandError, UploadedFile

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_CODECS = {"bmp", "mjpeg", "png", "webp"}


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
        with av.open(io.BytesIO(image.data), mode="r") as container:
            if not container.streams.video:
                raise CommandError(
                    "invalid_image", f"{image.name} has no image stream."
                )
            stream = container.streams.video[0]
            codec = stream.codec_context.name
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            if codec not in _UPLOAD_CODECS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            frame = next(container.decode(stream), None)
            if frame is None or frame.width != width or frame.height != height:
                raise CommandError("invalid_image", f"{image.name} cannot be decoded.")
    except CommandError:
        raise
    except (av.FFmpegError, EOFError, OSError, ValueError) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error
