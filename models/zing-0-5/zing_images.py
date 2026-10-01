"""Decode and center-fit Zing image conditions once on the CPU."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_MAX_BYTES = 25 * 1024 * 1024
_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def prepare_image(
    upload: UploadedFile, width: int = 1248, height: int = 704
) -> np.ndarray:
    if len(upload.data) > _MAX_BYTES:
        raise CommandError(
            "image_too_large", "The first-frame image must be at most 25 MiB."
        )
    try:
        with Image.open(io.BytesIO(upload.data)) as image:
            if image.format not in _FORMATS:
                raise CommandError(
                    "image_format", "Use a JPEG, PNG, WebP, or BMP image."
                )
            if image.width * image.height > 100_000_000:
                raise CommandError(
                    "image_dimensions",
                    "The image must contain at most 100 million pixels.",
                )
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(
                ImageOps.fit(rgb, (width, height), method=Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )
    except (
        Image.DecompressionBombError,
        ValueError,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise CommandError(
            "invalid_image", "The uploaded bytes are not a readable image."
        ) from exc
