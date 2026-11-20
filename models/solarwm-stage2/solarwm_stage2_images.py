"""Validate SolarWM anchor uploads and normalize generated frames."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000
_HEIGHT, _WIDTH = 480, 864
_MIME_FORMATS = {
    "image/bmp": "BMP",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def prepare_uploaded_image(upload: UploadedFile) -> np.ndarray:
    """Validate an uploaded image and return its prepared (3,480,864) float32 anchor."""
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
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise CommandError(
                    "image_dimensions_invalid",
                    "Anchor images must contain at most 100 million pixels.",
                )
            if actual != expected:
                raise CommandError(
                    "image_type_mismatch",
                    f"The upload declares {expected} but contains {actual or 'unknown'} data.",
                )
            rgb = np.array(
                ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8, copy=True
            )
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        UnidentifiedImageError,
    ) as error:
        raise CommandError(
            "image_invalid", "The uploaded anchor image cannot be decoded."
        ) from error
    return anchor_pixels(rgb)


def anchor_pixels(rgb: np.ndarray) -> np.ndarray:
    """Return SolarWM's (3,480,864) anchor in [-1,1], as upstream decode_video makes it.

    Upstream (backends/wan22/runtime/data.py decode_video) scales uint8 frames
    to [0,1], resizes them with torch bilinear interpolation (no antialias) to
    cover 480x864, center-crops, maps to [-1,1] and stores bfloat16. The VAE
    then encodes that tensor as float32, so these float32 values are exact.
    """
    import torch
    import torch.nn.functional as functional

    frames = torch.from_numpy(rgb)[None].permute(0, 3, 1, 2).contiguous().float() / 255.0
    _, _, source_height, source_width = frames.shape
    if (source_height, source_width) != (_HEIGHT, _WIDTH):
        scale = max(_HEIGHT / source_height, _WIDTH / source_width)
        resized_height = round(source_height * scale)
        resized_width = round(source_width * scale)
        if (resized_height, resized_width) != (source_height, source_width):
            frames = functional.interpolate(
                frames,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
        top = (resized_height - _HEIGHT) // 2
        left = (resized_width - _WIDTH) // 2
        frames = frames[:, :, top : top + _HEIGHT, left : left + _WIDTH]
    pixels = (frames * 2.0 - 1.0).to(torch.bfloat16)
    return np.ascontiguousarray(pixels[0].float().numpy())


def normalize_output_frames(frames: np.ndarray) -> np.ndarray:
    """Return contiguous uint8 RGB frames suitable for Reactor's video track."""
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise RuntimeError(
            f"SolarWM output must have shape (T, H, W, 3); got {value.shape}"
        )
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)
