"""Bounded image validation and MovieRoll's native cover/center-crop fit."""

import io
import warnings

import numpy as np
from PIL import Image, ImageOps

from movieroll_model import HEIGHT, WIDTH

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000


class ImageError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode_image(data: bytes, mime_type: str) -> np.ndarray:
    if mime_type not in ("image/png", "image/jpeg", "image/webp"):
        raise ImageError("unsupported_image", "Choose a PNG, JPEG, or WebP image")
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageError("image_size", "Choose an image smaller than 20 MiB")
    expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                if opened.format != expected[mime_type]:
                    raise ImageError(
                        "invalid_image", "The image type does not match its contents"
                    )
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ImageError(
                        "image_size", "Choose an image with at most 32 million pixels"
                    )
                if (
                    opened.width < 16
                    or opened.height < 16
                    or getattr(opened, "is_animated", False)
                ):
                    raise ImageError(
                        "invalid_image",
                        "Choose a still image at least 16 pixels on each side",
                    )
                image = ImageOps.exif_transpose(opened).convert("RGB")
    except ImageError:
        raise
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageError("invalid_image", "The image could not be decoded") from exc
    # Matches torchvision Resize + CenterCrop in upstream encode_input_image:
    # round the covered size first, then round the centered crop coordinates.
    scale = max(WIDTH / image.width, HEIGHT / image.height)
    width, height = round(image.width * scale), round(image.height * scale)
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageError(
            "image_size", "Choose an image with a less extreme aspect ratio"
        )
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = round((width - WIDTH) / 2), round((height - HEIGHT) / 2)
    return np.array(image.crop((left, top, left + WIDTH, top + HEIGHT)), dtype=np.uint8)
