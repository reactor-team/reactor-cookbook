"""Validate and center-crop the first image as in Apple's interactive demo."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from fastframes_lingbot_model import HEIGHT, WIDTH

MAX_IMAGE_BYTES = 8 * 1024**2
MAX_IMAGE_PIXELS = 16_000_000


def prepare_image(raw: bytes) -> np.ndarray:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Choose an image smaller than 8 MiB")
    try:
        with Image.open(BytesIO(raw), formats=("JPEG", "PNG", "WEBP")) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Choose an image smaller than 16 megapixels")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            cropped = ImageOps.fit(rgb, (WIDTH, HEIGHT), method=Image.Resampling.BICUBIC)
            return np.asarray(cropped)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Choose a readable JPEG, PNG or WebP image") from exc
