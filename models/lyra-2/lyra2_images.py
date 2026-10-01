"""Prepare Lyra's native 768x448 seed pixels on the CPU."""

import io
import numpy as np
from PIL import Image
from reactor_runtime import CommandError, UploadedFile


def prepare_image(upload: UploadedFile) -> np.ndarray:
    if (
        not upload.mime_type.startswith("image/")
        or not upload.data
        or upload.size > 25 * 1024**2
    ):
        raise CommandError(
            "invalid_image", "Upload a nonempty image no larger than 25 MiB"
        )
    try:
        with Image.open(io.BytesIO(upload.data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise ValueError("Unsupported image format")
            if image.width * image.height > 100_000_000:
                raise ValueError("Image exceeds 100 million pixels")
            import cv2

            return cv2.resize(
                np.asarray(image.convert("RGB")),
                (768, 448),
                interpolation=cv2.INTER_AREA,
            )
    except (Image.DecompressionBombError, OSError, ValueError) as error:
        raise CommandError("invalid_image", str(error)) from error
