"""Prepare the official script's 832x480 seed pixels on the CPU."""

import io

import numpy as np
from lyra2_backend import HEIGHT, WIDTH
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

        # Match lyra2_custom_traj_inference: OpenCV decode (EXIF-oriented BGR),
        # RGB conversion, then the DA3 helper's direct INTER_LINEAR resize.
        bgr = cv2.imdecode(np.frombuffer(upload.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("OpenCV cannot decode the image")
        return cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
    except (Image.DecompressionBombError, OSError, ValueError) as error:
        raise CommandError("invalid_image", str(error)) from error
