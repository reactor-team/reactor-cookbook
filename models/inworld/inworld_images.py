"""Bounded upload decoding, called off the application's event loop."""

import io

import av
import numpy as np
from PIL import Image, ImageOps


def decode_reference(data: bytes) -> np.ndarray:
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("Reference images must be non-empty and at most 25 MiB")
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Reference image exceeds 20 million pixels")
        # Preserve pixels here: native reference fitting depends on the ordered
        # reference list and is performed by the pinned upstream image stage.
        return np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def validate_video(data: bytes) -> None:
    if not data or len(data) > 256 * 1024 * 1024:
        raise ValueError("Parent video must be non-empty and at most 256 MiB")
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise ValueError("Upload contains no video")
        stream = container.streams.video[0]
        if stream.average_rate != 24:
            raise ValueError("Parent video must be 24 FPS; resample it before upload")
        if stream.width * stream.height > 4096 * 2160:
            raise ValueError("Parent video exceeds 4096 x 2160 pixels")
        count = 0
        for _ in container.decode(video=0):
            count += 1
            if count > 1440:
                raise ValueError("Parent video exceeds 60 seconds")
        if count < 17:
            raise ValueError("Parent video must contain at least 17 frames")
