"""Bounded CPU decoding of uploaded starting frames."""

import io

import av
import numpy as np
from PIL import Image, ImageOps


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 100_000_000:
                raise ValueError("image exceeds 100 million pixels")
            fitted = (
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize((640, 360), Image.Resampling.BILINEAR)
            )
            return np.asarray(fitted, dtype=np.uint8)[None]
    except Image.DecompressionBombError as error:
        raise ValueError("image exceeds the safe pixel limit") from error


def decode_video(data: bytes, offset: int, count: int) -> np.ndarray:
    """Decode only through the last requested frame, retaining at most count frames."""
    if offset < 0 or not 1 <= count <= 32:
        raise ValueError("offset must be non-negative and count must be 1 through 32")
    frames = []
    try:
        with av.open(io.BytesIO(data)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= offset:
                    frames.append(_resize_video_frame(frame.to_ndarray(format="rgb24")))
                if len(frames) == count:
                    break
    except (av.FFmpegError, IndexError) as error:
        raise ValueError("video cannot be decoded") from error
    if len(frames) != count:
        raise ValueError(f"video contains fewer than {offset + count} frames")
    return np.ascontiguousarray(np.stack(frames))


def _resize_video_frame(frame: np.ndarray) -> np.ndarray:
    """Bilinear half-pixel coordinates, matching align_corners=False without torch."""
    height, width = frame.shape[:2]
    if (height, width) == (360, 640):
        return frame
    y = np.maximum(
        (np.arange(360, dtype=np.float32) + 0.5) * np.float32(height / 360) - 0.5, 0
    )
    x = np.maximum(
        (np.arange(640, dtype=np.float32) + 0.5) * np.float32(width / 640) - 0.5, 0
    )
    y0, x0 = y.astype(np.intp), x.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    wy, wx = (
        (y - y0).astype(np.float32)[:, None, None],
        (x - x0).astype(np.float32)[None, :, None],
    )
    top = (1 - wx) * frame[y0[:, None], x0] + wx * frame[y0[:, None], x1]
    bottom = (1 - wx) * frame[y1[:, None], x0] + wx * frame[y1[:, None], x1]
    return np.clip((1 - wy) * top + wy * bottom, 0, 255).astype(np.uint8)
