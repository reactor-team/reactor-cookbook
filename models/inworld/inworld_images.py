"""Bounded upload decoding, called off the application's event loop."""

import io
from fractions import Fraction
from itertools import pairwise

import av
import numpy as np
from PIL import Image, ImageOps

from inworld_model import FPS, PREFIX_FRAMES


class InvalidUpload(ValueError):
    """An upload does not meet the model's public media requirements."""


def decode_reference(data: bytes) -> np.ndarray:
    if not data or len(data) > 25 * 1024 * 1024:
        raise InvalidUpload("Reference images must be non-empty and at most 25 MiB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 20_000_000:
                raise InvalidUpload("Reference image exceeds 20 million pixels")
            return np.array(
                ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8
            )
    except InvalidUpload:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidUpload("The file is not a decodable image.") from exc


def _full_frame_count(data: bytes) -> int:
    """Fallback for codecs or containers without one timed display frame per packet."""
    with av.open(io.BytesIO(data)) as container:
        count = 0
        for _ in container.decode(video=0):
            count += 1
            if count > 60 * FPS:
                break
        return count


def validate_video(data: bytes) -> None:
    if not data or len(data) > 256 * 1024 * 1024:
        raise InvalidUpload("Parent video must be non-empty and at most 256 MiB")
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.video:
                raise InvalidUpload("Upload contains no video")
            stream = container.streams.video[0]
            if stream.average_rate != FPS:
                raise InvalidUpload(
                    "Parent video must be 24 FPS; resample it before upload"
                )
            if stream.width * stream.height > 4096 * 2160:
                raise InvalidUpload("Parent video exceeds 4096 x 2160 pixels")
            # H.264/HEVC MP4 packets normally hold one display frame. Require
            # complete, unique 24 FPS timestamps and durations before using that
            # count; other layouts retain the bounded decoding fallback.
            timed = stream.codec_context.name in {"h264", "hevc"}
            timestamps = []
            for packet in container.demux(stream):
                if not packet.size:
                    continue
                if packet.pts is None or packet.duration * stream.time_base != Fraction(
                    1, FPS
                ):
                    timed = False
                else:
                    timestamps.append(packet.pts)
            timestamps.sort()
            count = len(timestamps)
            timed = (
                timed
                and count >= PREFIX_FRAMES
                and all(
                    (b - a) * stream.time_base == Fraction(1, FPS)
                    for a, b in pairwise(timestamps)
                )
            )
            if timed and count <= 60 * FPS:
                # Seeking lands on the preceding keyframe. Decode the dependent
                # frames too, then verify the exact tail used for continuation.
                tail_start = timestamps[-PREFIX_FRAMES]
                try:
                    container.seek(tail_start, stream=stream, backward=True)
                    tail_count = sum(
                        1
                        for frame in container.decode(video=0)
                        if frame.pts is not None and frame.pts >= tail_start
                    )
                    timed = tail_count == PREFIX_FRAMES
                except av.FFmpegError:
                    timed = False
        if not timed:
            count = _full_frame_count(data)
        if count > 60 * FPS:
            raise InvalidUpload("Parent video exceeds 60 seconds")
        if count < PREFIX_FRAMES:
            raise InvalidUpload("Parent video must contain at least 17 frames")
    except (av.FFmpegError, OSError) as exc:
        raise InvalidUpload("The file is not a decodable video.") from exc
