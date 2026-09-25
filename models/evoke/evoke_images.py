"""Validate uploaded EVOKE media and select native loader suffixes."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 250 * 1024 * 1024
MAX_POSE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
_IMAGE_MIME_TYPES = {"image/bmp", "image/jpeg", "image/png", "image/webp"}
_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime", "video/webm"}


def validate_uploaded_image(image: UploadedFile) -> None:
    """Validate an image upload without retaining decoded pixels."""
    if image.mime_type.lower() not in _IMAGE_MIME_TYPES:
        raise CommandError(
            "unsupported_image_type", "Upload JPEG, PNG, WebP, or BMP image bytes."
        )
    if image.size <= 0 or image.size > MAX_IMAGE_BYTES:
        raise CommandError(
            "invalid_image_size", "Image uploads must be between 1 byte and 25 MiB."
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            decoded.verify()
        with Image.open(io.BytesIO(image.data)) as decoded:
            oriented = ImageOps.exif_transpose(decoded)
            if oriented.width * oriented.height > MAX_IMAGE_PIXELS:
                raise CommandError(
                    "image_too_large",
                    "Decoded images may contain at most 100 million pixels.",
                )
    except CommandError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise CommandError(
            "invalid_image", "The uploaded image cannot be decoded."
        ) from error


def validate_uploaded_video(video: UploadedFile) -> None:
    """Validate a reference-video upload before the worker decodes it."""
    if video.mime_type.lower() not in _VIDEO_MIME_TYPES:
        raise CommandError(
            "unsupported_video_type", "Upload MP4, MOV, or WebM reference video bytes."
        )
    if video.size <= 0 or video.size > MAX_VIDEO_BYTES:
        raise CommandError(
            "invalid_video_size", "Reference videos must be between 1 byte and 250 MiB."
        )


def validate_uploaded_pose(pose: UploadedFile) -> None:
    """Validate a camera-pose NPZ upload and its required arrays."""
    if pose.size <= 0 or pose.size > MAX_POSE_BYTES:
        raise CommandError(
            "invalid_pose_size", "Pose tracks must be between 1 byte and 64 MiB."
        )
    try:
        with np.load(io.BytesIO(pose.data), allow_pickle=False) as archive:
            camera_key = next(
                (key for key in ("cam_c2w", "extrinsic", "data") if key in archive),
                None,
            )
            intrinsic_key = next(
                (key for key in ("intrinsics", "intrinsic", "K") if key in archive),
                None,
            )
            if camera_key is None or intrinsic_key is None:
                raise ValueError("missing pose arrays")
            cameras = np.asarray(archive[camera_key])
            intrinsics = np.asarray(archive[intrinsic_key])
    except (OSError, ValueError, KeyError) as error:
        raise CommandError(
            "invalid_pose",
            "Pose NPZ must contain cam_c2w/extrinsic/data and intrinsics/intrinsic/K arrays.",
        ) from error
    if cameras.ndim != 3 or cameras.shape[1:] != (4, 4) or len(cameras) < 2:
        raise CommandError(
            "invalid_pose", "Camera poses must have shape (N, 4, 4) with N at least 2."
        )
    if intrinsics.shape not in {(3, 3), (4,)} and not (
        intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3)
    ):
        raise CommandError(
            "invalid_pose", "Camera intrinsics must contain a 3x3 matrix."
        )
    if not np.isfinite(cameras).all() or not np.isfinite(intrinsics).all():
        raise CommandError(
            "invalid_pose", "Pose arrays must contain only finite values."
        )


def upload_suffix(upload: UploadedFile) -> str:
    """Preserve the native loader's media suffix without passing an upload object."""
    suffixes = {
        "image/bmp": ".bmp",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "application/x-npz": ".npz",
        "application/octet-stream": ".npz",
    }
    return suffixes.get(
        upload.mime_type.lower(), Path(upload.name).suffix.lower() or ".bin"
    )
