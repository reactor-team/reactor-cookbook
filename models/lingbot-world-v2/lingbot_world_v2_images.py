"""Validate uploaded anchors and load public example camera calibration."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError
from reactor_runtime import UploadedFile

from lingbot_world_v2_assets import BuiltInScene

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
_MIME_TYPES = {"image/bmp", "image/jpeg", "image/png", "image/webp"}


def validate_uploaded_image(image: UploadedFile) -> None:
    """Validate upload type, size, dimensions, and decodability.

    Args:
        image: File supplied through Reactor's upload protocol.

    Raises:
        ValueError: If the upload is empty, too large, mislabeled, or not a
            decodable image within the pixel limit.
    """
    if not image.data:
        raise ValueError("uploaded image is empty")
    if image.size > MAX_UPLOAD_BYTES:
        raise ValueError("uploaded image exceeds 25 MiB")
    if image.mime_type.lower() not in _MIME_TYPES:
        raise ValueError("uploaded image must be JPEG, PNG, WebP, or BMP")
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("uploaded image exceeds the 100-million-pixel limit")
            decoded.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("uploaded image bytes cannot be decoded") from error


def load_scene_camera(scene: BuiltInScene) -> tuple[np.ndarray, np.ndarray]:
    """Return the first pose and packed intrinsics from a public example."""
    poses = np.load(scene.initial_poses, allow_pickle=False)
    intrinsics = np.load(scene.intrinsics, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(
            f"{scene.initial_poses} must contain camera poses shaped (N, 4, 4)"
        )
    if not np.isfinite(poses[0]).all():
        raise ValueError(f"{scene.initial_poses} contains a non-finite initial pose")
    if intrinsics.ndim == 2 and intrinsics.shape[1:] == (4,) and len(intrinsics) > 0:
        intrinsics = intrinsics[0]
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise ValueError(
            f"{scene.intrinsics} must contain finite intrinsics shaped (N, 4)"
        )
    if bool(np.any(intrinsics[:2] <= 0)):
        raise ValueError(f"{scene.intrinsics} must contain positive focal lengths")
    return (
        np.ascontiguousarray(poses[0], dtype=np.float32),
        np.ascontiguousarray(intrinsics, dtype=np.float32),
    )
