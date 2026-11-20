"""Validate uploads without expanding their encoded representation."""
import io
import warnings

from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_PIXELS = 20_000_000
UPLOAD_FORMATS = ("PNG", "JPEG", "WEBP")
UNDECODABLE = (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError,
               Image.DecompressionBombWarning)


def validate_image(data: bytes) -> bytes:
    """Return the upload's original bytes once they decode as an acceptable image.

    Raises:
        ValueError: The upload is empty, too large, not PNG/JPEG/WebP, smaller
            than 64 pixels per side, larger than 20 megapixels, or does not
            decode.
    """
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload size is outside 1 byte to 20 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                kind, (width, height) = source.format, source.size
                acceptable = (kind in UPLOAD_FORMATS and min(width, height) >= 64
                              and width * height <= MAX_UPLOAD_PIXELS)
                if acceptable:
                    source.load()
    except UNDECODABLE as exc:
        raise ValueError(f"Upload does not decode as an image: {exc}") from exc
    if kind not in UPLOAD_FORMATS:
        raise ValueError(f"Unsupported format {kind}")
    if not acceptable:
        raise ValueError(f"Unsupported dimensions {width}x{height}")
    return data
