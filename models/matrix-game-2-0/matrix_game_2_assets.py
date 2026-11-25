"""Read the Matrix-Game-2.0 config and check its source, weights, and image inputs."""

from __future__ import annotations

import io
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from matrix_game_2_types import MatrixGame2Config
from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile, get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

SOURCE_ENV = "MATRIX_GAME_2_PATH"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_IMAGE_FORMATS = {"BMP", "JPEG", "PNG", "WEBP"}
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
# The files of the pinned repository that the model reads. The repository
# also holds the base and other distilled checkpoints, which this model skips.
MODEL_ALLOW_PATTERNS = (
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "xlm-roberta-large/*",
    "base_distilled_model/base_distill.safetensors",
)


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> MatrixGame2Config:
    """Read and validate the Matrix-Game-2.0 adapter YAML.

    ``weights_root`` is the folder that ``prepare_weights.py`` builds. It
    defaults to Runtime's weights path.
    """
    if config_path is None:
        raise ValueError("Matrix-Game-2.0 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    model = _mapping(document.get("model"), "model")
    inference = _mapping(document.get("inference"), "inference")
    inputs = _mapping(document.get("inputs"), "inputs")

    configured_source = os.environ.get(SOURCE_ENV)
    if configured_source:
        source_path = Path(configured_source).expanduser().resolve()
    else:
        source_path = (
            Path(str(source.get("path", "/opt/Matrix-Game"))) / "Matrix-Game-2"
        )

    max_latent_frames = int(inference.get("max_latent_frames", 360))
    if max_latent_frames != 360:
        raise ValueError(
            "inference.max_latent_frames must remain 360 to match the official rollout"
        )

    random_values = inputs.get("random_images")
    if not isinstance(random_values, list) or not random_values:
        raise ValueError("inputs.random_images must be a non-empty list")
    random_images = tuple(
        source_path / _relative_path(value, "inputs.random_images")
        for value in random_values
    )

    repo_id = str(model.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError("model.repo_id must identify a public Hugging Face repository")
    checkpoint_file = str(
        model.get(
            "checkpoint_file",
            "base_distilled_model/base_distill.safetensors",
        )
    )
    if checkpoint_file != "base_distilled_model/base_distill.safetensors":
        raise ValueError(
            "model.checkpoint_file must select the universal distilled checkpoint"
        )

    return MatrixGame2Config(
        source_path=source_path,
        source_url=_public_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        model_repo_id=repo_id,
        model_revision=_revision(model.get("revision"), "model.revision"),
        weights_root=weights_root if weights_root is not None else get_weights_path(),
        checkpoint_file=checkpoint_file,
        seed=int(inference.get("seed", 0)),
        max_latent_frames=max_latent_frames,
        random_images=random_images,
    )


def check_runtime_assets(config: MatrixGame2Config) -> Path:
    """Check the source in the image and the weights folder. Return the weights root.

    This function never downloads. The image holds the source at the pinned
    revision, and ``prepare_weights.py`` builds the weights folder.
    """
    check_source(config)
    for image in config.random_images:
        if not image.is_file():
            raise FileNotFoundError(
                f"configured Matrix example image does not exist: {image}"
            )
    check_checkpoint(config)
    return config.weights_root


def check_source(config: MatrixGame2Config) -> None:
    """Check that the source directory exists at the pinned revision."""
    if not config.source_path.is_dir():
        raise FileNotFoundError(
            f"Matrix-Game-2 source directory does not exist: {config.source_path}; "
            "rebuild the image with `reactor build`"
        )
    _validate_source_files(config)
    actual = _run_git(
        ["-C", str(config.source_path), "rev-parse", "HEAD"]
    ).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"Matrix source revision is {actual}; expected {config.source_revision}"
        )


def check_checkpoint(config: MatrixGame2Config) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    root = config.weights_root
    required = (
        config.checkpoint_file,
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "xlm-roberta-large/tokenizer.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Matrix-Game-2.0 weights in {root} lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    if not image.mime_type.startswith("image/"):
        raise CommandError("unsupported_media", f"{image.name} is not an image.")
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            if decoded.format not in _IMAGE_FORMATS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.verify()
    except CommandError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error


def _validate_source_files(config: MatrixGame2Config) -> None:
    """Raise a precise error when a required upstream inference file is absent."""
    required = (
        "configs/inference_yaml/inference_universal.yaml",
        "configs/distilled_model/universal/config.json",
        "pipeline/causal_inference.py",
        "demo_utils/vae_block3.py",
        "wan/vae/wanx_vae.py",
    )
    missing = [
        relative
        for relative in required
        if not (config.source_path / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Matrix source is missing required inference files: " + ", ".join(missing)
        )


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], dict(value))


def _relative_path(value: object, name: str) -> Path:
    """Return a safe path relative to the upstream source root."""
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{name} entries must be relative paths inside the source checkout"
        )
    return path


def _revision(value: object, name: str) -> str:
    """Return one full immutable Git-style revision."""
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _public_url(value: object, name: str) -> str:
    """Return one public HTTPS repository URL."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git without an interactive credential prompt."""
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
