"""Prepare public Matrix-Game-2.0 source, weights, and image inputs."""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
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
_MODEL_ALLOW_PATTERNS = (
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "xlm-roberta-large/*",
    "base_distilled_model/base_distill.safetensors",
)


def read_config(config_path: Path | None) -> MatrixGame2Config:
    """Read and validate the Matrix-Game-2.0 adapter YAML."""
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
        checkout_path = source_path.parent
    else:
        checkout_value = str(source.get("path", "Matrix-Game"))
        checkout_path = _weights_relative_path(checkout_value)
        source_path = checkout_path / "Matrix-Game-2"

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
        checkout_path=checkout_path,
        source_path=source_path,
        source_url=_public_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        model_repo_id=repo_id,
        model_revision=_revision(model.get("revision"), "model.revision"),
        model_cache=get_weights_path() / "matrix-game-2-0" / "huggingface",
        checkpoint_file=checkpoint_file,
        seed=int(inference.get("seed", 0)),
        max_latent_frames=max_latent_frames,
        random_images=random_images,
    )


def prepare_runtime_assets(config: MatrixGame2Config) -> Path:
    """Prepare the pinned upstream checkout and distilled public checkpoint."""
    ensure_source_checkout(config)
    _validate_source_files(config)
    for image in config.random_images:
        if not image.is_file():
            raise FileNotFoundError(
                f"configured Matrix example image does not exist: {image}"
            )

    from huggingface_hub import snapshot_download

    config.model_cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "preparing Matrix-Game-2.0 checkpoint",
        repo_id=config.model_repo_id,
        revision=config.model_revision,
        cache=str(config.model_cache),
    )
    snapshot = Path(
        snapshot_download(
            repo_id=config.model_repo_id,
            revision=config.model_revision,
            cache_dir=config.model_cache,
            allow_patterns=list(_MODEL_ALLOW_PATTERNS),
        )
    )
    _validate_model_snapshot(config, snapshot)
    return snapshot


def ensure_source_checkout(config: MatrixGame2Config) -> None:
    """Clone the pinned public source when it is absent and verify its revision."""
    if not config.source_path.exists():
        if os.environ.get(SOURCE_ENV):
            raise FileNotFoundError(
                f"{SOURCE_ENV} does not exist: {config.source_path}"
            )
        destination = config.checkout_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "downloading Matrix source checkout",
            url=config.source_url,
            revision=config.source_revision,
            destination=str(destination),
        )
        with tempfile.TemporaryDirectory(
            prefix=".reactor-matrix-game-2-",
            dir=destination.parent,
        ) as temporary:
            checkout = Path(temporary) / "checkout"
            _run_git(
                [
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--sparse",
                    config.source_url,
                    str(checkout),
                ]
            )
            _run_git(["-C", str(checkout), "sparse-checkout", "set", "Matrix-Game-2"])
            _run_git(
                ["-C", str(checkout), "checkout", "--detach", config.source_revision]
            )
            with suppress(FileExistsError):
                checkout.rename(destination)
    if not config.source_path.is_dir():
        raise FileNotFoundError(
            f"Matrix-Game-2 source directory does not exist: {config.source_path}"
        )
    actual = _run_git(
        ["-C", str(config.source_path), "rev-parse", "HEAD"]
    ).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"Matrix source revision is {actual}; expected {config.source_revision}"
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


def _validate_model_snapshot(config: MatrixGame2Config, snapshot: Path) -> None:
    """Raise a precise error when the downloaded inference snapshot is incomplete."""
    required = (
        config.checkpoint_file,
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "xlm-roberta-large/tokenizer.json",
    )
    missing = [relative for relative in required if not (snapshot / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Matrix checkpoint snapshot is missing required files: "
            + ", ".join(missing)
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


def _weights_relative_path(value: str) -> Path:
    """Resolve a configured checkout directory under Runtime's weights root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else get_weights_path() / path


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
