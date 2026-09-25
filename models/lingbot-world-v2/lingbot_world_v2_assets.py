"""Prepare pinned public source and checkpoint assets for LingBot-World-V2."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SOURCE_PATH_ENV = "LINGBOT_WORLD_V2_PATH"
CHECKPOINT_PATH_ENV = "LINGBOT_WORLD_V2_CHECKPOINT_PATH"
_REVISION = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_FILES = (
    "Wan2.1_VAE.pth",
    "config.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "transformers/diffusion_pytorch_model.safetensors.index.json",
    *(f"transformers/model-{index:05d}-of-00008.safetensors" for index in range(1, 9)),
)


@dataclass(frozen=True)
class BuiltInScene:
    """Describe one public image, prompt, and camera calibration."""

    name: str
    image: Path
    prompt: str
    intrinsics: Path
    initial_poses: Path


@dataclass(frozen=True)
class LingBotConfig:
    """Hold validated paths and native causal-fast inference settings."""

    source_path: Path
    source_url: str
    source_revision: str
    checkpoint_path: Path
    checkpoint_repo_id: str
    checkpoint_revision: str
    scenes: tuple[BuiltInScene, ...]
    upload_default_prompt: str
    seed: int
    max_area: int
    chunk_latents: int
    timesteps: tuple[int, ...]
    shift: float
    local_attention_frames: int
    attention_sink_frames: int
    max_chunks: int
    rotation_degrees_per_second: float
    upload_intrinsics: tuple[float, float, float, float]


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> LingBotConfig:
    """Read and validate the LingBot adapter YAML.

    Args:
        config_path: Path supplied by ``runtime.config`` in ``reactor.yaml``.
        weights_root: Root for source and checkpoint paths. Defaults to the
            configuration file's directory for standalone use.

    Returns:
        Validated public source, asset, inference, and interaction settings.

    Raises:
        ValueError: If the file is missing or contains an invalid setting.
    """
    if config_path is None:
        raise ValueError("LingBot-World-V2 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    weights_path = (
        weights_root if weights_root is not None else config_path.resolve().parent
    )
    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    checkpoint = _mapping(assets.get("checkpoint"), "assets.checkpoint")
    inputs = _mapping(document.get("inputs"), "inputs")
    inference = _mapping(document.get("inference"), "inference")
    motion = _mapping(document.get("motion"), "motion")
    stream = _mapping(document.get("stream"), "stream")

    source_path = _configured_path(
        os.environ.get(SOURCE_PATH_ENV), weights_path, source.get("path"), "source.path"
    )
    checkpoint_path = _configured_path(
        os.environ.get(CHECKPOINT_PATH_ENV),
        weights_path,
        checkpoint.get("path"),
        "assets.checkpoint.path",
    )
    source_revision = _revision(source.get("revision"), "source.revision")
    checkpoint_revision = _revision(
        checkpoint.get("revision"), "assets.checkpoint.revision"
    )

    chunk_latents = int(inference.get("chunk_latents", 4))
    if chunk_latents != 4:
        raise ValueError(
            "inference.chunk_latents must remain 4 for the public fast checkpoint"
        )
    timesteps = tuple(
        int(value) for value in _sequence(inference.get("timesteps"), "timesteps")
    )
    if not timesteps or any(value < 0 or value >= 1000 for value in timesteps):
        raise ValueError("inference.timesteps must contain indices from 0 through 999")
    local_attention_frames = int(inference.get("local_attention_frames", 18))
    attention_sink_frames = int(inference.get("attention_sink_frames", 6))
    if local_attention_frames != 18 or attention_sink_frames != 6:
        raise ValueError(
            "LingBot's released fast checkpoint uses an 18-frame local window and 6-frame sink"
        )
    max_chunks = int(stream.get("max_chunks", 256))
    if not 1 <= max_chunks <= 256:
        raise ValueError(
            "stream.max_chunks must be between 1 and the native limit of 256"
        )
    max_area = int(inference.get("max_area", 480 * 832))
    if max_area <= 0:
        raise ValueError("inference.max_area must be positive")
    shift = float(inference.get("shift", 10.0))
    if shift <= 0:
        raise ValueError("inference.shift must be positive")
    rotation_speed = float(motion.get("rotation_degrees_per_second", 45.0))
    if rotation_speed <= 0:
        raise ValueError("motion.rotation_degrees_per_second must be positive")
    intrinsics_values = tuple(
        float(value)
        for value in _sequence(
            inputs.get("upload_intrinsics"), "inputs.upload_intrinsics"
        )
    )
    if (
        len(intrinsics_values) != 4
        or intrinsics_values[0] <= 0
        or intrinsics_values[1] <= 0
    ):
        raise ValueError(
            "inputs.upload_intrinsics must be [fx, fy, cx, cy] with positive focal lengths"
        )

    scenes = tuple(
        _scene(source_path, value, index)
        for index, value in enumerate(
            _sequence(inputs.get("random_images"), "inputs.random_images")
        )
    )
    if not scenes:
        raise ValueError(
            "inputs.random_images must contain at least one built-in scene"
        )
    upload_prompt = str(inputs.get("upload_default_prompt", "")).strip()
    if not upload_prompt:
        raise ValueError("inputs.upload_default_prompt must not be empty")

    return LingBotConfig(
        source_path=source_path,
        source_url=_url(source.get("url"), "source.url"),
        source_revision=source_revision,
        checkpoint_path=checkpoint_path,
        checkpoint_repo_id=_repo_id(checkpoint.get("repo_id")),
        checkpoint_revision=checkpoint_revision,
        scenes=scenes,
        upload_default_prompt=upload_prompt,
        seed=int(inference.get("seed", 42)),
        max_area=max_area,
        chunk_latents=chunk_latents,
        timesteps=timesteps,
        shift=shift,
        local_attention_frames=local_attention_frames,
        attention_sink_frames=attention_sink_frames,
        max_chunks=max_chunks,
        rotation_degrees_per_second=rotation_speed,
        upload_intrinsics=intrinsics_values,
    )


def prepare_runtime_assets(config: LingBotConfig, weights_root: Path) -> None:
    """Clone the pinned public source and download the pinned public checkpoint."""
    _ensure_source(config)
    _ensure_checkpoint(config, weights_root)
    _validate_scenes(config.scenes)


def _ensure_source(config: LingBotConfig) -> None:
    """Create or verify an unmodified checkout at the configured revision."""
    path = config.source_path
    if not path.exists():
        logger.info(
            "downloading LingBot-World-V2 source",
            extra={
                "url": config.source_url,
                "revision": config.source_revision,
                "destination": str(path),
            },
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".lingbot-source-", dir=path.parent
        ) as temporary:
            checkout = Path(temporary) / "checkout"
            _run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    config.source_url,
                    str(checkout),
                ]
            )
            _run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "checkout",
                    "--detach",
                    config.source_revision,
                ]
            )
            checkout.rename(path)
    if not (path / ".git").is_dir():
        raise RuntimeError(f"LingBot source at {path} must be a Git checkout")
    actual = _git(path, "rev-parse", "HEAD").stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"LingBot source revision is {actual}; expected {config.source_revision}"
        )
    dirty = _git(path, "status", "--porcelain").stdout.strip()
    if dirty:
        raise RuntimeError(f"LingBot source at {path} has local modifications")


def _ensure_checkpoint(config: LingBotConfig, weights_root: Path) -> None:
    """Download missing checkpoint files and provide the subfolder config upstream expects."""
    missing = [
        relative
        for relative in _CHECKPOINT_FILES
        if not (config.checkpoint_path / relative).is_file()
    ]
    if missing:
        from huggingface_hub import snapshot_download

        logger.info(
            "downloading LingBot-World-V2 checkpoint",
            extra={
                "repo_id": config.checkpoint_repo_id,
                "revision": config.checkpoint_revision,
                "destination": str(config.checkpoint_path),
            },
        )
        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir = weights_root / ".huggingface"
        snapshot_download(
            repo_id=config.checkpoint_repo_id,
            revision=config.checkpoint_revision,
            local_dir=config.checkpoint_path,
            cache_dir=cache_dir,
        )
    still_missing = [
        relative
        for relative in _CHECKPOINT_FILES
        if not (config.checkpoint_path / relative).is_file()
    ]
    if still_missing:
        raise FileNotFoundError(
            "LingBot checkpoint is incomplete: " + ", ".join(still_missing[:4])
        )

    transformer_config = config.checkpoint_path / "transformers" / "config.json"
    if not transformer_config.is_file():
        transformer_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.checkpoint_path / "config.json", transformer_config)
    marker = config.checkpoint_path / ".reactor-checkpoint.json"
    marker.write_text(
        json.dumps(
            {
                "repo_id": config.checkpoint_repo_id,
                "revision": config.checkpoint_revision,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_scenes(scenes: tuple[BuiltInScene, ...]) -> None:
    """Verify every built-in image and camera calibration after source checkout."""
    for scene in scenes:
        for label, path in (
            ("image", scene.image),
            ("intrinsics", scene.intrinsics),
            ("poses", scene.initial_poses),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"built-in scene {scene.name} {label} is missing: {path}"
                )


def _scene(source_path: Path, value: object, index: int) -> BuiltInScene:
    """Return one configured built-in scene."""
    document = _mapping(value, f"inputs.random_images[{index}]")
    name = str(document.get("name", "")).strip()
    prompt = str(document.get("prompt", "")).strip()
    if not name or not prompt:
        raise ValueError(
            f"inputs.random_images[{index}] requires non-empty name and prompt"
        )
    action_path = _relative_path(
        source_path, document.get("action_path"), "action_path"
    )
    return BuiltInScene(
        name=name,
        image=_relative_path(source_path, document.get("image"), "image"),
        prompt=prompt,
        intrinsics=action_path / "intrinsics.npy",
        initial_poses=action_path / "poses.npy",
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    """Return a YAML mapping or raise a field-specific error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    """Return a non-string YAML sequence or raise a field-specific error."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _configured_path(
    override: str | None, root: Path, value: object, name: str
) -> Path:
    """Resolve an optional environment override or a weights-relative path."""
    raw = override.strip() if override else str(value or "").strip()
    if not raw:
        raise ValueError(f"{name} must not be empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    return path if path.is_absolute() else root / path


def _relative_path(root: Path, value: object, name: str) -> Path:
    """Resolve one source-relative path while rejecting an empty value."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{name} must not be empty")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _revision(value: object, name: str) -> str:
    """Return a pinned 40-character Git or Hub revision."""
    revision = str(value or "").strip().lower()
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _url(value: object, name: str) -> str:
    """Return a public HTTPS Git URL."""
    url = str(value or "").strip()
    if not url.startswith("https://") or not url.endswith(".git"):
        raise ValueError(f"{name} must be a public HTTPS Git URL ending in .git")
    return url


def _repo_id(value: object) -> str:
    """Return a Hugging Face repository identifier."""
    repo_id = str(value or "").strip()
    if repo_id.count("/") != 1:
        raise ValueError("assets.checkpoint.repo_id must have owner/name form")
    return repo_id


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a source-management command and return captured output."""
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _git(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only Git command for one explicitly trusted checkout."""
    return _run(["git", "-c", f"safe.directory={path}", "-C", str(path), *arguments])
