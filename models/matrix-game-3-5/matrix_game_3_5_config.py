"""Read the Matrix-Game-3.5 config and check its source, worker, and weights."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
WORKER_PYTHON = Path(".venv/bin/python")
INFERENCE_CONFIG = Path("configs/infer_distilled.yaml")
CHECKPOINT_PATH = Path("checkpoints/Matrix-Game-3.5-Distilled/first-person.safetensors")
WAN_PATH = Path("checkpoints/Wan2.2-TI2V-5B")
TOKENIZER_PATH = WAN_PATH / "google/umt5-xxl"
DEPTH_PATH = Path("checkpoints/DA3NESTED-GIANT-LARGE-1.1")
DEFAULT_SAMPLE = Path("samples/first_person/case_0")
ANCHOR_IMAGE = DEFAULT_SAMPLE / "input.png"
CAMERA = DEFAULT_SAMPLE / "camera.npz"

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
WAN_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
DEPTH_REQUIRED_FILES = ("config.json", "model.safetensors")


@dataclass(frozen=True)
class MatrixAsset:
    """Describe one pinned public model asset and its local location."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class MatrixConfig:
    """Hold validated Matrix adapter settings."""

    worker_python: Path
    source_path: Path
    source_url: str
    source_revision: str
    inference_config: Path
    checkpoint: MatrixAsset
    wan: MatrixAsset
    tokenizer_dir: Path
    depth: MatrixAsset
    anchor_image: Path
    camera: Path
    default_prompt: str
    seed: int
    max_chunks: int
    translation_meters_per_second: float
    rotation_degrees_per_second: float


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> MatrixConfig:
    """Read and validate the Matrix adapter YAML.

    ``weights_root`` is the folder that ``prepare_weights.py`` builds. It
    defaults to ``weights/`` beside the config.
    """
    if config_path is None:
        raise ValueError("Matrix-Game-3.5 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    source_revision = _revision(source.get("revision"), "source.revision")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    motion = _mapping(document.get("motion"), "motion")
    stream = _mapping(document.get("stream"), "stream")
    source_path = _source_path(source["path"], config_path.resolve().parent)
    root = (
        weights_root
        if weights_root is not None
        else config_path.resolve().parent / "weights"
    ).resolve()
    checkpoint = _asset(
        root,
        assets.get("checkpoint"),
        "assets.checkpoint",
        CHECKPOINT_PATH,
    )
    wan = _asset(root, assets.get("wan"), "assets.wan", WAN_PATH)
    depth = _asset(root, assets.get("depth"), "assets.depth", DEPTH_PATH)
    translation_speed = float(motion.get("translation_meters_per_second", 1.5))
    rotation_speed = float(motion.get("rotation_degrees_per_second", 45.0))
    if translation_speed <= 0:
        raise ValueError("motion.translation_meters_per_second must be positive")
    if rotation_speed <= 0:
        raise ValueError("motion.rotation_degrees_per_second must be positive")
    max_chunks = int(stream.get("max_chunks", 512))
    if max_chunks < 8:
        raise ValueError("stream.max_chunks must be at least 8")
    default_prompt = str(inference.get("default_prompt", "")).strip()
    if not default_prompt:
        raise ValueError("inference.default_prompt must be non-empty")

    return MatrixConfig(
        worker_python=source_path / WORKER_PYTHON,
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=source_revision,
        inference_config=source_path / INFERENCE_CONFIG,
        checkpoint=checkpoint,
        wan=wan,
        tokenizer_dir=root / TOKENIZER_PATH,
        depth=depth,
        anchor_image=source_path / ANCHOR_IMAGE,
        camera=source_path / CAMERA,
        default_prompt=default_prompt,
        seed=int(inference.get("seed", 3407)),
        max_chunks=max_chunks,
        translation_meters_per_second=translation_speed,
        rotation_degrees_per_second=rotation_speed,
    )


def check_runtime(config: MatrixConfig) -> tuple[np.ndarray, np.ndarray]:
    """Check the source, worker, and weights, then return the default camera pose.

    This function never downloads. The image holds the patched source and the
    worker environment, and ``prepare_weights.py`` builds the weights folder.
    """
    check_source(config)
    check_weights(config)
    _validate_runtime_paths(config)
    return _load_initial_camera(config.camera)


def check_source(config: MatrixConfig) -> None:
    """Require the pinned source checkout with the stateful patch applied."""
    source_path = config.source_path
    if not (source_path / ".git").exists():
        raise FileNotFoundError(
            f"Matrix source is missing at {source_path}; rebuild the image with `reactor build`"
        )
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"Matrix source revision is {actual}; expected {config.source_revision}"
        )
    patch = Path(__file__).with_name("stateful_rollout.patch")
    reverse = _check_git(
        ["-C", str(source_path), "apply", "--reverse", "--check", str(patch)]
    )
    if reverse.returncode != 0:
        raise RuntimeError(
            "Matrix source lacks the stateful rollout patch; rebuild the image with `reactor build`"
        )


def check_weights(config: MatrixConfig) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    required = [config.checkpoint.path]
    required += [config.wan.path / name for name in WAN_REQUIRED_FILES]
    required += [config.depth.path / name for name in DEPTH_REQUIRED_FILES]
    missing = [str(path) for path in required if not _is_nonempty_file(path)]
    if missing:
        raise FileNotFoundError(
            f"Matrix weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _asset(
    source_path: Path, value: object, name: str, relative_path: Path
) -> MatrixAsset:
    """Return one pinned public asset under the Matrix source root."""
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a public repository")
    return MatrixAsset(
        path=source_path / relative_path,
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _revision(value: object, name: str) -> str:
    """Return one full immutable Git-style revision."""
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    """Return an HTTPS URL for a public source repository."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _source_path(value: object, base: Path) -> Path:
    """Resolve the Matrix checkout path. The image build places it there."""
    path = Path(str(value)).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git and report its error."""
    try:
        return subprocess.run(
            _git_command(arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to check the Matrix source") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to check the Matrix source: {detail}") from error


def _check_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a non-mutating Git check without raising for an unsuccessful check."""
    try:
        return subprocess.run(
            _git_command(arguments),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to check the Matrix source") from error


def _git_command(arguments: list[str]) -> list[str]:
    """Trust only the configured checkout when a bind mount changes its owner."""
    if len(arguments) >= 2 and arguments[0] == "-C":
        return ["git", "-c", f"safe.directory={arguments[1]}", *arguments]
    return ["git", *arguments]


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a model file exists and contains data."""
    return path.is_file() and path.stat().st_size > 0


def _validate_runtime_paths(config: MatrixConfig) -> None:
    """Require every source file, input, environment, and model asset."""
    files = {
        "Matrix worker Python": config.worker_python,
        "inference config": config.inference_config,
        "distilled checkpoint": config.checkpoint.path,
        "anchor image": config.anchor_image,
        "camera trajectory": config.camera,
    }
    directories = {
        "Matrix source": config.source_path,
        "Wan2.2 model": config.wan.path,
        "tokenizer": config.tokenizer_dir,
        "Depth-Anything-3 model": config.depth.path,
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    for label, path in directories.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")


def _load_initial_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the first c2w pose and intrinsics from a public Matrix camera NPZ."""
    with np.load(path) as archive:
        if "extrinsics_c2w" not in archive or "intrinsics" not in archive:
            raise ValueError(f"{path}: expected extrinsics_c2w and intrinsics arrays")
        extrinsics = np.asarray(archive["extrinsics_c2w"], dtype=np.float32)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"{path}: extrinsics_c2w must have shape (N, 4, 4)")
    if int(extrinsics.shape[0]) == 0:
        raise ValueError(f"{path}: camera trajectory is empty")
    return np.ascontiguousarray(extrinsics[0]), intrinsics
