"""Load and prepare public Matrix-Game-3.5 source and model assets."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

SOURCE_ENV = "MATRIX_GAME_3_5_PATH"
WORKER_PYTHON = Path(".venv/bin/python")
INFERENCE_CONFIG = Path("configs/infer_distilled.yaml")
CHECKPOINT_PATH = Path("checkpoints/Matrix-Game-3.5-Distilled/first-person.safetensors")
WAN_PATH = Path("checkpoints/Wan2.2-TI2V-5B")
TOKENIZER_PATH = WAN_PATH / "google/umt5-xxl"
DEPTH_PATH = Path("checkpoints/DA3NESTED-GIANT-LARGE-1.1")
DEFAULT_SAMPLE = Path("samples/first_person/case_0")
ANCHOR_IMAGE = DEFAULT_SAMPLE / "input.png"
CAMERA = DEFAULT_SAMPLE / "camera.npz"
SNAPSHOT_MARKER = ".reactor-snapshot.json"
WORKER_ENV_MARKER = ".reactor-worker-environment.json"
WORKER_ENV_VERSION = 1
WORKER_PYTHON_VERSION = "3.10"
WORKER_TORCH = "torch==2.10.0"
WORKER_TORCHVISION = "torchvision==0.25.0"
WORKER_INDEX_URL = "https://download.pytorch.org/whl/cu128"

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_WAN_REQUIRED_FILES = (
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
_DEPTH_REQUIRED_FILES = ("config.json", "model.safetensors")


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


def read_config(config_path: Path | None) -> MatrixConfig:
    """Read and validate the Matrix adapter YAML."""
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
    source_path = _source_path(source["path"])
    checkpoint = _asset(
        source_path,
        assets.get("checkpoint"),
        "assets.checkpoint",
        CHECKPOINT_PATH,
    )
    wan = _asset(source_path, assets.get("wan"), "assets.wan", WAN_PATH)
    depth = _asset(source_path, assets.get("depth"), "assets.depth", DEPTH_PATH)
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
        tokenizer_dir=source_path / TOKENIZER_PATH,
        depth=depth,
        anchor_image=source_path / ANCHOR_IMAGE,
        camera=source_path / CAMERA,
        default_prompt=default_prompt,
        seed=int(inference.get("seed", 3407)),
        max_chunks=max_chunks,
        translation_meters_per_second=translation_speed,
        rotation_degrees_per_second=rotation_speed,
    )


def prepare_runtime(config: MatrixConfig) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the pinned checkout, worker environment, and model assets."""
    ensure_source_checkout(config)
    ensure_worker_environment(config)
    _validate_bootstrap_paths(config)
    _restore_default_sample(config)
    _ensure_model_assets(config)
    _validate_runtime_paths(config)
    return _load_initial_camera(config.camera)


def ensure_source_checkout(config: MatrixConfig) -> None:
    """Clone the missing Matrix source and apply the adapter's resumable-rollout patch."""
    source_path = config.source_path
    if not source_path.exists():
        logger.info(
            "downloading Matrix source checkout",
            url=config.source_url,
            revision=config.source_revision,
            destination=str(source_path),
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".reactor-matrix-source-",
            dir=source_path.parent,
        ) as temporary:
            checkout = Path(temporary) / "checkout"
            _run_git(
                [
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    config.source_url,
                    str(checkout),
                ]
            )
            _run_git(
                ["-C", str(checkout), "checkout", "--detach", config.source_revision]
            )
            with suppress(FileExistsError):
                checkout.rename(source_path)
    if not (source_path / ".git").exists():
        raise RuntimeError(f"Matrix source at {source_path} must be a Git checkout")
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"Matrix source revision is {actual}; expected {config.source_revision}"
        )
    _ensure_stateful_patch(source_path)


def _ensure_stateful_patch(source_path: Path) -> None:
    """Apply the adapter's stateful rollout patch exactly once."""
    patch = Path(__file__).with_name("stateful_rollout.patch")
    reverse = _check_git(
        ["-C", str(source_path), "apply", "--reverse", "--check", str(patch)]
    )
    if reverse.returncode == 0:
        return
    forward = _check_git(["-C", str(source_path), "apply", "--check", str(patch)])
    if forward.returncode != 0:
        detail = (
            forward.stderr.strip() or reverse.stderr.strip() or "patch check failed"
        )
        raise RuntimeError(
            f"Matrix source is incompatible with the stateful patch: {detail}"
        )
    logger.info("applying Matrix stateful rollout patch", source=str(source_path))
    _run_git(["-C", str(source_path), "apply", str(patch)])


def ensure_worker_environment(config: MatrixConfig) -> None:
    """Create the isolated upstream Python environment when it is missing or stale."""
    marker = config.worker_python.parents[1] / WORKER_ENV_MARKER
    expected = {
        "version": WORKER_ENV_VERSION,
        "source_revision": config.source_revision,
        "python": WORKER_PYTHON_VERSION,
        "torch": WORKER_TORCH,
        "torchvision": WORKER_TORCHVISION,
        "index_url": WORKER_INDEX_URL,
    }
    if config.worker_python.is_file() and _json_matches(marker, expected):
        return
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare the Matrix worker environment")
    requirements = config.source_path / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(
            f"Matrix requirements file does not exist: {requirements}"
        )
    environment_dir = config.worker_python.parents[1]
    cache_root = config.source_path.parent / ".reactor-uv"
    uv_environment = os.environ.copy()
    uv_environment.update(
        {
            "UV_CACHE_DIR": str(cache_root / "cache"),
            "UV_PYTHON_INSTALL_DIR": str(cache_root / "python"),
        }
    )
    logger.info(
        "preparing Matrix worker environment",
        python=WORKER_PYTHON_VERSION,
        destination=str(environment_dir),
    )
    _run_uv(
        [
            uv,
            "venv",
            "--python",
            WORKER_PYTHON_VERSION,
            "--clear",
            str(environment_dir),
        ],
        uv_environment,
    )
    _run_uv(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(config.worker_python),
            WORKER_TORCH,
            WORKER_TORCHVISION,
            "--index-url",
            WORKER_INDEX_URL,
        ],
        uv_environment,
    )
    _run_uv(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(config.worker_python),
            "--requirement",
            str(requirements),
            "ftfy==6.3.1",
        ],
        uv_environment,
    )
    if not config.worker_python.is_file():
        raise RuntimeError(
            f"Matrix worker environment did not create Python at {config.worker_python}"
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(pending, marker)


def snapshot_marker_matches(local_dir: Path, asset: MatrixAsset) -> bool:
    """Return whether a local snapshot marker matches the pinned public revision."""
    return _json_matches(
        local_dir / SNAPSHOT_MARKER,
        {"repo_id": asset.repo_id, "revision": asset.revision},
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


def _source_path(value: object) -> Path:
    """Resolve the Matrix checkout under the CLI-managed weights directory."""
    configured = os.environ.get(SOURCE_ENV)
    path = Path(configured if configured else str(value)).expanduser()
    candidate = path if path.is_absolute() else get_weights_path() / path
    return Path(os.path.abspath(candidate))


def _json_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a JSON completion marker exactly matches its expected identity."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return document == expected


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git and report a public-resource preparation error."""
    try:
        return subprocess.run(
            _git_command(arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to prepare the Matrix source") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to prepare the Matrix source: {detail}") from error


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
        raise RuntimeError("Git is required to prepare the Matrix source") from error


def _git_command(arguments: list[str]) -> list[str]:
    """Trust only the configured checkout when a bind mount changes its owner."""
    if len(arguments) >= 2 and arguments[0] == "-C":
        return ["git", "-c", f"safe.directory={arguments[1]}", *arguments]
    return ["git", *arguments]


def _run_uv(command: list[str], environment: dict[str, str]) -> None:
    """Run one uv preparation command and preserve its full failure output."""
    try:
        subprocess.run(command, check=True, env=environment)
    except FileNotFoundError as error:
        raise RuntimeError(
            "uv is required to prepare the Matrix worker environment"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Unable to prepare the Matrix Python 3.10 worker environment"
        ) from error


def _restore_default_sample(config: MatrixConfig) -> None:
    """Restore missing default inputs from the pinned public source revision."""
    relative_paths = (ANCHOR_IMAGE, CAMERA)
    missing = [
        path for path in relative_paths if not (config.source_path / path).is_file()
    ]
    if not missing:
        return
    logger.info(
        "restoring Matrix default sample",
        files=[str(path) for path in missing],
        revision=config.source_revision,
    )
    try:
        _run_git(
            [
                "-C",
                str(config.source_path),
                "checkout",
                config.source_revision,
                "--",
                *(str(path) for path in missing),
            ]
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Unable to restore the default Matrix sample from the pinned source checkout"
        ) from error
    unresolved = [
        str(path) for path in missing if not (config.source_path / path).is_file()
    ]
    if unresolved:
        raise RuntimeError(f"Matrix default sample remains incomplete: {unresolved}")


def _validate_bootstrap_paths(config: MatrixConfig) -> None:
    """Require the source files needed to restore inputs and download weights."""
    files = {
        "Matrix worker Python": config.worker_python,
        "Matrix inference config": config.inference_config,
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")


def _ensure_model_assets(config: MatrixConfig) -> None:
    """Download every missing or mismatched model snapshot at its pinned revision."""
    _ensure_hf_snapshot(
        config,
        config.checkpoint,
        name="Matrix distilled checkpoint",
        local_dir=config.checkpoint.path.parent,
        required_files=(config.checkpoint.path,),
    )
    _ensure_hf_snapshot(
        config,
        config.wan,
        name="Wan2.2 base model",
        local_dir=config.wan.path,
        required_files=tuple(config.wan.path / path for path in _WAN_REQUIRED_FILES),
        ignore_patterns=("assets/*", "examples/*"),
    )
    _ensure_hf_snapshot(
        config,
        config.depth,
        name="Depth-Anything-3 model",
        local_dir=config.depth.path,
        required_files=tuple(
            config.depth.path / path for path in _DEPTH_REQUIRED_FILES
        ),
    )


def _ensure_hf_snapshot(
    config: MatrixConfig,
    asset: MatrixAsset,
    *,
    name: str,
    local_dir: Path,
    required_files: tuple[Path, ...],
    ignore_patterns: tuple[str, ...] = (),
) -> None:
    """Download a missing Hugging Face snapshot with the Matrix worker environment."""
    if snapshot_marker_matches(local_dir, asset) and all(
        _is_nonempty_file(path) for path in required_files
    ):
        return
    if not config.worker_python.is_file():
        raise FileNotFoundError(
            f"Matrix worker Python does not exist: {config.worker_python}. "
            "Create the documented upstream environment before starting Reactor."
        )
    logger.info(
        "downloading Matrix model asset",
        asset=name,
        repo_id=asset.repo_id,
        revision=asset.revision,
        destination=str(local_dir),
    )
    downloader = Path(__file__).with_name("download_snapshot.py")
    command = [
        str(config.worker_python),
        str(downloader),
        "--repo-id",
        asset.repo_id,
        "--revision",
        asset.revision,
        "--local-dir",
        str(local_dir),
    ]
    for pattern in ignore_patterns:
        command.extend(("--ignore-pattern", pattern))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Unable to download {name} from {asset.repo_id}. Check network access and "
            "run `hf auth login` if Hugging Face requests authentication."
        ) from error
    unresolved = [str(path) for path in required_files if not _is_nonempty_file(path)]
    if unresolved:
        raise RuntimeError(
            f"{name} download is incomplete; missing files: {unresolved}"
        )
    if not snapshot_marker_matches(local_dir, asset):
        raise RuntimeError(f"{name} download did not record its pinned revision")


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a model file exists and contains data."""
    return path.is_file() and path.stat().st_size > 0


def _validate_runtime_paths(config: MatrixConfig) -> None:
    """Require every prepared source, input, environment, and model asset."""
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
