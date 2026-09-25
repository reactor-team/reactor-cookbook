"""Prepare and validate public EVOKE source, environments, and model assets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

logger = logging.getLogger(__name__)

SOURCE_ENV = "EVOKE_PATH"
WORKER_PYTHON = Path(".reactor-venv/bin/python")
WORKER_ENV_MARKER = ".reactor-worker-environment.json"
SNAPSHOT_MARKER = ".reactor-snapshot.json"
WORKER_ENV_VERSION = 4
WORKER_PYTHON_VERSION = "3.10"
WORKER_TORCH = "torch==2.9.1"
WORKER_TORCHVISION = "torchvision==0.24.1"
WORKER_INDEX_URL = "https://download.pytorch.org/whl/cu128"
ATTENTION_RUNTIME = "flash-attn-4==4.0.0b26"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")

_EVOKE_REQUIRED = (
    "evoke-base/model_index.json",
    "evoke-base/vae/diffusion_pytorch_model.safetensors",
    "evoke-base/text_encoder/model.safetensors.index.json",
    "evoke/stage3_post_distillation/transformer/config.json",
    "evoke/stage3_post_distillation/transformer/diffusion_pytorch_model.safetensors.index.json",
)


@dataclass(frozen=True)
class HubAsset:
    """Describe one immutable public Hub snapshot."""

    repo_id: str
    revision: str


@dataclass(frozen=True)
class EvokeConfig:
    """Hold validated EVOKE adapter settings."""

    source_path: Path
    source_url: str
    source_revision: str
    worker_python: Path
    weights: HubAsset
    vigeo: HubAsset
    seed: int
    stability_prompt: str
    max_chunks: int
    translation_units_per_second: float
    rotation_degrees_per_second: float
    reference_seconds: float

    @property
    def base_model(self) -> Path:
        """Return the EVOKE base-component directory."""
        return self.source_path / "models/evoke-base"

    @property
    def transformer(self) -> Path:
        """Return the post-distillation transformer directory."""
        return self.source_path / "models/evoke/stage3_post_distillation"

    @property
    def vigeo_path(self) -> Path:
        """Return the required ViGeo weight directory."""
        return self.source_path / "models/ViGeo1.1"

    @property
    def default_image(self) -> Path:
        """Return the bundled i2v conditioning image."""
        return self.source_path / "examples/i2v/image.jpg"


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> EvokeConfig:
    """Read and validate the EVOKE adapter YAML."""
    if config_path is None:
        raise ValueError("EVOKE requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")
    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    motion = _mapping(document.get("motion"), "motion")
    stream = _mapping(document.get("stream"), "stream")
    source_path = _source_path(
        source.get("path"),
        weights_root if weights_root is not None else config_path.resolve().parent,
    )
    max_chunks = int(stream.get("max_chunks", 2048))
    if max_chunks < 12:
        raise ValueError("stream.max_chunks must be at least 12")
    translation_speed = float(motion.get("translation_units_per_second", 1.0))
    rotation_speed = float(motion.get("rotation_degrees_per_second", 6.0))
    if translation_speed <= 0 or rotation_speed <= 0:
        raise ValueError("motion rates must be positive")
    reference_seconds = float(inference.get("reference_seconds", 5.0))
    if reference_seconds <= 0:
        raise ValueError("inference.reference_seconds must be positive")
    stability_prompt = str(inference.get("stability_prompt", "")).strip()
    if not stability_prompt or len(stability_prompt) > 4096:
        raise ValueError(
            "inference.stability_prompt must contain between 1 and 4096 characters"
        )
    return EvokeConfig(
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        worker_python=source_path / WORKER_PYTHON,
        weights=_asset(assets.get("weights"), "assets.weights"),
        vigeo=_asset(assets.get("vigeo"), "assets.vigeo"),
        seed=int(inference.get("seed", 42)),
        stability_prompt=stability_prompt,
        max_chunks=max_chunks,
        translation_units_per_second=translation_speed,
        rotation_degrees_per_second=rotation_speed,
        reference_seconds=reference_seconds,
    )


def prepare_runtime(config: EvokeConfig) -> None:
    """Prepare the pinned source, Python worker, and inference assets."""
    _configure_download_caches(config.source_path.parent)
    ensure_source_checkout(config)
    ensure_worker_environment(config)
    _ensure_model_assets(config)
    _validate_runtime_paths(config)


def ensure_source_checkout(config: EvokeConfig) -> None:
    """Clone the pinned EVOKE source and apply the resumable-chunk patch."""
    source_path = config.source_path
    if not source_path.exists():
        logger.info(
            "downloading EVOKE source checkout %s revision=%s to %s",
            config.source_url,
            config.source_revision,
            source_path,
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".reactor-evoke-source-", dir=source_path.parent
        ) as temp:
            checkout = Path(temp) / "checkout"
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
        raise RuntimeError(f"EVOKE source at {source_path} must be a Git checkout")
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"EVOKE source revision is {actual}; expected {config.source_revision}"
        )
    _ensure_stateful_patch(source_path)


def ensure_worker_environment(config: EvokeConfig) -> None:
    """Create the isolated Python 3.10 inference environment on the weights volume."""
    marker = config.worker_python.parents[1] / WORKER_ENV_MARKER
    requirements = Path(__file__).with_name("worker-requirements.txt")
    expected = {
        "version": WORKER_ENV_VERSION,
        "source_revision": config.source_revision,
        "python": WORKER_PYTHON_VERSION,
        "torch": WORKER_TORCH,
        "torchvision": WORKER_TORCHVISION,
        "attention_runtime": ATTENTION_RUNTIME,
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    }
    if config.worker_python.is_file() and _json_matches(marker, expected):
        return
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare the EVOKE worker environment")
    environment_dir = config.worker_python.parents[1]
    environment = os.environ.copy()
    logger.info("preparing EVOKE worker environment: %s", environment_dir)
    _run_uv(
        [
            uv,
            "venv",
            "--python",
            WORKER_PYTHON_VERSION,
            "--clear",
            str(environment_dir),
        ],
        environment,
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
            "--extra-index-url",
            "https://pypi.org/simple",
        ],
        environment,
    )
    _run_uv(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(config.worker_python),
            "-r",
            str(requirements),
        ],
        environment,
    )
    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(pending, marker)


def _configure_download_caches(weights_root: Path) -> None:
    """Place package, model, compiler, and temporary caches on the weights volume."""
    cache_root = weights_root / ".cache"
    defaults = {
        "UV_CACHE_DIR": cache_root / "uv",
        "UV_PYTHON_INSTALL_DIR": cache_root / "python",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "CUTE_DSL_CACHE_DIR": cache_root / "cute-dsl",
        "TMPDIR": cache_root / "tmp",
    }
    for variable, default in defaults.items():
        directory = Path(os.environ.setdefault(variable, str(default))).expanduser()
        directory.mkdir(parents=True, exist_ok=True)


def _ensure_stateful_patch(source_path: Path) -> None:
    """Apply the adapter's stateful chunk-boundary patch exactly once."""
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
            f"EVOKE source is incompatible with the stateful patch: {detail}"
        )
    logger.info("applying EVOKE stateful rollout patch: %s", source_path)
    _run_git(["-C", str(source_path), "apply", str(patch)])


def _ensure_model_assets(config: EvokeConfig) -> None:
    """Download the minimal post-distillation and ViGeo snapshots."""
    _ensure_snapshot(
        config,
        asset=config.weights,
        name="EVOKE post-distillation weights",
        local_dir=config.source_path / "models",
        required=tuple(
            config.source_path / "models" / path for path in _EVOKE_REQUIRED
        ),
        allow_patterns=("evoke-base/**", "evoke/stage3_post_distillation/**"),
    )
    _ensure_snapshot(
        config,
        asset=config.vigeo,
        name="ViGeo 1.1 weights",
        local_dir=config.vigeo_path,
        required=(config.vigeo_path / "vigeo.pt",),
        allow_patterns=("vigeo.pt",),
    )


def _ensure_snapshot(
    config: EvokeConfig,
    *,
    asset: HubAsset,
    name: str,
    local_dir: Path,
    required: tuple[Path, ...],
    allow_patterns: tuple[str, ...],
) -> None:
    """Download one missing immutable Hub snapshot through the worker environment."""
    marker = local_dir / SNAPSHOT_MARKER
    identity = {
        "repo_id": asset.repo_id,
        "revision": asset.revision,
        "allow": list(allow_patterns),
    }
    if _json_matches(marker, identity) and all(_nonempty(path) for path in required):
        return
    logger.info(
        "downloading EVOKE model asset %s repo=%s to %s",
        name,
        asset.repo_id,
        local_dir,
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
    for pattern in allow_patterns:
        command.extend(("--allow-pattern", pattern))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Unable to download {name} from {asset.repo_id}") from error
    unresolved = [str(path) for path in required if not _nonempty(path)]
    if unresolved:
        raise RuntimeError(
            f"{name} download is incomplete; missing files: {unresolved}"
        )


def _validate_runtime_paths(config: EvokeConfig) -> None:
    """Require every prepared executable, input, and weight used at load time."""
    files = {
        "EVOKE worker Python": config.worker_python,
        "default image": config.default_image,
        "ViGeo checkpoint": config.vigeo_path / "vigeo.pt",
        "base model index": config.base_model / "model_index.json",
        "post-distillation transformer config": config.transformer
        / "transformer/config.json",
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _asset(value: object, name: str) -> HubAsset:
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a public repository")
    return HubAsset(repo_id, _revision(document.get("revision"), f"{name}.revision"))


def _revision(value: object, name: str) -> str:
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _source_path(value: object, weights_root: Path) -> Path:
    override = os.environ.get(SOURCE_ENV)
    configured = Path(override if override else str(value)).expanduser()
    candidate = configured if configured.is_absolute() else weights_root / configured
    return Path(os.path.abspath(candidate))


def _json_matches(path: Path, expected: object) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")) == expected
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            _git_command(arguments), check=True, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to prepare the EVOKE source") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to prepare the EVOKE source: {detail}") from error


def _check_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            _git_command(arguments), check=False, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to prepare the EVOKE source") from error


def _git_command(arguments: list[str]) -> list[str]:
    if len(arguments) >= 2 and arguments[0] == "-C":
        return ["git", "-c", f"safe.directory={arguments[1]}", *arguments]
    return ["git", *arguments]


def _run_uv(command: list[str], environment: dict[str, str]) -> None:
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Unable to prepare the EVOKE Python 3.10 worker environment"
        ) from error
