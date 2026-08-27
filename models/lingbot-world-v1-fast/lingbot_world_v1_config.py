"""Prepare pinned public LingBot-World v1 source and model assets."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

SOURCE_ENV = "LINGBOT_WORLD_V1_PATH"
WORKER_PYTHON = Path(".venv/bin/python")
CHECKPOINT_PATH = Path("checkpoints/lingbot-world-base-cam")
FAST_SUBDIR = Path("lingbot_world_fast")
SNAPSHOT_MARKER = ".reactor-snapshot.json"
WORKER_ENV_MARKER = ".reactor-worker-environment.json"
WORKER_ENV_VERSION = 1
WORKER_PYTHON_VERSION = "3.12"
WORKER_TORCH = "torch==2.8.0"
WORKER_TORCHVISION = "torchvision==0.23.0"
WORKER_TORCHAUDIO = "torchaudio==2.8.0"
WORKER_INDEX_URL = "https://download.pytorch.org/whl/cu128"
WORKER_FLASH_ATTN = "flash-attn==2.8.3"

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_BASE_PATTERNS = (
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
)
_BASE_REQUIRED_FILES = (
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


@dataclass(frozen=True)
class ModelAsset:
    """Describe a pinned public model snapshot and its local destination."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class Sample:
    """Describe one built-in image, prompt, and camera calibration."""

    image: Path
    intrinsics: Path
    prompt: str


@dataclass(frozen=True)
class LingBotConfig:
    """Hold validated LingBot adapter settings."""

    worker_python: Path
    source_path: Path
    source_url: str
    source_revision: str
    checkpoint: ModelAsset
    fast_checkpoint: ModelAsset
    samples: tuple[Sample, ...]
    seed: int
    max_chunks: int
    context_latents: int
    max_area: int
    shift: float
    translation_units_per_latent: float
    rotation_degrees_per_latent: float
    runtime_root: Path


def read_config(config_path: Path | None) -> LingBotConfig:
    """Read and validate the LingBot adapter YAML."""
    if config_path is None:
        raise ValueError("LingBot-World v1 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")
    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    stream = _mapping(document.get("stream"), "stream")
    motion = _mapping(document.get("motion"), "motion")
    source_path = _source_path(source.get("path"))
    checkpoint = _asset(
        source_path / CHECKPOINT_PATH,
        assets.get("base"),
        "assets.base",
    )
    fast_checkpoint = _asset(
        checkpoint.path / FAST_SUBDIR,
        assets.get("fast"),
        "assets.fast",
    )
    samples = _samples(source_path, document.get("samples"))
    max_chunks = int(stream.get("max_chunks", 320))
    context_latents = int(stream.get("context_latents", 21))
    if max_chunks <= 0 or max_chunks * 3 > 1024:
        raise ValueError(
            "stream.max_chunks must keep the latent RoPE timeline within 1024"
        )
    if context_latents < 3 or context_latents % 3:
        raise ValueError("stream.context_latents must be a positive multiple of 3")
    max_area = int(inference.get("max_area", 480 * 832))
    if max_area <= 0:
        raise ValueError("inference.max_area must be positive")
    translation = float(motion.get("translation_units_per_latent", 1.0))
    rotation = float(motion.get("rotation_degrees_per_latent", 8.0))
    if translation <= 0 or rotation <= 0:
        raise ValueError("motion rates must be positive")
    runtime_root = source_path.parent / ".reactor-lingbot-world-v1"
    return LingBotConfig(
        worker_python=source_path / WORKER_PYTHON,
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        checkpoint=checkpoint,
        fast_checkpoint=fast_checkpoint,
        samples=samples,
        seed=int(inference.get("seed", 42)),
        max_chunks=max_chunks,
        context_latents=context_latents,
        max_area=max_area,
        shift=float(inference.get("shift", 10.0)),
        translation_units_per_latent=translation,
        rotation_degrees_per_latent=rotation,
        runtime_root=runtime_root,
    )


def prepare_runtime(config: LingBotConfig) -> None:
    """Prepare the pinned source, isolated environment, and Fast checkpoints."""
    ensure_source_checkout(config)
    ensure_worker_environment(config)
    ensure_model_assets(config)
    _validate_runtime_paths(config)


def ensure_source_checkout(config: LingBotConfig) -> None:
    """Clone the pinned source and apply the resumable Fast rollout extension."""
    source_path = config.source_path
    if not source_path.exists():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "downloading LingBot-World source",
            url=config.source_url,
            revision=config.source_revision,
            destination=str(source_path),
        )
        with tempfile.TemporaryDirectory(
            prefix=".reactor-lingbot-source-",
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
    if not (source_path / ".git").is_dir():
        raise RuntimeError(f"LingBot source at {source_path} must be a Git checkout")
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"LingBot source revision is {actual}; expected {config.source_revision}"
        )
    _ensure_stateful_patch(source_path)


def ensure_worker_environment(config: LingBotConfig) -> None:
    """Create the isolated NumPy-1 model environment when missing or stale."""
    marker = config.worker_python.parents[1] / WORKER_ENV_MARKER
    expected = {
        "version": WORKER_ENV_VERSION,
        "source_revision": config.source_revision,
        "python": WORKER_PYTHON_VERSION,
        "torch": WORKER_TORCH,
        "torchvision": WORKER_TORCHVISION,
        "torchaudio": WORKER_TORCHAUDIO,
        "flash_attn": WORKER_FLASH_ATTN,
        "index_url": WORKER_INDEX_URL,
    }
    if config.worker_python.is_file() and _json_matches(marker, expected):
        return
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare the LingBot worker environment")
    environment_dir = config.worker_python.parents[1]
    cache_root = config.source_path.parent / ".cache"
    uv_environment = _download_environment(cache_root)
    logger.info(
        "preparing LingBot model environment",
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
            WORKER_TORCHAUDIO,
            "--index-url",
            WORKER_INDEX_URL,
        ],
        uv_environment,
    )
    requirements = Path(__file__).with_name("worker-requirements.txt")
    _run_uv(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(config.worker_python),
            "--requirement",
            str(requirements),
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
            "--no-build-isolation",
            WORKER_FLASH_ATTN,
        ],
        uv_environment,
    )
    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(pending, marker)


def ensure_model_assets(config: LingBotConfig) -> None:
    """Download only the base components and Fast weights used by this adapter."""
    _ensure_snapshot(
        config,
        config.checkpoint,
        allow_patterns=_BASE_PATTERNS,
        required_files=_BASE_REQUIRED_FILES,
    )
    fast_required = (
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
        *(f"model-{index:05d}-of-00016.safetensors" for index in range(1, 17)),
    )
    _ensure_snapshot(
        config,
        config.fast_checkpoint,
        allow_patterns=(),
        required_files=fast_required,
    )


def _ensure_snapshot(
    config: LingBotConfig,
    asset: ModelAsset,
    *,
    allow_patterns: Sequence[str],
    required_files: Sequence[str],
) -> None:
    """Download one resumable pinned snapshot and verify its required files."""
    marker = asset.path / SNAPSHOT_MARKER
    expected = {"repo_id": asset.repo_id, "revision": asset.revision}
    if _json_matches(marker, expected) and all(
        (asset.path / relative).is_file() for relative in required_files
    ):
        return
    asset.path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloading LingBot model snapshot",
        repo_id=asset.repo_id,
        revision=asset.revision,
        destination=str(asset.path),
    )
    command = [
        str(config.worker_python),
        str(Path(__file__).with_name("download_snapshot.py")),
        "--repo-id",
        asset.repo_id,
        "--revision",
        asset.revision,
        "--local-dir",
        str(asset.path),
    ]
    for pattern in allow_patterns:
        command.extend(["--allow-pattern", pattern])
    subprocess.run(
        command,
        check=True,
        env=_download_environment(config.source_path.parent / ".cache"),
    )
    missing = [
        relative for relative in required_files if not (asset.path / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"snapshot {asset.repo_id}@{asset.revision} is missing required files: {missing}"
        )
    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(pending, marker)


def _ensure_stateful_patch(source_path: Path) -> None:
    """Apply the adapter's stateful Fast rollout extension exactly once."""
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
            f"LingBot source is incompatible with the stateful patch: {detail}"
        )
    logger.info("applying LingBot stateful rollout extension", source=str(source_path))
    _run_git(["-C", str(source_path), "apply", str(patch)])


def _validate_runtime_paths(config: LingBotConfig) -> None:
    """Fail startup with precise paths when source or assets are incomplete."""
    required = [
        config.worker_python,
        config.source_path / "wan" / "interactive_fast.py",
        config.checkpoint.path / "Wan2.1_VAE.pth",
        config.fast_checkpoint.path / "config.json",
    ]
    for sample in config.samples:
        required.extend([sample.image, sample.intrinsics])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LingBot runtime assets are missing: {missing}")
    config.runtime_root.mkdir(parents=True, exist_ok=True)


def _samples(source_path: Path, value: object) -> tuple[Sample, ...]:
    """Return validated built-in samples from the pinned source checkout."""
    if not isinstance(value, list) or not value:
        raise ValueError("samples must be a non-empty YAML list")
    result = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"samples[{index}]")
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"samples[{index}].prompt must not be empty")
        result.append(
            Sample(
                image=source_path / str(item.get("image", "")),
                intrinsics=source_path / str(item.get("intrinsics", "")),
                prompt=prompt,
            )
        )
    return tuple(result)


def _asset(path: Path, value: object, name: str) -> ModelAsset:
    """Return one pinned public model asset."""
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(
            f"{name}.repo_id must identify a public Hugging Face repository"
        )
    return ModelAsset(
        path=path,
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _source_path(value: object) -> Path:
    """Resolve the source checkout under Runtime's mounted weights root."""
    configured = os.environ.get(SOURCE_ENV)
    path = Path(
        configured if configured else str(value or "lingbot-world-v1")
    ).expanduser()
    return (
        path.resolve() if path.is_absolute() else (get_weights_path() / path).resolve()
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _revision(value: object, name: str) -> str:
    """Return one full immutable Git-style revision."""
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    """Return a public HTTPS source URL."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _json_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a JSON marker exactly matches expected metadata."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) == dict(expected)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _download_environment(cache_root: Path) -> dict[str, str]:
    """Return download and build caches rooted beside the model checkout."""
    environment = os.environ.copy()
    cache_root.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(cache_root / "python"),
            "HF_HOME": str(cache_root / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "TORCH_HOME": str(cache_root / "torch"),
            "PIP_CACHE_DIR": str(cache_root / "pip"),
            "TMPDIR": str(cache_root / "tmp"),
            "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
        }
    )
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return environment


def _run_uv(command: list[str], environment: Mapping[str, str]) -> None:
    """Run uv while preserving full subprocess failures."""
    subprocess.run(command, check=True, env=dict(environment))


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run git and return its captured output."""
    return subprocess.run(
        _git_command(arguments),
        check=True,
        text=True,
        capture_output=True,
    )


def _check_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a non-mutating git check and return its status."""
    return subprocess.run(
        _git_command(arguments),
        check=False,
        text=True,
        capture_output=True,
    )


def _git_command(arguments: list[str]) -> list[str]:
    """Allow Git to inspect the explicit checkout mounted by Runtime."""
    try:
        checkout_index = arguments.index("-C") + 1
        checkout = Path(arguments[checkout_index]).resolve()
    except (ValueError, IndexError):
        return ["git", *arguments]
    return ["git", "-c", f"safe.directory={checkout}", *arguments]
