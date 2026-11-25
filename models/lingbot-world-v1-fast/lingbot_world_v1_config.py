"""Read the LingBot-World v1 config and check its source and weights."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
CACHE_ROOT = Path("~/.cache/lingbot-world-v1")
CHECKPOINT_PATH = Path("checkpoints/lingbot-world-base-cam")
FAST_SUBDIR = Path("lingbot_world_fast")

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
BASE_PATTERNS = (
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
)
FAST_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors.index.json",
    *(f"model-{index:05d}-of-00016.safetensors" for index in range(1, 17)),
)
BASE_REQUIRED_FILES = (
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

    source_path: Path
    world_size: int
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


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> LingBotConfig:
    """Read and validate the LingBot adapter YAML.

    ``weights_root`` is the folder that ``prepare_weights.py`` builds. It
    defaults to Runtime's weights path.
    """
    if config_path is None:
        raise ValueError("LingBot-World v1 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")
    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    world_size = inference.get("world_size", 1)
    if type(world_size) is not int or world_size not in (1, 2, 4):
        raise ValueError("inference.world_size must be 1, 2, or 4")
    stream = _mapping(document.get("stream"), "stream")
    motion = _mapping(document.get("motion"), "motion")
    root = (weights_root or get_weights_path()).resolve()
    source_path = _source_path(source.get("path"))
    checkpoint = _asset(
        root / CHECKPOINT_PATH,
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
    runtime_root = CACHE_ROOT.expanduser()
    return LingBotConfig(
        world_size=world_size,
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


def check_runtime(config: LingBotConfig) -> None:
    """Point caches at the home folder, then check the source and weights.

    This function never downloads. The image holds the patched source, and
    ``prepare_weights.py`` builds the weights folder.
    """
    os.environ.update(_cache_environment())
    check_source(config)
    check_weights(config)
    _validate_runtime_paths(config)


def check_source(config: LingBotConfig) -> None:
    """Require the pinned source checkout with the stateful patch applied."""
    source_path = config.source_path
    if not (source_path / ".git").is_dir():
        raise FileNotFoundError(
            f"LingBot source is missing at {source_path}; rebuild the image with `reactor build`"
        )
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"LingBot source revision is {actual}; expected {config.source_revision}"
        )
    patch = Path(__file__).with_name("stateful_rollout.patch")
    reverse = _check_git(
        ["-C", str(source_path), "apply", "--reverse", "--check", str(patch)]
    )
    if reverse.returncode != 0:
        raise RuntimeError(
            "LingBot source lacks the stateful rollout patch; rebuild the image with `reactor build`"
        )


def check_weights(config: LingBotConfig) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    required = [config.checkpoint.path / name for name in BASE_REQUIRED_FILES]
    required += [config.fast_checkpoint.path / name for name in FAST_REQUIRED_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"LingBot weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def _validate_runtime_paths(config: LingBotConfig) -> None:
    """Fail startup with precise paths when source or assets are incomplete."""
    required = [
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
    """Return the source checkout path. The image build places it here."""
    return Path(str(value or "/opt/lingbot-world")).expanduser().resolve()


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


def _cache_environment() -> dict[str, str]:
    """Return cache variables that point into the home folder, never the weights folder."""
    cache_root = CACHE_ROOT.expanduser()
    environment = {
        "TORCH_HOME": str(cache_root / "torch"),
        "TMPDIR": str(cache_root / "tmp"),
    }
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return environment


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
