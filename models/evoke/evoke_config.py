"""Read the EVOKE adapter config and check its source, worker, and weights."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

WORKER_PYTHON = Path(".reactor-venv/bin/python")
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
PATCHED_UPSTREAM = Path(__file__).with_name("upstream_patched")
CACHE_ROOT = Path("~/.cache/evoke")
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
    weights_root: Path
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
        return self.weights_root / "evoke-base"

    @property
    def transformer(self) -> Path:
        """Return the post-distillation transformer directory."""
        return self.weights_root / "evoke/stage3_post_distillation"

    @property
    def vigeo_path(self) -> Path:
        """Return the required ViGeo weight directory."""
        return self.weights_root / "ViGeo1.1"

    @property
    def default_image(self) -> Path:
        """Return the bundled i2v conditioning image."""
        return self.source_path / "examples/i2v/image.jpg"


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> EvokeConfig:
    """Read and validate the EVOKE adapter YAML.

    ``weights_root`` is the prepared weights folder. It defaults to ``weights/``
    beside the config.
    """
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
    root = Path(
        weights_root
        if weights_root is not None
        else config_path.resolve().parent / "weights"
    ).resolve()
    source_path = _source_path(source.get("path"), config_path.resolve().parent)
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
        weights_root=root,
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


def check_runtime(config: EvokeConfig) -> None:
    """Point caches at the home folder, then check the source, worker, and weights.

    This function never downloads. The image holds the patched source and the
    worker environment, and ``prepare_weights.py`` builds the weights folder.
    """
    _configure_caches()
    check_source(config)
    check_weights(config)
    _validate_runtime_paths(config)


def check_source(config: EvokeConfig) -> None:
    """Require the pinned source checkout, and patched copies taken from that revision."""
    source_path = config.source_path
    if not (source_path / ".git").exists():
        raise FileNotFoundError(
            f"EVOKE source is missing at {source_path}; rebuild the image with `reactor build`"
        )
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"EVOKE source revision is {actual}; expected {config.source_revision}"
        )
    for copy in sorted(PATCHED_UPSTREAM.rglob("*.py")):
        if f" at {config.source_revision}," not in copy.read_text().splitlines()[0]:
            raise RuntimeError(
                f"{copy} was not copied from EVOKE revision {config.source_revision}; "
                "copy it again from that revision and reapply the change"
            )


def check_weights(config: EvokeConfig) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    required = [config.weights_root / path for path in _EVOKE_REQUIRED]
    required.append(config.vigeo_path / "vigeo.pt")
    missing = [str(path) for path in required if not _nonempty(path)]
    if missing:
        raise FileNotFoundError(
            f"EVOKE weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def _configure_caches() -> None:
    """Place compiler and temporary caches in the home folder, never the weights folder."""
    cache_root = CACHE_ROOT.expanduser()
    defaults = {
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "CUTE_DSL_CACHE_DIR": cache_root / "cute-dsl",
        "TMPDIR": cache_root / "tmp",
    }
    for variable, default in defaults.items():
        directory = Path(os.environ.setdefault(variable, str(default))).expanduser()
        directory.mkdir(parents=True, exist_ok=True)


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


def _source_path(value: object, base: Path) -> Path:
    configured = Path(str(value)).expanduser()
    candidate = configured if configured.is_absolute() else base / configured
    return Path(os.path.abspath(candidate))


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            _git_command(arguments), check=True, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to check the EVOKE source") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to check the EVOKE source: {detail}") from error


def _git_command(arguments: list[str]) -> list[str]:
    if len(arguments) >= 2 and arguments[0] == "-C":
        return ["git", "-c", f"safe.directory={arguments[1]}", *arguments]
    return ["git", *arguments]
