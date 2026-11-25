"""Read the DreamX-World config and check its source and inference assets."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml
from dreamx_world_model import DreamXConfig, RepositoryAsset
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
DREAMX_FILE = "model.safetensors"
WAN_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> DreamXConfig:
    """Read and validate the DreamX-World adapter YAML.

    ``weights_root`` is the folder that ``prepare_weights.py`` builds. It
    defaults to Runtime's weights path.
    """
    if config_path is None:
        raise ValueError("DreamX-World requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    inputs = _mapping(document.get("inputs"), "inputs")
    stream = _mapping(document.get("stream", {}), "stream")
    weights_root = (weights_root or get_weights_path()).resolve()
    source_path = _source_path(source["path"], weights_root)

    random_images_raw = inputs.get("random_images")
    if not isinstance(random_images_raw, list) or not random_images_raw:
        raise ValueError("inputs.random_images must be a non-empty YAML list")
    motion_speed = float(inference.get("motion_speed", 1.5))
    if motion_speed <= 0:
        raise ValueError("inference.motion_speed must be positive")
    color_strength = float(inference.get("color_correction_strength", 1.0))
    if not 0.0 <= color_strength <= 1.0:
        raise ValueError("inference.color_correction_strength must be between 0 and 1")
    max_chunks = int(stream.get("max_chunks_per_rollout", 512))
    if max_chunks < 1:
        raise ValueError("stream.max_chunks_per_rollout must be positive")
    default_upload_prompt = str(inputs.get("default_upload_prompt", "")).strip()
    if not default_upload_prompt:
        raise ValueError("inputs.default_upload_prompt must be non-empty")

    dreamx = _asset(weights_root, assets.get("dreamx"), "assets.dreamx")
    wan = _asset(weights_root, assets.get("wan"), "assets.wan")
    return DreamXConfig(
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        upstream_config=_resolve(source_path, inference["config"]),
        transformer_config=_resolve(source_path, inference["transformer_config"]),
        evaluation_inputs=_resolve(source_path, inputs["evaluation_inputs"]),
        random_images=tuple(
            _resolve(source_path, value) for value in random_images_raw
        ),
        dreamx=dreamx,
        wan=wan,
        seed=int(inference.get("seed", 42)),
        motion_speed=motion_speed,
        color_correction_strength=color_strength,
        max_chunks_per_rollout=max_chunks,
        default_upload_prompt=default_upload_prompt,
    )


def check_runtime_assets(config: DreamXConfig) -> dict[Path, str]:
    """Check the source and weights, then return built-in image prompts.

    This function never downloads. The image holds the source at the pinned
    revision, and ``prepare_weights.py`` builds the weights folder.
    """
    _check_source(config)
    check_weights(config)
    _validate_runtime_paths(config)
    return _load_scene_prompts(config)


def _check_source(config: DreamXConfig) -> None:
    """Require the pinned clean source checkout in the image."""
    path = config.source_path
    if not (path / ".git").exists():
        raise FileNotFoundError(
            f"DreamX-World source is missing at {path}; "
            "rebuild the image with `reactor build`"
        )
    actual = _run_git(["-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"DreamX-World source revision is {actual}; expected {config.source_revision}"
        )
    tracked_changes = _run_git(
        ["-C", str(path), "status", "--porcelain", "--untracked-files=no"]
    ).stdout.strip()
    if tracked_changes:
        raise RuntimeError(
            "DreamX-World source has tracked modifications; use an unmodified pinned checkout"
        )


def check_weights(config: DreamXConfig) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    required = [config.dreamx.path] + [
        config.wan.path / name for name in WAN_REQUIRED_FILES
    ]
    missing = [
        str(path) for path in required if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"DreamX-World weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def _validate_runtime_paths(config: DreamXConfig) -> None:
    """Require the pinned source files, configs, and built-in images."""
    files = {
        "DreamX upstream config": config.upstream_config,
        "DreamX transformer config": config.transformer_config,
        "DreamX evaluation inputs": config.evaluation_inputs,
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for path in config.random_images:
        if not path.is_file():
            raise FileNotFoundError(f"DreamX built-in image does not exist: {path}")


def _load_scene_prompts(config: DreamXConfig) -> dict[Path, str]:
    """Return configured built-in images paired with upstream evaluation prompts."""
    document = json.loads(config.evaluation_inputs.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise TypeError("DreamX evaluation input file must contain a JSON list")
    prompts: dict[Path, str] = {}
    for item in document:
        if not isinstance(item, dict) or not isinstance(item.get("image_path"), str):
            continue
        relative = str(item["image_path"]).removeprefix("./")
        image = (config.source_path / relative).resolve()
        prompt = str(item.get("caption", item.get("prompt", ""))).strip()
        if prompt:
            prompts[image] = prompt
    missing = [path for path in config.random_images if path.resolve() not in prompts]
    if missing:
        raise ValueError(
            "DreamX built-in images are missing prompts in the upstream evaluation JSON: "
            + ", ".join(str(path) for path in missing)
        )
    return {path.resolve(): prompts[path.resolve()] for path in config.random_images}


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _asset(base: Path, value: object, name: str) -> RepositoryAsset:
    """Return one local asset path and immutable public repository identity."""
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a public repository")
    return RepositoryAsset(
        path=_resolve(base, document["path"]),
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _source_path(value: object, weights_root: Path) -> Path:
    """Resolve the configured source path. An absolute path stays as it is."""
    return _resolve(weights_root, value)


def _resolve(base: Path, value: object) -> Path:
    """Resolve a configured path relative to its owning directory."""
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _revision(value: object, name: str) -> str:
    """Return a full immutable Git-style revision."""
    revision = str(value or "")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    """Return a public HTTPS repository URL."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git and report its full public-resource error."""
    command = ["git", *arguments]
    if len(arguments) >= 2 and arguments[0] == "-C":
        command = ["git", "-c", f"safe.directory={arguments[1]}", *arguments]
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Git is required to check the DreamX-World source"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to check DreamX-World source: {detail}") from error
