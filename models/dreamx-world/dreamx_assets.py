"""Prepare pinned DreamX-World source and inference assets."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import yaml
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

from dreamx_types import DreamXConfig, RepositoryAsset

logger = get_logger(__name__)

SOURCE_ENV = "DREAMX_WORLD_PATH"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_ASSET_MARKER = ".reactor-assets.json"
_WAN_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


def read_config(config_path: Path | None) -> DreamXConfig:
    """Read and validate the DreamX-World adapter YAML."""
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
    weights_root = get_weights_path().resolve()
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


def prepare_runtime_assets(config: DreamXConfig) -> dict[Path, str]:
    """Prepare public source and weights, then return built-in image prompts."""
    _ensure_source_checkout(config)
    _ensure_hf_file(config.dreamx, "model.safetensors", "DreamX-World checkpoint")
    _ensure_hf_files(config.wan, _WAN_REQUIRED_FILES, "Wan2.2 text encoder and VAE")
    _validate_runtime_paths(config)
    return _load_scene_prompts(config)


def _ensure_source_checkout(config: DreamXConfig) -> None:
    """Clone a missing upstream checkout and require the pinned clean revision."""
    path = config.source_path
    if not path.exists():
        logger.info(
            "downloading DreamX-World source",
            url=config.source_url,
            revision=config.source_revision,
            destination=str(path),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".reactor-dreamx-source-", dir=path.parent
        ) as tmp:
            checkout = Path(tmp) / "checkout"
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
                checkout.rename(path)
    if not (path / ".git").exists():
        raise RuntimeError(f"DreamX-World source at {path} must be a Git checkout")
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


def _ensure_hf_file(asset: RepositoryAsset, filename: str, name: str) -> None:
    """Download one pinned Hugging Face file when its completion marker is absent."""
    destination = asset.path
    marker = destination.parent / f".{destination.name}.reactor-asset.json"
    expected = {
        "repo_id": asset.repo_id,
        "revision": asset.revision,
        "files": [filename],
    }
    if (
        destination.is_file()
        and destination.stat().st_size > 0
        and _json_matches(marker, expected)
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloading model asset", asset=name, repo_id=asset.repo_id, file=filename
    )
    downloaded = _hf_hub_download(
        repo_id=asset.repo_id,
        filename=filename,
        revision=asset.revision,
        local_dir=destination.parent,
    )
    if downloaded.resolve() != destination.resolve() or not destination.is_file():
        raise RuntimeError(f"{name} download did not create {destination}")
    _write_json(marker, expected)


def _ensure_hf_files(
    asset: RepositoryAsset,
    filenames: tuple[str, ...],
    name: str,
) -> None:
    """Download the selected files required from one pinned repository."""
    marker = asset.path / _ASSET_MARKER
    expected = {
        "repo_id": asset.repo_id,
        "revision": asset.revision,
        "files": list(filenames),
    }
    if all(
        (asset.path / filename).is_file() for filename in filenames
    ) and _json_matches(marker, expected):
        return
    asset.path.mkdir(parents=True, exist_ok=True)
    logger.info("downloading model assets", asset=name, repo_id=asset.repo_id)
    for filename in filenames:
        _hf_hub_download(
            repo_id=asset.repo_id,
            filename=filename,
            revision=asset.revision,
            local_dir=asset.path,
        )
    missing = [
        filename for filename in filenames if not (asset.path / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"{name} download is incomplete: {', '.join(missing)}")
    _write_json(marker, expected)


def _validate_runtime_paths(config: DreamXConfig) -> None:
    """Require the pinned source, configs, checkpoints, and built-in images."""
    files = {
        "DreamX upstream config": config.upstream_config,
        "DreamX transformer config": config.transformer_config,
        "DreamX evaluation inputs": config.evaluation_inputs,
        "DreamX checkpoint": config.dreamx.path,
        "Wan2.2 VAE": config.wan.path / "Wan2.2_VAE.pth",
        "Wan2.2 text encoder": config.wan.path / "models_t5_umt5-xxl-enc-bf16.pth",
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
    """Resolve the source override or configured path without hard-coded host paths."""
    configured = os.environ.get(SOURCE_ENV)
    return _resolve(weights_root, configured if configured else value)


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
        raise RuntimeError("Git is required to prepare DreamX-World") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(
            f"Unable to prepare DreamX-World source: {detail}"
        ) from error


def _hf_hub_download(
    *, repo_id: str, filename: str, revision: str, local_dir: Path
) -> Path:
    """Download one Hugging Face file and retain its actionable failure detail."""
    try:
        hugging_face = importlib.import_module("huggingface_hub")
        return Path(
            hugging_face.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Unable to download public asset from {repo_id}: {error}"
        ) from error


def _json_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a completion marker exactly matches its asset identity."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) == expected
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    """Atomically write a small asset completion marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    os.replace(pending, path)
