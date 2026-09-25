"""Prepare ABot-World's pinned public source and checkpoint."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

logger = logging.getLogger(__name__)

SOURCE_ENV = "ABOT_WORLD_PATH"
SNAPSHOT_MARKER = ".reactor-snapshot.json"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_FILES = (
    "Wan2.2_VAE.pth",
    "taew2_2.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model.safetensors",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
_SOURCE_FILES = (
    "configs/default_config.yaml",
    "configs/long_forcing_dmd.yaml",
    "pipeline/causal_inference.py",
    "utils/misc.py",
    "utils/wan_wrapper.py",
    "wan/modules/causal_model.py",
)


@dataclass(frozen=True)
class ModelAsset:
    """Describe one pinned public model snapshot and its local directory."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class ExampleScene:
    """Pair one built-in starting image with its scene prompt."""

    image: Path
    prompt: str


@dataclass(frozen=True)
class ABotWorldConfig:
    """Hold validated source, checkpoint, stream, and example settings."""

    source_path: Path
    source_url: str
    source_revision: str
    checkpoint: ModelAsset
    seed: int
    height: int
    width: int
    max_chunks: int
    examples: tuple[ExampleScene, ...]


def read_config(
    config_path: Path | None, weights_root: Path | None = None
) -> ABotWorldConfig:
    """Read and validate the ABot-World adapter YAML."""
    if config_path is None:
        raise ValueError("ABot-World requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    checkpoint_document = _mapping(document.get("checkpoint"), "checkpoint")
    stream = _mapping(document.get("stream"), "stream")
    weights_root = (
        weights_root if weights_root is not None else config_path.resolve().parent
    )
    source_path = _source_path(source.get("path"), weights_root)
    checkpoint = ModelAsset(
        path=_weights_path(
            checkpoint_document.get("path"), "checkpoint.path", weights_root
        ),
        repo_id=_repo_id(checkpoint_document.get("repo_id"), "checkpoint.repo_id"),
        revision=_revision(checkpoint_document.get("revision"), "checkpoint.revision"),
    )
    height = int(stream.get("height", 704))
    width = int(stream.get("width", 1280))
    max_chunks = int(stream.get("max_chunks", 512))
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("stream height and width must be positive multiples of 16")
    if max_chunks < 16:
        raise ValueError("stream.max_chunks must be at least 16")

    examples_document = document.get("examples")
    if not isinstance(examples_document, list) or not examples_document:
        raise ValueError("examples must contain at least one built-in scene")
    examples: list[ExampleScene] = []
    for index, value in enumerate(examples_document):
        example = _mapping(value, f"examples[{index}]")
        image_value = example.get("image")
        prompt = str(example.get("prompt", "")).strip()
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"examples[{index}].image must be a non-empty path")
        if not prompt:
            raise ValueError(f"examples[{index}].prompt must be non-empty")
        image = (config_path.parent / image_value).resolve()
        examples.append(ExampleScene(image=image, prompt=prompt))

    return ABotWorldConfig(
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        checkpoint=checkpoint,
        seed=int(document.get("seed", 42)),
        height=height,
        width=width,
        max_chunks=max_chunks,
        examples=tuple(examples),
    )


def prepare_assets(config: ABotWorldConfig, weights_root: Path) -> None:
    """Prepare the pinned upstream checkout, model snapshot, and built-in images."""
    _ensure_source_checkout(config)
    _ensure_checkpoint(config.checkpoint, weights_root)
    missing_examples = [
        str(scene.image) for scene in config.examples if not scene.image.is_file()
    ]
    if missing_examples:
        raise FileNotFoundError(
            f"ABot-World built-in images are missing: {missing_examples}"
        )


def load_upstream_modules(config: ABotWorldConfig) -> dict[str, Any]:
    """Import the unmodified upstream inference modules from the pinned checkout."""
    missing = [
        path for path in _SOURCE_FILES if not (config.source_path / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"ABot-World source is incomplete; missing: {missing}")
    source = str(config.source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    os.environ["TAEW2_2_CHECKPOINT"] = str(config.checkpoint.path / "taew2_2.pth")

    pipeline_module = importlib.import_module("pipeline")
    misc_module = importlib.import_module("utils.misc")
    wrapper_module = importlib.import_module("utils.wan_wrapper")
    kernels_module = importlib.import_module("wan.modules.helios_kernels")
    return {
        "torch": importlib.import_module("torch"),
        "OmegaConf": importlib.import_module("omegaconf").OmegaConf,
        "pipeline_type": pipeline_module.CausalInferencePipeline,
        "set_seed": misc_module.set_seed,
        "create_vae": wrapper_module.create_vae_from_config,
        "replace_norms": kernels_module.replace_all_norms_with_flash_norms,
        "replace_rope": kernels_module.replace_rope_with_flash_rope,
    }


def build_upstream_config(config: ABotWorldConfig, modules: Mapping[str, Any]) -> Any:
    """Load upstream YAML and point every model component at the pinned snapshot."""
    omega_conf = modules["OmegaConf"]
    source = config.source_path
    upstream = omega_conf.merge(
        omega_conf.load(str(source / "configs/default_config.yaml")),
        omega_conf.load(str(source / "configs/long_forcing_dmd.yaml")),
    )
    checkpoint = config.checkpoint.path
    upstream.taew2_2_checkpoint = str(checkpoint / "taew2_2.pth")
    upstream.lightvae_encoder_checkpoint = str(checkpoint / "Wan2.2_VAE.pth")
    upstream.model_kwargs.model_name = str(checkpoint)
    upstream.text_encoder_kwargs.tokenizer_path = (
        str(checkpoint / "google/umt5-xxl") + "/"
    )
    upstream.text_encoder_kwargs.encoder_pth_path = str(
        checkpoint / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    upstream.vae_kwargs.pretrained_path = str(checkpoint / "Wan2.2_VAE.pth")
    upstream.vae_type = "taew2_2"
    upstream.use_fp8_gemm = False
    return upstream


def _ensure_source_checkout(config: ABotWorldConfig) -> None:
    """Clone the exact upstream revision when its checkout is absent."""
    source_path = config.source_path
    if not source_path.exists():
        logger.info(
            "downloading ABot-World source checkout %s revision=%s to %s",
            config.source_url,
            config.source_revision,
            source_path,
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".reactor-abot-source-", dir=source_path.parent
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
                checkout.rename(source_path)
    if not (source_path / ".git").is_dir():
        raise RuntimeError(f"ABot-World source at {source_path} must be a Git checkout")
    actual = _run_git(["-C", str(source_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"ABot-World source revision is {actual}; expected {config.source_revision}"
        )


def _ensure_checkpoint(asset: ModelAsset, weights_root: Path) -> None:
    """Download the pinned public checkpoint when files or identity are incomplete."""
    marker = asset.path / SNAPSHOT_MARKER
    expected = {"repo_id": asset.repo_id, "revision": asset.revision}
    required = tuple(asset.path / path for path in _CHECKPOINT_FILES)
    if _json_matches(marker, expected) and all(
        _is_nonempty_file(path) for path in required
    ):
        return
    logger.info(
        "downloading ABot-World checkpoint %s revision=%s to %s",
        asset.repo_id,
        asset.revision,
        asset.path,
    )
    snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    asset.path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.repo_id,
        revision=asset.revision,
        local_dir=asset.path,
        cache_dir=weights_root / "huggingface",
    )
    unresolved = [str(path) for path in required if not _is_nonempty_file(path)]
    if unresolved:
        raise RuntimeError(
            f"ABot-World checkpoint download is incomplete: {unresolved}"
        )
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(temporary, marker)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _revision(value: object, name: str) -> str:
    """Return one immutable 40-character public revision."""
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _repo_id(value: object, name: str) -> str:
    """Return one public Hugging Face repository identifier."""
    repo_id = str(value or "")
    if "/" not in repo_id:
        raise ValueError(f"{name} must identify a public repository")
    return repo_id


def _repository_url(value: object, name: str) -> str:
    """Return one public HTTPS Git repository URL."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _source_path(value: object, weights_root: Path) -> Path:
    """Resolve the upstream checkout under the runtime's weights root."""
    configured = os.environ.get(SOURCE_ENV)
    raw = configured if configured else str(value or "")
    if not raw:
        raise ValueError("source.path must be non-empty")
    path = Path(raw).expanduser()
    candidate = path if path.is_absolute() else weights_root / path
    return Path(os.path.abspath(candidate))


def _weights_path(value: object, name: str, weights_root: Path) -> Path:
    """Resolve one model asset directory under the runtime's weights root."""
    raw = str(value or "")
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    path = Path(raw).expanduser()
    candidate = path if path.is_absolute() else weights_root / path
    return Path(os.path.abspath(candidate))


def _json_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a JSON completion marker matches an asset identity."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return document == expected


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a required asset exists and contains bytes."""
    return path.is_file() and path.stat().st_size > 0


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git and preserve public checkout failures."""
    command = ["git", *arguments]
    if len(arguments) >= 2 and arguments[0] == "-C":
        command = ["git", "-c", f"safe.directory={arguments[1]}", *arguments]
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Git is required to prepare the ABot-World source"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(
            f"Unable to prepare the ABot-World source: {detail}"
        ) from error
