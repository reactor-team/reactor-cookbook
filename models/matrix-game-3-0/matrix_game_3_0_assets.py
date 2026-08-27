"""Prepare pinned public Matrix-Game 3.0 source and distilled model assets."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

SOURCE_ENV = "MATRIX_GAME_3_0_PATH"
SNAPSHOT_MARKER = ".reactor-snapshot.json"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODEL_ALLOW_PATTERNS = (
    "MG-LightVAE_v2.pth",
    "Wan2.2_VAE.pth",
    "base_distilled_model/*",
    "google/umt5-xxl/*",
    "models_t5_umt5-xxl-enc-bf16.pth",
)


@dataclass(frozen=True)
class ExampleScene:
    """Describe one public image and prompt available as a rollout anchor."""

    image: Path
    prompt: str


@dataclass(frozen=True)
class MatrixGame30Config:
    """Hold validated Matrix-Game 3.0 asset and inference settings."""

    source_path: Path
    source_url: str
    source_revision: str
    checkpoint_path: Path
    checkpoint_repo_id: str
    checkpoint_revision: str
    examples: tuple[ExampleScene, ...]
    size: str
    seed: int
    num_inference_steps: int
    sample_shift: float
    guide_scale: float
    use_int8: bool
    vae_type: str
    lightvae_pruning_rate: float
    max_chunks: int
    chunk_timeout_seconds: float


def read_config(config_path: Path | None) -> MatrixGame30Config:
    """Read and validate the Matrix-Game 3.0 adapter YAML."""
    if config_path is None:
        raise ValueError("Matrix-Game 3.0 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    checkpoint = _mapping(document.get("checkpoint"), "checkpoint")
    inference = _mapping(document.get("inference"), "inference")
    stream = _mapping(document.get("stream"), "stream")
    source_path = _source_path(source.get("path"))
    checkpoint_path = get_weights_path() / "checkpoints" / "Matrix-Game-3.0"

    examples_document = document.get("examples")
    if not isinstance(examples_document, list) or not examples_document:
        raise ValueError("examples must be a non-empty YAML list")
    examples = tuple(
        _example(config_path.parent, value, index)
        for index, value in enumerate(examples_document)
    )

    size = str(inference.get("size", "704*1280"))
    if not re.fullmatch(r"[1-9][0-9]*\*[1-9][0-9]*", size):
        raise ValueError("inference.size must use HEIGHT*WIDTH")
    num_inference_steps = int(inference.get("num_inference_steps", 3))
    if num_inference_steps <= 0:
        raise ValueError("inference.num_inference_steps must be positive")
    max_chunks = int(stream.get("max_chunks", 12))
    if max_chunks != 12:
        raise ValueError("stream.max_chunks must remain 12 to match upstream inference")
    timeout = float(stream.get("chunk_timeout_seconds", 1800.0))
    if timeout <= 0:
        raise ValueError("stream.chunk_timeout_seconds must be positive")
    vae_type = str(inference.get("vae_type", "mg_lightvae_v2"))
    if vae_type != "mg_lightvae_v2":
        raise ValueError(
            "inference.vae_type must be mg_lightvae_v2 for this fast recipe"
        )

    return MatrixGame30Config(
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        checkpoint_path=checkpoint_path,
        checkpoint_repo_id=_repo_id(checkpoint.get("repo_id"), "checkpoint.repo_id"),
        checkpoint_revision=_revision(
            checkpoint.get("revision"), "checkpoint.revision"
        ),
        examples=examples,
        size=size,
        seed=int(inference.get("seed", 42)),
        num_inference_steps=num_inference_steps,
        sample_shift=float(inference.get("sample_shift", 5.0)),
        guide_scale=float(inference.get("guide_scale", 5.0)),
        use_int8=bool(inference.get("use_int8", True)),
        vae_type=vae_type,
        lightvae_pruning_rate=float(inference.get("lightvae_pruning_rate", 0.75)),
        max_chunks=max_chunks,
        chunk_timeout_seconds=timeout,
    )


def prepare_assets(config: MatrixGame30Config) -> None:
    """Prepare and verify the unmodified source checkout and distilled weights."""
    ensure_source_checkout(config)
    ensure_checkpoint(config)


def ensure_source_checkout(config: MatrixGame30Config) -> None:
    """Clone the pinned public source when absent and verify its revision."""
    checkout = config.source_path
    if not checkout.exists():
        logger.info(
            "downloading Matrix-Game 3.0 source checkout",
            url=config.source_url,
            revision=config.source_revision,
            destination=str(checkout),
        )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".reactor-matrix-game-3-0-source-", dir=checkout.parent
        ) as temporary:
            candidate = Path(temporary) / "checkout"
            _run_git(
                [
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    config.source_url,
                    str(candidate),
                ]
            )
            _run_git(
                ["-C", str(candidate), "checkout", "--detach", config.source_revision]
            )
            with suppress(FileExistsError):
                candidate.rename(checkout)
    if not (checkout / ".git").is_dir():
        raise RuntimeError(f"Matrix source at {checkout} must be a Git checkout")
    actual = _run_git(["-C", str(checkout), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"Matrix source revision is {actual}; expected {config.source_revision}"
        )
    required = (
        checkout / "Matrix-Game-3" / "pipeline" / "inference_interactive_pipeline.py"
    )
    if not required.is_file():
        raise FileNotFoundError(f"Matrix interactive pipeline is missing: {required}")


def ensure_checkpoint(config: MatrixGame30Config) -> None:
    """Download and verify the pinned fast distilled checkpoint subset."""
    from huggingface_hub import snapshot_download

    expected = {
        "repo_id": config.checkpoint_repo_id,
        "revision": config.checkpoint_revision,
        "allow_patterns": list(_MODEL_ALLOW_PATTERNS),
    }
    marker = config.checkpoint_path / SNAPSHOT_MARKER
    required = (
        config.checkpoint_path
        / "base_distilled_model"
        / "diffusion_pytorch_model.safetensors",
        config.checkpoint_path / "MG-LightVAE_v2.pth",
        config.checkpoint_path / "Wan2.2_VAE.pth",
        config.checkpoint_path / "models_t5_umt5-xxl-enc-bf16.pth",
        config.checkpoint_path / "google" / "umt5-xxl" / "tokenizer.json",
    )
    if _json_matches(marker, expected) and all(path.is_file() for path in required):
        return
    config.checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloading Matrix-Game 3.0 distilled checkpoint",
        repo_id=config.checkpoint_repo_id,
        revision=config.checkpoint_revision,
        destination=str(config.checkpoint_path),
    )
    snapshot_download(
        repo_id=config.checkpoint_repo_id,
        revision=config.checkpoint_revision,
        local_dir=config.checkpoint_path,
        cache_dir=get_weights_path() / ".huggingface",
        allow_patterns=list(_MODEL_ALLOW_PATTERNS),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Matrix checkpoint download did not produce required files: "
            + ", ".join(missing)
        )
    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(pending, marker)


def _example(root: Path, value: object, index: int) -> ExampleScene:
    """Return one validated built-in scene from the YAML document."""
    document = _mapping(value, f"examples[{index}]")
    image = (root / str(document.get("image", ""))).resolve()
    if not image.is_file():
        raise FileNotFoundError(f"examples[{index}].image does not exist: {image}")
    prompt = str(document.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"examples[{index}].prompt must be non-empty")
    return ExampleScene(image=image, prompt=prompt)


def _source_path(value: object) -> Path:
    """Resolve the source checkout under the Runtime weights directory."""
    configured = os.environ.get(SOURCE_ENV)
    path = Path(configured if configured else str(value or "Matrix-Game")).expanduser()
    candidate = path if path.is_absolute() else get_weights_path() / path
    return Path(os.path.abspath(candidate))


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
    """Return one public HTTPS repository URL."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _repo_id(value: object, name: str) -> str:
    """Return one public Hugging Face repository identifier."""
    repo_id = str(value or "")
    if "/" not in repo_id:
        raise ValueError(f"{name} must identify a public repository")
    return repo_id


def _json_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a JSON completion marker matches the expected identity."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(document == expected)


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git and report a public-resource preparation error."""
    try:
        return subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to prepare Matrix-Game 3.0") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(
            f"Unable to prepare Matrix-Game 3.0 source: {detail}"
        ) from error
