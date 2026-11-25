"""YUME source, checkpoint, cache, and inference configuration."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

# The checkpoint files that the model reads, at their repository paths.
REQUIRED_FILES = (
    "diffusion_pytorch_model.safetensors",
    "config.json",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/special_tokens_map.json",
)


@dataclass(frozen=True)
class YumeConfig:
    source_path: Path
    source_url: str
    source_revision: str
    checkpoint_path: Path
    checkpoint_repo: str
    checkpoint_revision: str
    cache_dir: Path
    width: int
    height: int
    frames_per_chunk: int
    latent_frames_per_chunk: int
    sample_steps: int
    shift: float
    seed: int
    default_upload_prompt: str
    max_chunks: int


def read_config(path: Path | None, weights_root: Path | None = None) -> YumeConfig:
    """Read and strictly validate the native YUME-5B rollout settings."""
    if path is None:
        raise ValueError("YUME requires yume.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    def local(value: object) -> Path:
        return _local_path(value, weights_root or path.parent)

    source, assets, inference = raw["source"], raw["assets"], raw["inference"]
    config = YumeConfig(
        source_path=_local_path(source["path"], path.parent),
        source_url=str(source["url"]),
        source_revision=str(source["revision"]),
        checkpoint_path=local(assets["checkpoint_path"]),
        checkpoint_repo=str(assets["checkpoint_repo"]),
        checkpoint_revision=str(assets["checkpoint_revision"]),
        cache_dir=local(assets["cache_dir"]),
        width=int(inference["width"]),
        height=int(inference["height"]),
        frames_per_chunk=int(inference["frames_per_chunk"]),
        latent_frames_per_chunk=int(inference["latent_frames_per_chunk"]),
        sample_steps=int(inference["sample_steps"]),
        shift=float(inference["shift"]),
        seed=int(inference["seed"]),
        default_upload_prompt=str(inference["default_upload_prompt"]).strip(),
        max_chunks=int(inference["max_chunks"]),
    )
    if (config.width, config.height) != (1280, 704):
        raise ValueError("YUME-5B public checkpoint uses 1280x704 generation")
    if config.frames_per_chunk != 32 or config.latent_frames_per_chunk != 8:
        raise ValueError(
            "YUME's native continuation window is 32 pixel / 8 latent frames"
        )
    if config.sample_steps <= 0 or config.max_chunks <= 0:
        raise ValueError("sample_steps and max_chunks must be positive")
    if not config.default_upload_prompt:
        raise ValueError("default_upload_prompt must be non-empty")
    return config


def configure_environment(config: YumeConfig) -> None:
    """Set deployment caches without overriding operator choices."""
    paths = {
        "HF_HOME": config.cache_dir,
        "HUGGINGFACE_HUB_CACHE": config.cache_dir / "hub",
        "TRANSFORMERS_CACHE": config.cache_dir / "transformers",
        "XDG_CACHE_HOME": config.cache_dir / "xdg",
        "TRITON_CACHE_DIR": config.cache_dir / "triton",
        "TORCHINDUCTOR_CACHE_DIR": config.cache_dir / "torchinductor",
        "CUDA_CACHE_PATH": config.cache_dir / "cuda",
    }
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(value))
    if os.environ.get("HF_KEY") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_KEY"]


def weights_ref(config: YumeConfig) -> str:
    """Return the ``hf://`` weight set that holds the pinned checkpoint."""
    return f"hf://{config.checkpoint_repo}@{config.checkpoint_revision}"


def check_assets(config: YumeConfig) -> None:
    """Check the source in the image and the checkpoint in the weight set.

    This function never downloads. The image holds the source at the pinned
    revision, and the weight set holds the checkpoint files at their
    repository paths.
    """
    if not (config.source_path / ".git").is_dir():
        raise FileNotFoundError(
            f"YUME source is missing at {config.source_path}; "
            "rebuild the image with `reactor build`"
        )
    actual = subprocess.run(
        ["git", "-C", str(config.source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"YUME source is {actual}; expected {config.source_revision}"
        )
    missing = [
        name
        for name in REQUIRED_FILES
        if not (config.checkpoint_path / name).is_file()
        or (config.checkpoint_path / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"YUME weights in {config.checkpoint_path} lack: {', '.join(missing)}. "
            f"Deploy or run with `--weights {weights_ref(config)}`."
        )


def _local_path(value: object, root: Path) -> Path:
    """Resolve one configured path under Reactor's mounted weights root."""
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else root / path).resolve()
