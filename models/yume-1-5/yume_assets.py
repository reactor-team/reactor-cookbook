"""YUME source, checkpoint, cache, and inference configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from reactor_runtime import get_weights_path


@dataclass(frozen=True)
class YumeConfig:
    source_path: Path
    source_url: str
    source_revision: str
    checkpoint_path: Path
    checkpoint_repo: str
    checkpoint_revision: str
    cache_dir: Path
    runtime_dir: Path
    width: int
    height: int
    frames_per_chunk: int
    latent_frames_per_chunk: int
    sample_steps: int
    shift: float
    seed: int
    default_upload_prompt: str
    warmup_chunks: int


def read_config(path: Path | None) -> YumeConfig:
    """Read and strictly validate the native YUME-5B rollout settings."""
    if path is None:
        raise ValueError("YUME requires yume.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, assets, inference = raw["source"], raw["assets"], raw["inference"]
    config = YumeConfig(
        source_path=_local_path(source["path"]),
        source_url=str(source["url"]),
        source_revision=str(source["revision"]),
        checkpoint_path=_local_path(assets["checkpoint_path"]),
        checkpoint_repo=str(assets["checkpoint_repo"]),
        checkpoint_revision=str(assets["checkpoint_revision"]),
        cache_dir=_local_path(assets["cache_dir"]),
        runtime_dir=_local_path(assets["runtime_dir"]),
        width=int(inference["width"]),
        height=int(inference["height"]),
        frames_per_chunk=int(inference["frames_per_chunk"]),
        latent_frames_per_chunk=int(inference["latent_frames_per_chunk"]),
        sample_steps=int(inference["sample_steps"]),
        shift=float(inference["shift"]),
        seed=int(inference["seed"]),
        default_upload_prompt=str(inference["default_upload_prompt"]).strip(),
        warmup_chunks=int(inference["warmup_chunks"]),
    )
    if (config.width, config.height) != (1280, 704):
        raise ValueError("YUME-5B public checkpoint uses 1280x704 generation")
    if config.frames_per_chunk != 32 or config.latent_frames_per_chunk != 8:
        raise ValueError(
            "YUME's native continuation window is 32 pixel / 8 latent frames"
        )
    if config.sample_steps <= 0 or config.warmup_chunks < 0:
        raise ValueError("sample_steps must be positive and warmup_chunks non-negative")
    if not config.default_upload_prompt:
        raise ValueError("default_upload_prompt must be non-empty")
    return config


def configure_environment(config: YumeConfig) -> None:
    """Keep every mutable model/tool cache on the NVMe volume."""
    paths = {
        "HF_HOME": config.cache_dir,
        "HUGGINGFACE_HUB_CACHE": config.cache_dir / "hub",
        "TRANSFORMERS_CACHE": config.cache_dir / "transformers",
        "XDG_CACHE_HOME": config.runtime_dir / "xdg",
        "TRITON_CACHE_DIR": config.runtime_dir / "triton",
        "TORCHINDUCTOR_CACHE_DIR": config.runtime_dir / "torchinductor",
        "CUDA_CACHE_PATH": config.runtime_dir / "cuda",
    }
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(value))
    if os.environ.get("HF_KEY") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_KEY"]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def prepare_assets(config: YumeConfig) -> None:
    """Ensure immutable upstream source and checkpoint snapshots exist."""
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    if not (config.source_path / ".git").is_dir():
        config.source_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", config.source_url, str(config.source_path)], check=True
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
    required = config.checkpoint_path / "diffusion_pytorch_model.safetensors"
    if not required.is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=config.checkpoint_repo,
            revision=config.checkpoint_revision,
            local_dir=config.checkpoint_path,
            cache_dir=config.cache_dir,
        )


def activate_source(config: YumeConfig) -> None:
    """Import the pinned upstream checkout without modifying it."""
    root = str(config.source_path.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _local_path(value: object) -> Path:
    """Resolve one configured path under Reactor's mounted weights root."""
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else get_weights_path() / path).resolve()
