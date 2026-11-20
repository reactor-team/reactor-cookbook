"""Application-owned source, checkpoint preparation, and configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class OpenOasisConfig:
    source_path: str
    source_revision: str
    checkpoint_repo_id: str
    checkpoint_revision: str
    model_filename: str
    vae_filename: str
    seed: int
    ddim_steps: int
    context_frames: int


def read_config(path: Path | None) -> OpenOasisConfig:
    if path is None:
        raise ValueError("Open-Oasis requires open_oasis.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OpenOasisConfig(
        source_path=str(
            (path.parent / Path(str(raw["source"]["path"])).expanduser()).resolve()
        ),
        source_revision=str(raw["source"]["revision"]),
        checkpoint_repo_id=str(raw["checkpoint"]["repo_id"]),
        checkpoint_revision=str(raw["checkpoint"]["revision"]),
        model_filename=str(raw["checkpoint"]["model_filename"]),
        vae_filename=str(raw["checkpoint"]["vae_filename"]),
        seed=int(raw["inference"]["seed"]),
        ddim_steps=int(raw["inference"]["ddim_steps"]),
        context_frames=int(raw["inference"]["context_frames"]),
    )


def prepare_source(config: OpenOasisConfig) -> Path:
    root = Path(config.source_path)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/etched-ai/open-oasis.git", str(root)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", config.source_revision],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config.source_revision:
        raise RuntimeError(
            f"Open-Oasis source revision {revision} does not match pinned {config.source_revision}"
        )
    return root


def download_checkpoints(
    config: OpenOasisConfig, weights_root: Path
) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    cache = weights_root / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": config.checkpoint_repo_id,
        "revision": config.checkpoint_revision,
        "cache_dir": cache,
        "token": os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY"),
    }
    return (
        Path(hf_hub_download(filename=config.model_filename, **kwargs)),
        Path(hf_hub_download(filename=config.vae_filename, **kwargs)),
    )
