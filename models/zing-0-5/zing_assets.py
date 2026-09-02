"""Resolve Zing's pinned source, checkpoint, and NVMe cache paths."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from reactor_runtime import get_weights_path


@dataclass(frozen=True)
class ZingAdapterConfig:
    source_path: Path
    source_url: str
    source_revision: str
    repo_id: str
    asset_revision: str
    asset_path: Path
    width: int
    height: int
    seed: int
    max_chunks: int
    local_attn_size: int
    sink_size: int
    default_prompt: str
    example_prompt: str


def read_config(path: Path | None) -> ZingAdapterConfig:
    if path is None:
        raise ValueError("Zing requires zing.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, assets, inference = raw["source"], raw["assets"], raw["inference"]
    result = ZingAdapterConfig(
        source_path=Path(source["path"]), source_url=str(source["url"]),
        source_revision=str(source["revision"]), repo_id=str(assets["repo_id"]),
        asset_revision=str(assets["revision"]),
        asset_path=(get_weights_path() / Path(assets["path"])).resolve(),
        width=int(inference["width"]), height=int(inference["height"]),
        seed=int(inference["seed"]), max_chunks=int(inference["max_chunks"]),
        local_attn_size=int(inference["local_attn_size"]), sink_size=int(inference["sink_size"]),
        default_prompt=str(inference["default_prompt"]).strip(),
        example_prompt=str(inference["example_prompt"]).strip(),
    )
    if result.width % 32 or result.height % 32:
        raise ValueError("width and height must be divisible by 32")
    if (result.local_attn_size, result.sink_size) != (97, 9):
        raise ValueError("stage-one adapter preserves Zing's released 97/9 cache window")
    return result


def configure_environment(config: ZingAdapterConfig) -> None:
    cache = config.asset_path
    runtime = config.asset_path / "runtime-cache"
    values = {
        "HF_HOME": cache, "HUGGINGFACE_HUB_CACHE": cache / "hub",
        "XDG_CACHE_HOME": runtime, "TORCHINDUCTOR_CACHE_DIR": runtime / "torchinductor",
        "CUDA_CACHE_PATH": runtime / "cuda",
    }
    for name, value in values.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(value))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def prepare_assets(config: ZingAdapterConfig) -> None:
    if not (config.source_path / ".git").is_dir():
        config.source_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", config.source_url, str(config.source_path)], check=True)
        subprocess.run(
            ["git", "-C", str(config.source_path), "checkout", "--detach", config.source_revision],
            check=True,
        )
    actual = subprocess.run(
        ["git", "-C", str(config.source_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(f"Zing source is {actual}; expected {config.source_revision}")
    required = config.asset_path / "generator" / "model.pt"
    if not required.is_file():
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
        snapshot_download(
            repo_id=config.repo_id, revision=config.asset_revision,
            local_dir=config.asset_path, token=token,
        )


def activate_source(config: ZingAdapterConfig) -> None:
    source = str(config.source_path / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
