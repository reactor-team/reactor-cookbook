"""Resolve Zing's pinned source, checkpoint, and persistent cache paths."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ZingAdapterConfig:
    source_path: Path
    source_revision: str
    repo_id: str
    asset_revision: str
    asset_path: Path
    width: int
    height: int
    seed: int
    max_chunks: int
    default_prompt: str
    example_prompt: str


def read_config(path: Path | None, weights_root: Path) -> ZingAdapterConfig:
    if path is None:
        raise ValueError("Zing requires zing.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, assets, inference = raw["source"], raw["assets"], raw["inference"]
    result = ZingAdapterConfig(
        source_path=(path.parent / Path(source["path"]).expanduser()).resolve(),
        source_revision=str(source["revision"]),
        repo_id=str(assets["repo_id"]),
        asset_revision=str(assets["revision"]),
        asset_path=(weights_root / Path(assets["path"]).expanduser()).resolve(),
        width=int(inference["width"]),
        height=int(inference["height"]),
        seed=int(inference["seed"]),
        max_chunks=int(inference["max_chunks"]),
        default_prompt=str(inference["default_prompt"]).strip(),
        example_prompt=str(inference["example_prompt"]).strip(),
    )
    if result.width % 32 or result.height % 32:
        raise ValueError("width and height must be divisible by 32")
    if result.max_chunks < 1:
        raise ValueError("max_chunks must be positive")
    return result


def configure_environment(config: ZingAdapterConfig) -> None:
    cache = config.asset_path
    runtime = config.asset_path / "runtime-cache"
    values = {
        "HF_HOME": cache,
        "HUGGINGFACE_HUB_CACHE": cache / "hub",
        "XDG_CACHE_HOME": runtime,
        "TORCHINDUCTOR_CACHE_DIR": runtime / "torchinductor",
        "CUDA_CACHE_PATH": runtime / "cuda",
    }
    for name, value in values.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(value))


def prepare_assets(config: ZingAdapterConfig) -> None:
    if not (config.source_path / ".git").is_dir():
        raise RuntimeError("The serving image must include the pinned Zing checkout")
    actual = subprocess.run(
        ["git", "-C", str(config.source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(
            f"Zing source is {actual}; expected {config.source_revision}"
        )
    required = tuple(
        config.asset_path / name
        for name in (
            "generator/model.pt",
            "pretrained/vae/config.json",
            "pretrained/vae/diffusion_pytorch_model.safetensors",
            "pretrained/text_encoder/config.json",
            "pretrained/text_encoder/model.safetensors.index.json",
            "pretrained/text_encoder/model-00001-of-00003.safetensors",
            "pretrained/text_encoder/model-00002-of-00003.safetensors",
            "pretrained/text_encoder/model-00003-of-00003.safetensors",
            "pretrained/tokenizer/tokenizer_config.json",
            "pretrained/tokenizer/tokenizer.json",
            "pretrained/tokenizer/spiece.model",
            "pretrained/tokenizer/special_tokens_map.json",
        )
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        from huggingface_hub import snapshot_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
        snapshot_download(
            repo_id=config.repo_id,
            revision=config.asset_revision,
            local_dir=config.asset_path,
            token=token,
        )
    # Hub metadata can consider an existing empty file current; force only those files.
    empty = [
        path.relative_to(config.asset_path).as_posix()
        for path in required
        if path.is_file() and path.stat().st_size == 0
    ]
    if empty:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=config.repo_id,
            revision=config.asset_revision,
            local_dir=config.asset_path,
            allow_patterns=empty,
            force_download=True,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY"),
        )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Zing asset is missing or empty: {path}")
