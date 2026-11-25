"""Resolve Zing's pinned source, checkpoint, and persistent cache paths."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

# The checkpoint files that the model reads, at their repository paths.
REQUIRED_FILES = (
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
    """Point caches and compiled kernels at ``~/.cache/zing-0-5``, not the weights."""
    cache = Path.home() / ".cache" / "zing-0-5"
    values = {
        "HF_HOME": cache,
        "HUGGINGFACE_HUB_CACHE": cache / "hub",
        "XDG_CACHE_HOME": cache / "runtime-cache",
        "TORCHINDUCTOR_CACHE_DIR": cache / "runtime-cache" / "torchinductor",
        "CUDA_CACHE_PATH": cache / "runtime-cache" / "cuda",
    }
    for name, value in values.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(value))


def weights_ref(config: ZingAdapterConfig) -> str:
    """Return the ``hf://`` weight set that holds the pinned checkpoint."""
    return f"hf://{config.repo_id}@{config.asset_revision}"


def check_assets(config: ZingAdapterConfig) -> None:
    """Check the source in the image and the checkpoint in the weight set.

    This function never downloads. The image holds the source at the pinned
    revision, and the weight set holds the checkpoint at its repository paths.
    """
    if not (config.source_path / ".git").is_dir():
        raise FileNotFoundError(
            f"Zing source is missing at {config.source_path}; "
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
            f"Zing source is {actual}; expected {config.source_revision}"
        )
    missing = [
        name
        for name in REQUIRED_FILES
        if not (config.asset_path / name).is_file()
        or (config.asset_path / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Zing weights in {config.asset_path} lack: {', '.join(missing)}. "
            f"Deploy or run with `--weights {weights_ref(config)}`."
        )
