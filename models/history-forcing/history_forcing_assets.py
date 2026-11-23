"""Resolve the pinned source and inference-only weights before worker startup."""

import json
import os
import subprocess
from pathlib import Path

import yaml

from history_forcing_model import Settings

SOURCE_REVISION = "9d69c1e5da2beb219e635f4400b8cb494759ba06"
BASE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
CHECKPOINT_REVISION = "8604aafe4fada0311c7f5a993e7e620473c8e7ac"
PROVENANCE = {"base": BASE_REVISION, "checkpoint": CHECKPOINT_REVISION}
BASE_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)


def prepare(config_path: Path, weights: Path) -> Settings:
    config = yaml.safe_load(config_path.read_text()) or {}
    source = Path(config["source_path"]).expanduser()
    if not source.is_absolute():
        source = config_path.parent / source
    source = source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"Expected History-Forcing source {SOURCE_REVISION}, got {revision}"
        )
    weights = weights.resolve()
    prepare_weights(weights)
    return Settings(
        source=source,
        model_root=weights / "models",
        checkpoint=weights / "models/History-Forcing/dmd/model.pt",
    )


def prepare_weights(weights: Path) -> None:
    marker = weights / "history-forcing-assets.json"
    base = weights / "models/Wan2.1-T2V-1.3B"
    checkpoint = weights / "models/History-Forcing/dmd/model.pt"
    required = [
        base / n
        for n in (
            *BASE_FILES,
            "google/umt5-xxl/tokenizer_config.json",
            "google/umt5-xxl/spiece.model",
        )
    ]
    if (
        marker.is_file()
        and json.loads(marker.read_text()) == PROVENANCE
        and checkpoint.is_file()
        and all(p.is_file() for p in required)
    ):
        return
    from huggingface_hub import snapshot_download

    base_snapshot = Path(
        snapshot_download(
            "Wan-AI/Wan2.1-T2V-1.3B",
            revision=BASE_REVISION,
            allow_patterns=[*BASE_FILES, "google/umt5-xxl/*"],
            cache_dir=weights / "hub",
        )
    )
    snapshot = Path(
        snapshot_download(
            "xzxxntxdy/History-Forcing",
            revision=CHECKPOINT_REVISION,
            allow_patterns=["dmd/model.pt"],
            cache_dir=weights / "hub",
        )
    )
    for link, target in ((base, base_snapshot), (checkpoint.parent.parent, snapshot)):
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"Asset destination already exists: {link}")
        # Relative targets survive mounting this cache at Runtime's weights path.
        link.symlink_to(
            os.path.relpath(target.resolve(), link.parent), target_is_directory=True
        )
    marker.write_text(json.dumps(PROVENANCE, indent=2) + "\n")
