"""Application-owned paths and cache preparation for the pinned Lyra image."""

import os
from pathlib import Path

import yaml


def prepare_config(config_path: Path, weights_root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source_path"]).expanduser()
    config["source_path"] = str((config_path.parent / source).resolve())
    for key in ("cache_path", "output_path", "checkpoints_path"):
        config[key] = str((weights_root / Path(config[key]).expanduser()).resolve())
    for key in ("cache_path", "output_path"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_path"])
    for key, value in {
        "HF_HOME": cache,
        "HUGGINGFACE_HUB_CACHE": cache / "hub",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache / "xdg",
    }.items():
        os.environ.setdefault(key, str(value))
    checkpoints = Path(config["checkpoints_path"])
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"Prepare the public Lyra checkpoints at {checkpoints}")
    link = Path(config["source_path"]) / "checkpoints"
    if link.exists() or link.is_symlink():
        if link.resolve() != checkpoints:
            raise ValueError(
                f"Source checkpoint location {link} does not match {checkpoints}"
            )
    else:
        link.symlink_to(checkpoints, target_is_directory=True)
    return config
