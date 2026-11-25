"""Configuration, and the checks that the image holds the source and the weight set holds the checkpoints."""

from __future__ import annotations

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


def weights_ref(config: OpenOasisConfig) -> str:
    """The pinned weight set: ``hf://<repo>@<revision>``."""
    return f"hf://{config.checkpoint_repo_id}@{config.checkpoint_revision}"


def check_source(config: OpenOasisConfig) -> Path:
    """Return the upstream checkout that the image build made, after checking its revision."""
    root = Path(config.source_path)
    if not (root / ".git").exists():
        raise FileNotFoundError(
            f"Open-Oasis source not found at {root}; rebuild the image with `reactor build`"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config.source_revision:
        raise RuntimeError(
            f"Open-Oasis source revision {revision} does not match pinned {config.source_revision}"
        )
    return root


def resolve_checkpoints(
    config: OpenOasisConfig, weights_root: Path
) -> tuple[Path, Path]:
    """Return the model and VAE checkpoints in the weight set; never download."""
    paths = (weights_root / config.model_filename, weights_root / config.vae_filename)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Open-Oasis checkpoint not found at {path}; "
                f"deploy with `--weights {weights_ref(config)}`"
            )
    return paths
