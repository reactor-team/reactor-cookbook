"""Validate SolarWM adapter settings and prepare pinned public assets."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class SolarWMConfig:
    """Hold validated paths and inference settings for SolarWM Stage2."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    repo_id: str
    checkpoint_revision: str
    base_path: Path
    checkpoint_path: Path
    runtime_root: Path
    seed: int
    default_prompt: str
    context_latents: int
    max_chunks: int
    translation_units_per_latent: float
    rotation_degrees_per_latent: float


def read_config(config_path: Path | None) -> SolarWMConfig:
    """Read the adapter YAML and reject settings that alter the native cache contract."""
    if config_path is None:
        raise ValueError("SolarWM requires runtime.config in reactor.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")
    source, assets = raw["source"], raw["assets"]
    stream, motion = raw["stream"], raw["motion"]
    source_path = Path(os.environ.get("SOLARWM_SOURCE_PATH", source["path"])).resolve()
    base_path = Path(assets["root"]).resolve() / "SolarWM-5B-base"
    checkpoint_path = Path(assets["root"]).resolve() / "SolarWM-5B-sgf-stage2-81f"
    context = int(stream["context_latents"])
    if context != 18:
        raise ValueError("SolarWM's native local_attn_size requires context_latents=18")
    max_chunks = int(stream["max_chunks"])
    if not 1 <= max_chunks <= 320:
        raise ValueError("stream.max_chunks must be between 1 and 320")
    return SolarWMConfig(
        source_path=source_path,
        source_url=str(source["url"]),
        source_revision=str(source["revision"]),
        upstream_config=source_path / str(source["config"]),
        repo_id=str(assets["repo_id"]),
        checkpoint_revision=str(assets["revision"]),
        base_path=base_path,
        checkpoint_path=checkpoint_path,
        runtime_root=Path(assets["root"]).resolve() / "runtime",
        seed=int(raw["inference"]["seed"]),
        default_prompt=str(raw["inference"]["default_prompt"]).strip(),
        context_latents=context,
        max_chunks=max_chunks,
        translation_units_per_latent=float(motion["translation_units_per_latent"]),
        rotation_degrees_per_latent=float(motion["rotation_degrees_per_latent"]),
    )


def prepare_runtime(config: SolarWMConfig) -> None:
    """Verify the pinned source and download only the Stage2 5B files in use."""
    if not (config.source_path / ".git").is_dir():
        config.source_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                config.source_url,
                str(config.source_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(config.source_path),
                "checkout",
                "--detach",
                config.source_revision,
            ],
            check=True,
        )
    revision = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={config.source_path}",
            "-C",
            str(config.source_path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != config.source_revision:
        raise RuntimeError(
            f"SolarWM source revision is {revision}; expected {config.source_revision}"
        )
    required = (
        "SolarWM-5B-base/**",
        "SolarWM-5B-sgf-stage2-81f/**",
    )
    if not (config.checkpoint_path / "model.pt").is_file():
        snapshot_download(
            repo_id=config.repo_id,
            revision=config.checkpoint_revision,
            local_dir=config.base_path.parent,
            allow_patterns=list(required),
            token=os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN"),
        )
    for path in (
        config.base_path / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
        config.base_path / "vae/Wan2.2_VAE.pth",
        config.checkpoint_path / "model.pt",
        config.checkpoint_path / "release-manifest.json",
    ):
        if not path.is_file():
            raise RuntimeError(f"SolarWM asset is missing: {path}")
    config.runtime_root.mkdir(parents=True, exist_ok=True)
