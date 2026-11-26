"""Read SolarWM adapter settings and check the image's source and the weights folder."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

PREPARE_COMMAND = "HF_TOKEN=<your token> python prepare_weights.py --output ./weights"
BASE_FOLDER = "SolarWM-5B-base"
CHECKPOINT_FOLDER = "SolarWM-5B-sgf-stage2-81f"
# The files of the pinned repository that Stage2 inference reads. Stage2 loads
# the whole transformer from model.pt, so the base transformer shards are skipped.
ALLOW_PATTERNS = (
    f"{BASE_FOLDER}/LICENSE.txt",
    f"{BASE_FOLDER}/text_encoder/*",
    f"{BASE_FOLDER}/tokenizer/*",
    f"{BASE_FOLDER}/vae/*",
    f"{CHECKPOINT_FOLDER}/*",
)
REQUIRED_FILES = (
    f"{BASE_FOLDER}/text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
    f"{BASE_FOLDER}/vae/Wan2.2_VAE.pth",
    f"{BASE_FOLDER}/tokenizer/tokenizer.json",
    f"{BASE_FOLDER}/tokenizer/tokenizer_config.json",
    f"{BASE_FOLDER}/tokenizer/spiece.model",
    f"{BASE_FOLDER}/tokenizer/special_tokens_map.json",
    f"{CHECKPOINT_FOLDER}/model.pt",
    f"{CHECKPOINT_FOLDER}/COMPLETE.json",
    f"{CHECKPOINT_FOLDER}/checkpoint-manifest.json",
    f"{CHECKPOINT_FOLDER}/release-manifest.json",
)
_RUNTIME_ROOT = Path("~/.cache/solarwm-stage2")


@dataclass(frozen=True)
class SolarWMConfig:
    """Hold validated paths and inference settings for SolarWM Stage2."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    repo_id: str
    checkpoint_revision: str
    weights_root: Path
    base_path: Path
    checkpoint_path: Path
    runtime_root: Path
    seed: int
    default_prompt: str
    max_chunks: int
    translation_units_per_latent: float
    rotation_degrees_per_latent: float


def read_config(config_path: Path | None, weights_root: Path) -> SolarWMConfig:
    """Read the adapter YAML and reject settings that alter the native cache contract."""
    if config_path is None:
        raise ValueError("SolarWM requires runtime.config in reactor.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")
    source, assets = raw["source"], raw["assets"]
    stream, motion = raw["stream"], raw["motion"]

    source_path = Path(source["path"]).expanduser()
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    source_path = source_path.resolve()
    weights_root = weights_root.expanduser().resolve()
    max_chunks = int(stream["max_chunks"])
    if not 1 <= max_chunks <= 320:
        raise ValueError("stream.max_chunks must be between 1 and 320")
    return SolarWMConfig(
        source_path=source_path,
        source_url=str(source["url"]),
        source_revision=str(source["revision"]),
        upstream_config=(source_path / Path(source["config"]).expanduser()).resolve(),
        repo_id=str(assets["repo_id"]),
        checkpoint_revision=str(assets["revision"]),
        weights_root=weights_root,
        base_path=weights_root / BASE_FOLDER,
        checkpoint_path=weights_root / CHECKPOINT_FOLDER,
        runtime_root=_RUNTIME_ROOT.expanduser(),
        seed=int(raw["inference"]["seed"]),
        default_prompt=str(raw["inference"]["default_prompt"]).strip(),
        max_chunks=max_chunks,
        translation_units_per_latent=float(motion["translation_units_per_latent"]),
        rotation_degrees_per_latent=float(motion["rotation_degrees_per_latent"]),
    )


def check_runtime(config: SolarWMConfig) -> None:
    """Check the source and the weights folder, then create the scratch folder.

    Never downloads. The image holds the source at the pinned revision, and
    ``prepare_weights.py`` builds the weights folder.
    """
    check_source(config)
    check_weights(config)
    config.runtime_root.mkdir(parents=True, exist_ok=True)


def check_source(config: SolarWMConfig) -> None:
    """Raise unless the image holds the upstream checkout at the pinned revision."""
    if not (config.source_path / ".git").is_dir():
        raise FileNotFoundError(
            f"SolarWM source not found at {config.source_path}; "
            "rebuild the image with `reactor build`"
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


def check_weights(config: SolarWMConfig) -> None:
    """Raise when the weights folder lacks, or holds an empty copy of, a file Stage2 reads."""
    root = config.weights_root
    missing = [
        name
        for name in REQUIRED_FILES
        if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"SolarWM weights in {root} lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )
