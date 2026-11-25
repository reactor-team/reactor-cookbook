"""Check the pinned source and the weights folder for SGF+. Nothing here downloads."""

import subprocess
from pathlib import Path

import yaml
from reactor_runtime import get_weights_path

from sgf_plus_model import Settings

SOURCE_REVISION = "14cda9bb35f87000fbc11138a5170002d213effe"
BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
BASE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
BASE_DIR = "wan_models/Wan2.1-T2V-1.3B"
BASE_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
SGF_REPO = "ZihanSu/Self_Gradient_Forcing_Plus"
SGF_REVISION = "c11e68083d8904226ac18ad7874bd300649f7679"
SGF_DIR = "hf_weights"
SGF_FILES = ("chunkwise/model.pt",)
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"


def check_source(source: Path) -> None:
    """Require the upstream checkout in the image at the pinned revision."""
    if not (source / ".git").exists():
        raise FileNotFoundError(
            f"SGF+ source is missing at {source}; rebuild the image with `reactor build`"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected upstream source {SOURCE_REVISION}, got {revision}")


def check_weights(weights: Path) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    required = [weights / BASE_DIR / name for name in BASE_FILES] + [
        weights / SGF_DIR / name for name in SGF_FILES
    ]
    missing = [
        str(path) for path in required if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"SGF+ weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def prepare(config_path: Path, weights_root: Path | None = None) -> Settings:
    """Check the source and weights, then return the worker settings. Never downloads."""
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source_path"]).expanduser().resolve()
    check_source(source)
    weights = (weights_root or get_weights_path()).resolve()
    check_weights(weights)
    return Settings(
        source=source,
        weights=weights,
        output_latents=int(config.get("output_latents", 963)),
    )
