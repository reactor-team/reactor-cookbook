"""Application-owned preparation of pinned source and checkpoint paths."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from liveavatar_model import LiveAvatarSettings

SOURCE_REVISION = "c3c47d031d8bf3247428333c1fe610579c71a551"
BASE_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
LORA_REVISION = "92cdccd12a91e8a63767a7c821b7c75e51d5a172"


def configure_cache_environment(weights_root: Path) -> Path:
    """Keep generated caches under the mount, respecting operator overrides."""
    work = weights_root / ".runtime"
    for name, relative in {
        "HF_HOME": "cache",
        "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "torch",
        "TMPDIR": "tmp",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "CUTE_DSL_CACHE_DIR": "cute",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "fa4",
    }.items():
        path = Path(os.environ.setdefault(name, str(work / relative)))
        path.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return work


def prepare_assets(
    weights_root: Path, source: Path, *, local_only: bool
) -> tuple[Path, Path]:
    """Validate the build's source and prepare weights; never mutate a checkout."""
    if not source.is_dir():
        raise RuntimeError(
            "The serving image must include the pinned LiveAvatar source"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"Expected upstream {SOURCE_REVISION}, found {revision}")
    if local_only:
        base, lora = weights_root / "wan2_2", weights_root / "liveavatar_lora"
        required = [
            base / "config.json",
            base / "Wan2.1_VAE.pth",
            lora / "liveavatar.safetensors",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"Incomplete mounted weights; run prepare_weights.py: {missing}"
            )
        return base, lora

    from huggingface_hub import snapshot_download

    cache = weights_root / "huggingface"
    base = Path(
        snapshot_download(
            "Wan-AI/Wan2.2-S2V-14B", revision=BASE_REVISION, cache_dir=cache
        )
    )
    lora = Path(
        snapshot_download(
            "Quark-Vision/Live-Avatar", revision=LORA_REVISION, cache_dir=cache
        )
    )
    return base, lora


def prepare_settings(weights_root: Path) -> LiveAvatarSettings:
    """Resolve deployment choices once, before constructing the model."""
    source = (
        Path(os.environ.get("LIVEAVATAR_SOURCE", "/opt/liveavatar-source"))
        .expanduser()
        .resolve()
    )
    work = configure_cache_environment(weights_root)
    base, lora = prepare_assets(
        weights_root,
        source,
        local_only=os.environ.get("LIVEAVATAR_LOCAL_WEIGHTS_ONLY", "1") == "1",
    )
    if os.environ.get("LIVEAVATAR_STEPS", "4") != "4":
        raise ValueError("This serving recipe preserves the released four-step sampler")
    return LiveAvatarSettings(
        source=source,
        base=base,
        lora=lora,
        work=work,
        turbo=os.environ.get("LIVEAVATAR_TURBO", "1") == "1",
    )
