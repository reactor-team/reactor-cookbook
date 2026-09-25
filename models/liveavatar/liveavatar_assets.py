"""NVMe asset locations and pinned upstream setup for stage-one debugging."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path("/opt/dlami/nvme")
SOURCE_REVISION = "c3c47d031d8bf3247428333c1fe610579c71a551"
BASE_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
LORA_REVISION = "92cdccd12a91e8a63767a7c821b7c75e51d5a172"
SOURCE = Path(
    os.environ.get("LIVEAVATAR_SOURCE", ROOT / "ruixing/liveavatar-upstream-20260916")
)
LOCAL_WEIGHTS_ONLY = os.environ.get("LIVEAVATAR_LOCAL_WEIGHTS_ONLY") == "1"


def mounted_weights_path() -> Path:
    from reactor_runtime import get_weights_path

    return get_weights_path()


WORK = (
    mounted_weights_path() / ".runtime"
    if LOCAL_WEIGHTS_ONLY
    else ROOT / ".cache_hf/reactor_registry/liveavatar-stage1"
)


def configure_cache_environment() -> None:
    cache_root = WORK / "cache" if LOCAL_WEIGHTS_ONLY else ROOT / ".cache_hf"
    for name, value in {
        "UV_CACHE_DIR": WORK / "uv" if LOCAL_WEIGHTS_ONLY else ROOT / ".cache_uv",
        "UV_PYTHON_INSTALL_DIR": WORK / "python"
        if LOCAL_WEIGHTS_ONLY
        else ROOT / ".cache_uv/python",
        "HF_HOME": cache_root,
        "HF_HUB_CACHE": cache_root / "hub",
        "TRANSFORMERS_CACHE": cache_root / "hub",
        "XDG_CACHE_HOME": cache_root / "liveavatar-cache",
        "TORCH_HOME": cache_root / "torch",
        "TMPDIR": WORK / "tmp",
        "TORCHINDUCTOR_CACHE_DIR": WORK / "inductor",
        "CUTE_DSL_CACHE_DIR": WORK / "cute",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": WORK / "fa4",
    }.items():
        os.environ[name] = str(value)
        value.mkdir(parents=True, exist_ok=True)
    os.environ["ENABLE_COMPILE"] = "false"


def prepare_assets() -> tuple[Path, Path]:
    configure_cache_environment()
    if not SOURCE.exists():
        if LOCAL_WEIGHTS_ONLY:
            raise RuntimeError(
                "The container must include the pinned LiveAvatar source"
            )
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/Alibaba-Quark/LiveAvatar.git",
                str(SOURCE),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(SOURCE), "checkout", "--detach", SOURCE_REVISION],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"Expected upstream {SOURCE_REVISION}, found {revision}; use a separate checkout"
        )
    if LOCAL_WEIGHTS_ONLY:
        weights = mounted_weights_path()
        base, lora = weights / "wan2_2", weights / "liveavatar_lora"
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

    base = Path(snapshot_download("Wan-AI/Wan2.2-S2V-14B", revision=BASE_REVISION))
    lora = Path(snapshot_download("Quark-Vision/Live-Avatar", revision=LORA_REVISION))
    return base, lora


if __name__ == "__main__":
    print(prepare_assets())
