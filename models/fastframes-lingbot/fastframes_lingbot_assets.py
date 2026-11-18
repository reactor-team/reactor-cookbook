"""Resolve the pinned inference sources and selected Hugging Face assets."""

import argparse
import subprocess
from pathlib import Path

from fastframes_lingbot_model import Settings

APPLE_SOURCE = "https://github.com/apple-aiml-research/ml-fastframes.git"
APPLE_COMMIT = "6044a56f44bffec500d04f2c21be3c5066891431"
LINGBOT_SOURCE = "https://github.com/robbyant/lingbot-world-v2.git"
LINGBOT_COMMIT = "1895d300d8ac936401689b26389f51cbd36530eb"
APPLE = "apple/FastFrames-S", "aa099efcedf1b65447106bf9004cde75821f7387"
DIT = "robbyant/lingbot-world-v2-1.3b-causal-fast", "7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
SHARED = "robbyant/lingbot-world-v2-14b-causal-fast", "5c33dd40b213598c418fd25bff30fdbd23fd38a7"
APPLE_FILES = ("lingbot-1.3b/kv_refiner.safetensors", "lingbot-1.3b/pixel_decoder.safetensors")
DIT_FILES = ("model.safetensors.index.json",) + tuple(f"model-{i:05d}-of-00006.safetensors" for i in range(1, 7))
SHARED_FILES = (
    "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json", "google/umt5-xxl/special_tokens_map.json", "google/umt5-xxl/spiece.model",
)


def checkout(url: str, revision: str, destination: Path) -> None:
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", url, revision], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)


def snapshot(repo: str, revision: str, files: tuple[str, ...], cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    root = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if all((root / name).is_file() for name in files):
        return root
    root = Path(snapshot_download(repo, revision=revision, allow_patterns=list(files), cache_dir=cache))
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {repo} snapshot: {', '.join(missing)}")
    return root


def source_path(config_path: Path, value: str, revision: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Source paths must be non-empty strings")
    root = (config_path.parent / value).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"{root} must be checked out at {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    return root


def prepare(config_path: Path, weights: Path) -> Settings:
    import yaml

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or set(config) != {"fastframes_source", "lingbot_source"}:
        raise ValueError("fastframes_lingbot.yaml requires fastframes_source and lingbot_source")
    return Settings(
        fastframes_source=source_path(config_path, config["fastframes_source"], APPLE_COMMIT),
        lingbot_source=source_path(config_path, config["lingbot_source"], LINGBOT_COMMIT),
        networks=snapshot(*APPLE, APPLE_FILES, weights) / "lingbot-1.3b",
        dit=snapshot(*DIT, DIT_FILES, weights),
        shared=snapshot(*SHARED, SHARED_FILES, weights),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    checkout(APPLE_SOURCE, APPLE_COMMIT, args.checkout / "fastframes")
    checkout(LINGBOT_SOURCE, LINGBOT_COMMIT, args.checkout / "lingbot-world-v2")
