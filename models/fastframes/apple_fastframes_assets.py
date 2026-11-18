"""Resolve the pinned Apple inference code and only the checkpoints it reads."""

import argparse
import subprocess
from pathlib import Path

from apple_fastframes_model import Settings

SOURCE = "https://github.com/apple-aiml-research/ml-fastframes.git"
SOURCE_REVISION = "6044a56f44bffec500d04f2c21be3c5066891431"
APPLE_REVISIONS = {
    "ff-s": ("apple/FastFrames-S", "aa099efcedf1b65447106bf9004cde75821f7387"),
    "ff-l": ("apple/FastFrames-L", "18f30cbdd91943baa274e4237dd09b6d19e58ca9"),
}
WAN_REVISIONS = {
    "ff-s": ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a"),
    "ff-l": ("Wan-AI/Wan2.1-T2V-14B", "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"),
}
CF_REPO = "zhuhz22/Causal-Forcing"
CF_REVISION = "2f8eb8bb6eeb1238da9d13e5420d342a74d634a6"
CF_GENERATOR = "causal-forcing++/framewise-2step.pt"
WAN_FILES = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
)


def checkout(destination: Path) -> None:
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            SOURCE,
            SOURCE_REVISION,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )


def snapshot(repo: str, revision: str, files: tuple[str, ...], weights: Path) -> Path:
    from huggingface_hub import snapshot_download

    root = weights / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if all((root / name).is_file() for name in files):
        return root

    path = Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=list(files),
            cache_dir=weights,
        )
    )
    missing = [name for name in files if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {repo} snapshot: {', '.join(missing)}")
    return path


def prepare(config_path: Path, weights: Path) -> Settings:
    import yaml

    config = yaml.safe_load(config_path.read_text())
    if (
        not isinstance(config, dict)
        or not {"source_path"} <= set(config) <= {"source_path", "variant"}
        or not isinstance(config["source_path"], str)
        or not config["source_path"].strip()
    ):
        raise ValueError("apple_fastframes.yaml requires a non-empty source_path")
    variant = config.get("variant", "ff-s")
    if variant not in ("ff-s", "ff-l"):
        raise ValueError("FastFrames variant must be ff-s or ff-l")
    source = (config_path.parent / config["source_path"]).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOURCE_REVISION:
        raise ValueError(f"FastFrames source revision must be {SOURCE_REVISION}")
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "fastframes",
            "evaluation/latency",
        ],
        check=True,
    )
    size = "fastframes-s" if variant == "ff-s" else "fastframes-l"
    apple_files = (
        "LICENSE",
        f"{size}/kv_refiner.safetensors",
        f"{size}/pixel_decoder.safetensors",
    )
    if variant == "ff-l":
        apple_files += ("fastframes-l/dit.safetensors",)
    apple = snapshot(*APPLE_REVISIONS[variant], apple_files, weights)
    generator = (
        snapshot(CF_REPO, CF_REVISION, (CF_GENERATOR,), weights) / CF_GENERATOR
        if variant == "ff-s"
        else apple / "fastframes-l/dit.safetensors"
    )
    return Settings(
        variant=variant,
        source=source,
        apple=apple,
        wan=snapshot(*WAN_REVISIONS[variant], WAN_FILES, weights),
        generator=generator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    checkout(parser.parse_args().checkout)
