"""Prepare pinned inference assets in the application, before worker startup."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from movieroll_model import Settings

SOURCE_REVISION = "0b5f73e2d7cad86826dcf8fc420991d789d88646"
REVISIONS = {
    "movieroll": "fe3150d9233e54c504c04b4b42e12ebbc76b7765",
    "vae": "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038",
    "llm": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "byt5": "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
    "glyph": "b51a8604c03c9a4e71a389f4c7deeb526320f2e6",
}


def checkout(path: Path) -> None:
    if not path.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/Tencent-Hunyuan/MovieRoll.git",
                str(path),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--detach", SOURCE_REVISION], check=True
    )


def _complete(root: Path) -> bool:
    marker = root / "movieroll-assets.json"
    if not marker.is_file():
        return False
    manifest = json.loads(marker.read_text())
    return (
        manifest.get("revisions") == REVISIONS
        and bool(manifest.get("files"))
        and all(
            (root / name).is_file() and (root / name).stat().st_size == size
            for name, size in manifest["files"].items()
        )
    )


def prepare_weights(root: Path) -> None:
    if _complete(root):
        return
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    # Request gated weights first so an unauthenticated launch fails before
    # downloading the large shared encoders. HF_TOKEN is read by the Hub client.
    snapshots = [
        (
            "yunyangge/MovieRoll",
            REVISIONS["movieroll"],
            ["distilled/t2v/*", "distilled/i2v/*", "LICENSE", "NOTICE"],
            root / "movieroll",
        ),
        ("tencent/HunyuanVideo-1.5", REVISIONS["vae"], ["vae/*"], root),
        (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            REVISIONS["llm"],
            ["*.json", "*.safetensors", "*.txt", "*.jinja", "merges.txt", "vocab.json"],
            root / "text_encoder/llm",
        ),
        (
            "google/byt5-small",
            REVISIONS["byt5"],
            [
                "config.json",
                "pytorch_model.bin",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
            root / "text_encoder/byt5-small",
        ),
    ]
    for repo, revision, patterns, destination in snapshots:
        snapshot_download(
            repo, revision=revision, allow_patterns=patterns, local_dir=destination
        )
    from modelscope import snapshot_download as modelscope_download

    glyph = root / "text_encoder/Glyph-SDXL-v2"
    modelscope_download(
        "AI-ModelScope/Glyph-SDXL-v2",
        revision=REVISIONS["glyph"],
        allow_file_pattern=[
            "assets/multilingual_10-lang_idx.json",
            "assets/color_idx.json",
            "checkpoints/byt5_model.pt",
        ],
        local_dir=str(glyph),
    )
    required = [
        root / "movieroll/distilled" / mode / file
        for mode in ("t2v", "i2v")
        for file in ("config.json", "model.safetensors")
    ]
    required += [
        root / "vae/config.json",
        root / "vae/diffusion_pytorch_model.safetensors",
        root / "text_encoder/llm/config.json",
        root / "text_encoder/byt5-small/pytorch_model.bin",
        glyph / "assets/multilingual_10-lang_idx.json",
        glyph / "assets/color_idx.json",
        glyph / "checkpoints/byt5_model.pt",
    ]
    if not all(file.is_file() for file in required):
        raise FileNotFoundError("Incomplete MovieRoll inference assets")
    files = {
        str(p.relative_to(root)): p.stat().st_size
        for p in root.rglob("*")
        if p.is_file()
        and ".cache" not in p.relative_to(root).parts
        and p.name != "movieroll-assets.json"
    }
    temporary = root / "movieroll-assets.json.tmp"
    temporary.write_text(
        json.dumps({"revisions": REVISIONS, "files": files}, indent=2) + "\n"
    )
    temporary.replace(root / "movieroll-assets.json")


def prepare(config_path: Path, weights: Path) -> Settings:
    config = yaml.safe_load(config_path.read_text()) or {}
    unknown = set(config) - {
        "source_path",
        "session_path",
        "history_gib",
        "gpu_count",
    }
    if unknown:
        raise ValueError(f"Unknown MovieRoll settings: {sorted(unknown)}")
    source = Path(config["source_path"]).expanduser()
    if not source.is_absolute():
        source = config_path.parent / source
    source = source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected MovieRoll source {SOURCE_REVISION}, got {revision}")
    history_gib = config.get("history_gib", 16)
    if type(history_gib) is not int or not 1 <= history_gib <= 64:
        raise ValueError("history_gib must be an integer from 1 to 64")
    count = os.environ.get("MOVIEROLL_GPU_COUNT")
    gpu_count = int(count) if count is not None else config.get("gpu_count", 1)
    if type(gpu_count) is not int or gpu_count not in (1, 4):
        raise ValueError("gpu_count must be 1 or 4")
    sessions = Path(
        os.environ.get(
            "MOVIEROLL_SESSION_PATH",
            config.get(
                "session_path", str(Path(tempfile.gettempdir()) / "movieroll-sessions")
            ),
        )
    ).expanduser()
    if not sessions.is_absolute():
        sessions = config_path.parent / sessions
    sessions.mkdir(parents=True, exist_ok=True)
    weights = weights.expanduser().resolve()
    prepare_weights(weights)
    return Settings(
        source=source,
        weights=weights,
        sessions=sessions.resolve(),
        history_bytes=history_gib * 1024**3,
        gpu_count=gpu_count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.checkout:
        checkout(args.checkout)
    if args.weights:
        prepare_weights(args.weights.expanduser().resolve())
    if not args.checkout and not args.weights:
        parser.error("Use --checkout PATH or --weights PATH")
