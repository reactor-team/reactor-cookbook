"""Resolve pinned source, weights and trained text connectors before worker startup."""

import socket
import subprocess
import sys
from pathlib import Path

import yaml

from inworld_model import InWorldSettings

SOURCE_SHA = "4f57b6fd56b3d12624aebe95a1e3c747c8074908"
BASE_SHA = "a6de4b5354f078db24d9cf4778c14846788aea3d"
CHECKPOINT_SHA = "6f432abdd9fc0c2927122edf7c305c9e0e15c2a5"


def prepare_connectors(
    source: Path, checkpoint: Path, base: Path, output: Path
) -> None:
    """Reuse the pinned validator without taking its materialization write lock."""
    script = source / "inference/scripts/materialize_connectors.py"
    verify = """
import importlib.util, sys
from pathlib import Path
script, checkpoint, base, output = (Path(value).resolve() for value in sys.argv[1:])
spec = importlib.util.spec_from_file_location("connector_assets", script)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
valid = native.valid_existing(output, checkpoint, native.CHECKPOINT_SHA256, base,
    native.sha256(base / "diffusion_pytorch_model.safetensors.index.json"), verify_hashes=True)
if valid:
    manifest = native.load_json(output / "connector-overlay-manifest.json")
    if manifest.get("checkpoint_fingerprint") != native.fingerprint(checkpoint):
        valid = native.sha256(checkpoint) == native.CHECKPOINT_SHA256
sys.exit(0 if valid else 1)
"""
    verified = subprocess.run(
        [
            sys.executable,
            "-c",
            verify,
            str(script),
            str(checkpoint),
            str(base),
            str(output),
        ],
        check=False,
    )
    if verified.returncode == 0:
        return
    if verified.returncode != 1:
        verified.check_returncode()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--base-dir",
            str(base),
            "--output-dir",
            str(output),
            "--verify-existing-hashes",
        ],
        check=True,
    )


def prepare_assets(weights_root: Path, config_path: Path | None) -> InWorldSettings:
    config_path = config_path or Path(__file__).with_name("inworld.yaml")
    config = yaml.safe_load(config_path.read_text()) or {}
    attention_backend = config.get("attention_backend", "fa")
    num_gpus = config.get("num_gpus", 1)
    cuda_graph = config.get("cuda_graph", True)
    if type(num_gpus) is not int or num_gpus not in (1, 2, 4):
        raise ValueError("num_gpus must be 1, 2 or 4")
    if type(cuda_graph) is not bool:
        raise ValueError("cuda_graph must be a boolean")
    if attention_backend not in ("torch_sdpa", "fa"):
        raise ValueError(
            "attention_backend must be torch_sdpa or fa (FlashAttention 4)"
        )
    from huggingface_hub import hf_hub_download, snapshot_download

    source = Path(config.get("source", "upstream")).expanduser()
    if not source.is_absolute():
        source = config_path.parent / source
    source = source.resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_SHA:
        raise ValueError(f"InWorld source must be pinned to {SOURCE_SHA}")
    weights_root = Path(weights_root).expanduser().resolve()
    weights_root.mkdir(parents=True, exist_ok=True)
    cache = weights_root / "hub"
    base = Path(
        snapshot_download(
            "Lightricks/LTX-2.5-Diffusers",
            revision=BASE_SHA,
            cache_dir=cache,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "text_encoder/*",
                "connectors/config.json",
                "connectors/*index.json",
                "connectors/*-of-00002.safetensors",
                "vae/*",
                "audio_vae/*",
                "vocoder/*",
                "duration_head/*",
                "diffusion_decoder/*",
                "transformer/config.json",
                "transformer/*index.json",
                "transformer/*-of-00004.safetensors",
            ],
        )
    )
    checkpoint = Path(
        hf_hub_download(
            "MureaLabs/InWorld",
            "murea-fast.safetensors",
            revision=CHECKPOINT_SHA,
            cache_dir=cache,
        )
    )
    connectors = weights_root / "connectors-murea-fast"
    prepare_connectors(source, checkpoint, base / "connectors", connectors)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return InWorldSettings(
        source=source,
        base=base,
        checkpoint=checkpoint,
        connectors=connectors,
        master_port=port,
        attention_backend=attention_backend,
        num_gpus=num_gpus,
        cuda_graph=cuda_graph,
    )
