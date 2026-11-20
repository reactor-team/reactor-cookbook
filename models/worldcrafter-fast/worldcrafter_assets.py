"""Pinned source/checkpoint identity and validated deployment configuration."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import yaml

SOURCE_REVISION = "11712fe3a3f42cc448c10713a0b0ff79066e5935"
MODEL_REVISION = "51320c3eb58a6313ccd6021a1da6d723d53e1a58"
WEIGHT_MANIFEST = Path(__file__).with_name("weights-manifest.json")
# GeoCalib's pinhole checkpoint, staged by reactor.yaml's build.run at the path
# torch.hub reads, so calibration never downloads it.
GEOCALIB_FILE = Path("hub/geocalib/pinhole.tar")
GEOCALIB_SIZE = 116_074_121
GEOCALIB_SHA256 = "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    python: Path
    source: Path
    weights: Path
    scratch: Path
    torch_home: Path
    gpu_count: int = 8
    compile: bool = True
    max_chunks: int = 50
    warmup_chunks: int = 2
    timeout: float = 1800


def read_settings(path: Path | None, weights_path: Path | None = None) -> Settings:
    data = yaml.safe_load(path.read_text()) if path is not None else {}
    root = Path(os.environ.get("WORLDCRAFTER_ROOT", "/local-ssd/worldcrafter-reactor"))
    result = Settings(
        python=Path(os.environ.get("WORLDCRAFTER_PYTHON", str(root / "venv/bin/python"))),
        source=Path(os.environ.get("WORLDCRAFTER_SOURCE", str(root / "source"))),
        weights=Path(os.environ.get("WORLDCRAFTER_WEIGHTS", str(weights_path if weights_path is not None else root / "weights"))),
        scratch=Path(os.environ.get("WORLDCRAFTER_SCRATCH", str(root / "scratch"))),
        torch_home=Path(os.environ.get("TORCH_HOME", "/opt/worldcrafter/torch")),
        gpu_count=int(os.environ.get("WORLDCRAFTER_GPUS", data.get("gpu_count", 8))),
        compile=os.environ.get("WORLDCRAFTER_COMPILE", str(int(data.get("compile", True)))) == "1",
        max_chunks=int(data.get("max_chunks", 50)),
        warmup_chunks=int(data.get("warmup_chunks", 2)),
        timeout=float(data.get("timeout_seconds", 1800)),
    )
    if result.gpu_count not in (1, 8):
        raise ValueError("This adapter supports one or eight GPUs")
    if not 1 <= result.max_chunks <= 50 or not 1 <= result.timeout <= 3600:
        raise ValueError("Invalid chunk limit or worker startup timeout")
    if not 0 <= result.warmup_chunks <= 4:
        raise ValueError("Invalid warmup chunk count")
    result.scratch.mkdir(parents=True, exist_ok=True)
    return result


def prepare_runtime(settings: Settings):
    """Validate the image environment, the staged GeoCalib checkpoint and the mounted weights."""
    for path in (settings.python, settings.source / "demo/engine.py"):
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared asset: {path}; run bash prepare.sh before starting the Runtime")
    marker = settings.source / ".reactor-source.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("revision") != SOURCE_REVISION:
        raise ValueError("Source must be the pinned snapshot prepared by prepare.sh")
    verify_geocalib(settings.torch_home / GEOCALIB_FILE)
    verify_weights(settings.weights, settings.scratch / "weights-verified.json")


def verify_geocalib(path: Path):
    if not path.is_file() or path.stat().st_size != GEOCALIB_SIZE:
        raise FileNotFoundError(f"Missing or incomplete GeoCalib checkpoint: {path}; it is staged by reactor.yaml's build.run")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != GEOCALIB_SHA256:
        raise ValueError(f"GeoCalib checkpoint checksum mismatch: {path}")


def verify_weights(weights: Path, cache: Path):
    """Hash every file on load; write an audit receipt only after success."""
    # Filesystem timestamps can collide across rapid, same-size writes.
    # A prior receipt is evidence of a past check, not of current contents.
    cache.unlink(missing_ok=True)
    manifest_bytes = WEIGHT_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["revision"] != MODEL_REVISION:
        raise ValueError("Checkpoint inventory revision mismatch")
    fingerprints = {}
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe checkpoint inventory path")
        path = weights / relative
        try:
            stat = path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint is incomplete: missing {name}; finish the pinned weight download") from None
        if not path.is_file() or stat.st_size != expected["size"]:
            raise ValueError(f"Checkpoint is incomplete: incorrect size for {name}")
        fingerprints[name] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    receipt = dict(revision=MODEL_REVISION, root=str(weights.resolve()),
                   manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), files=fingerprints)
    logger.info("Verifying SHA256 of %d checkpoint files", len(fingerprints))

    def check(name):
        digest = hashlib.sha256()
        with (weights / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}; download this file again")
        stat = (weights / name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != fingerprints[name]:
            raise RuntimeError(f"Checkpoint changed while being verified: {name}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, fingerprints))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream)
    try:
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Checkpoint SHA256 verification passed")
