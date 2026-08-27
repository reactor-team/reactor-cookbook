"""Keep LingBot weights and causal state resident in an isolated worker."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from reactor_runtime import UploadedFile
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_RESPONSE_PREFIX = "REACTOR_LINGBOT_V1_RESPONSE "
_UPLOAD_SUFFIXES = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class WorkerSettings:
    """Hold paths and inference settings passed to the model worker."""

    python_executable: Path
    source_path: Path
    checkpoint_dir: Path
    runtime_root: Path
    max_chunks: int
    context_latents: int
    max_area: int
    shift: float


class LingBotWorkerBackend:
    """Run the NumPy-1 upstream stack without duplicating its model process."""

    def __init__(self, settings: WorkerSettings) -> None:
        settings.runtime_root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="reactor-lingbot-world-v1-",
            dir=settings.runtime_root,
        )
        self._root = Path(self._temporary.name)
        self._lock = threading.Lock()
        self._request_id = 0
        self._recent_output: deque[str] = deque(maxlen=100)
        worker = Path(__file__).with_name("worker.py")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(settings.source_path), environment.get("PYTHONPATH", "")))
        )
        cache_root = settings.source_path.parent / ".cache"
        environment.update(
            {
                "HF_HOME": str(cache_root / "huggingface"),
                "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
                "TORCH_HOME": str(cache_root / "torch"),
                "TMPDIR": str(cache_root / "tmp"),
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [str(settings.python_executable), str(worker)],
            cwd=settings.source_path,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)
        self._request(
            "initialize",
            checkpoint_dir=str(settings.checkpoint_dir),
            runtime_root=str(self._root / "worker"),
            max_chunks=settings.max_chunks,
            context_latents=settings.context_latents,
            max_area=settings.max_area,
            shift=settings.shift,
        )

    def reset(
        self,
        seed: int,
        anchor_image: Path | UploadedFile,
        intrinsics: Path,
        prompt: str,
    ) -> None:
        """Start a fresh rollout from one image, calibration, and prompt."""
        upload_path: Path | None = None
        if isinstance(anchor_image, UploadedFile):
            suffix = _UPLOAD_SUFFIXES.get(anchor_image.mime_type.lower(), ".image")
            upload_path = self._root / f"anchor_{self._request_id + 1}{suffix}"
            upload_path.write_bytes(anchor_image.data)
            image_path = upload_path
        else:
            image_path = anchor_image
        try:
            self._request(
                "reset",
                seed=int(seed),
                anchor_image=str(image_path),
                intrinsics=str(intrinsics),
                prompt=prompt,
            )
        finally:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)

    def generate_chunk(self, relative_c2ws: np.ndarray, prompt: str) -> np.ndarray:
        """Generate one native chunk while preserving upstream KV and VAE caches."""
        poses = np.asarray(relative_c2ws, dtype=np.float32)
        if poses.shape != (3, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(
                "LingBot relative camera poses must be finite with shape (3, 4, 4)"
            )
        response = self._request(
            "generate",
            relative_c2ws=poses.tolist(),
            prompt=prompt,
        )
        output = Path(str(response["output"]))
        try:
            frames = np.load(output, allow_pickle=False)
            expected = int(response["frame_count"])
            if frames.shape[0] != expected or frames.shape[-1] != 3:
                raise RuntimeError(
                    f"LingBot generated shape {frames.shape}; expected {expected} RGB frames"
                )
            return np.ascontiguousarray(frames, dtype=np.uint8)
        finally:
            output.unlink(missing_ok=True)

    def end_session(self) -> None:
        """Release rollout caches while retaining the loaded model weights."""
        self._request("end_session")

    def close(self) -> None:
        """Stop the worker and remove its NVMe request workspace."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(
                    json.dumps({"id": -1, "command": "shutdown"}) + "\n"
                )
                process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        self._temporary.cleanup()

    def _request(self, command: str, **payload: object) -> dict[str, Any]:
        """Send one JSON-line request and return its correlated response."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                detail = "\n".join(self._recent_output)
                raise RuntimeError(f"LingBot worker is not running\n{detail}")
            self._request_id += 1
            request_id = self._request_id
            request = {"id": request_id, "command": command, **payload}
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            while True:
                line = process.stdout.readline()
                if not line:
                    detail = "\n".join(self._recent_output)
                    raise RuntimeError(
                        f"LingBot worker exited with code {process.poll()}\n{detail}"
                    )
                line = line.rstrip()
                if not line.startswith(_RESPONSE_PREFIX):
                    self._recent_output.append(line)
                    if line:
                        logger.info("LingBot worker", output=line[-1000:])
                    continue
                response = json.loads(line.removeprefix(_RESPONSE_PREFIX))
                if int(response.get("id", -2)) != request_id:
                    continue
                if not bool(response.get("ok")):
                    detail = str(response.get("error", "LingBot worker request failed"))
                    logs = "\n".join(self._recent_output)
                    raise RuntimeError(f"{detail}\n{logs}")
                return response
