"""Keep EVOKE weights and autoregressive state resident across chunk requests."""

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

_RESPONSE_PREFIX = "REACTOR_EVOKE_RESPONSE "
_UPLOAD_SUFFIXES = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "application/x-npz": ".npz",
    "application/octet-stream": ".npz",
}


@dataclass(frozen=True)
class WorkerSettings:
    """Hold paths and fixed release-recipe settings passed to the EVOKE worker."""

    python_executable: Path
    source_path: Path
    base_model: Path
    transformer: Path
    vigeo_path: Path
    default_image: Path
    stability_prompt: str
    seed: int
    max_chunks: int
    reference_seconds: float


class EvokeWorkerBackend:
    """Generate native EVOKE chunks in a persistent Python 3.10 subprocess."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="reactor-evoke-")
        self._root = Path(self._temporary.name)
        self._lock = threading.Lock()
        self._request_id = 0
        self._recent_output: deque[str] = deque(maxlen=120)
        self._session_uploads: list[Path] = []
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(settings.source_path)
        environment["EVOKE_VIGEO_WEIGHTS"] = str(settings.vigeo_path)
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        environment.setdefault("DA3_LOG_LEVEL", "WARN")
        worker = Path(__file__).with_name("worker.py")
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
            source_path=str(settings.source_path),
            base_model=str(settings.base_model),
            transformer=str(settings.transformer),
            vigeo_path=str(settings.vigeo_path),
            default_image=str(settings.default_image),
            stability_prompt=settings.stability_prompt,
            runtime_root=str(self._root / "worker"),
            seed=settings.seed,
            max_chunks=settings.max_chunks,
            reference_seconds=settings.reference_seconds,
        )

    def reset(
        self,
        *,
        mode: str,
        media: Path | UploadedFile | None,
        pose: UploadedFile | None,
        prompt: str,
        seed: int,
        source_fps: int = 30,
        source_height: int = 720,
        source_width: int = 1280,
    ) -> None:
        """Start a fresh rollout without reloading model weights."""
        previous_uploads = self._session_uploads
        self._session_uploads = []
        media_path = self._materialize(media, "media")
        pose_path = self._materialize(pose, "pose")
        try:
            self._request(
                "reset",
                mode=mode,
                media=str(media_path) if media_path is not None else None,
                pose=str(pose_path) if pose_path is not None else None,
                prompt=prompt,
                seed=int(seed),
                source_fps=int(source_fps),
                source_height=int(source_height),
                source_width=int(source_width),
            )
        finally:
            for path in previous_uploads:
                path.unlink(missing_ok=True)

    def generate_chunk(
        self,
        trajectory_c2w: np.ndarray | None,
        *,
        seed: int,
        prompt: str,
    ) -> np.ndarray:
        """Generate one native chunk with the latest prompt and camera trajectory."""
        trajectory = None
        if trajectory_c2w is not None:
            value = np.asarray(trajectory_c2w, dtype=np.float32)
            if value.shape != (36, 4, 4) or not np.isfinite(value).all():
                raise ValueError("EVOKE camera trajectory must have shape (36, 4, 4)")
            trajectory = value.tolist()
        response = self._request(
            "generate",
            trajectory=trajectory,
            seed=int(seed),
            prompt=prompt,
        )
        output = Path(str(response["output"]))
        try:
            return np.ascontiguousarray(
                np.load(output, allow_pickle=False), dtype=np.uint8
            )
        finally:
            output.unlink(missing_ok=True)

    def end_session(self) -> None:
        """Release rollout caches while keeping loaded weights resident."""
        self._request("end_session")
        self._clear_session_uploads()

    def close(self) -> None:
        """Stop the worker and remove its request workspace."""
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

    def _materialize(self, value: Path | UploadedFile | None, stem: str) -> Path | None:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        suffix = _UPLOAD_SUFFIXES.get(value.mime_type.lower())
        if suffix is None:
            suffix = Path(value.name).suffix.lower() or ".bin"
        path = self._root / f"{stem}_{self._request_id + 1}{suffix}"
        path.write_bytes(value.data)
        self._session_uploads.append(path)
        return path

    def _clear_session_uploads(self) -> None:
        for path in self._session_uploads:
            path.unlink(missing_ok=True)
        self._session_uploads.clear()

    def _request(self, command: str, **payload: object) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                detail = "\n".join(self._recent_output)
                raise RuntimeError(f"EVOKE worker is not running\n{detail}")
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
                        f"EVOKE worker exited with code {process.poll()}\n{detail}"
                    )
                line = line.rstrip()
                if not line.startswith(_RESPONSE_PREFIX):
                    self._recent_output.append(line)
                    if line:
                        logger.info("EVOKE worker", output=line[-1200:])
                    continue
                response = json.loads(line.removeprefix(_RESPONSE_PREFIX))
                if int(response.get("id", -2)) != request_id:
                    continue
                if not bool(response.get("ok")):
                    detail = str(response.get("error", "EVOKE worker request failed"))
                    logs = "\n".join(self._recent_output)
                    raise RuntimeError(f"{detail}\n{logs}")
                return response
