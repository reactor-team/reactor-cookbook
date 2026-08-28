"""Keep Matrix weights and causal state resident while generating chunks."""

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

_RESPONSE_PREFIX = "REACTOR_MATRIX_RESPONSE "
_RGB_FRAMES_PER_CHUNK = 12
_UPLOAD_SUFFIXES = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class WorkerSettings:
    """Hold paths and default inference inputs passed to the Matrix worker."""

    python_executable: Path
    source_path: Path
    inference_config: Path
    checkpoint: Path
    wan_dir: Path
    tokenizer_dir: Path
    da3_dir: Path
    anchor_image: Path
    default_camera: Path
    default_prompt: str
    seed: int
    max_chunks: int


class MatrixWorkerBackend:
    """Generate causal chunks in a persistent subprocess.

    The subprocess isolates Matrix's top-level ``examples`` package from this
    repository's package of the same name. It constructs the 5B model once and
    reuses those weights for every request. One rollout keeps its causal KV,
    dynamic visual context, and Patch Memory across chunk requests.
    """

    def __init__(self, settings: WorkerSettings, intrinsics: np.ndarray) -> None:
        self._settings = settings
        self._intrinsics = _normalize_intrinsics(intrinsics)
        self._temporary = tempfile.TemporaryDirectory(prefix="reactor-matrix-game-3-5-")
        self._root = Path(self._temporary.name)
        self._lock = threading.Lock()
        self._request_id = 0
        self._recent_output: deque[str] = deque(maxlen=80)

        worker = Path(__file__).with_name("worker.py")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(settings.source_path), environment.get("PYTHONPATH", "")),
            )
        )
        environment["DA3_MODEL_PATH"] = str(settings.da3_dir)
        environment.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
            inference_config=str(settings.inference_config),
            checkpoint=str(settings.checkpoint),
            wan_dir=str(settings.wan_dir),
            tokenizer_dir=str(settings.tokenizer_dir),
            da3_dir=str(settings.da3_dir),
            anchor_image=str(settings.anchor_image),
            default_camera=str(settings.default_camera),
            default_prompt=settings.default_prompt,
            runtime_root=str(self._root / "worker"),
            seed=settings.seed,
            max_chunks=settings.max_chunks,
        )

    def reset(
        self,
        seed: int,
        anchor_image: Path | UploadedFile,
        prompt: str,
    ) -> None:
        """Start a fresh stateful rollout from an image and prompt."""
        upload_path: Path | None = None
        if isinstance(anchor_image, UploadedFile):
            suffix = _UPLOAD_SUFFIXES.get(anchor_image.mime_type.lower())
            if suffix is None:
                candidate = Path(anchor_image.name).suffix.lower()
                suffix = (
                    candidate
                    if candidate in set(_UPLOAD_SUFFIXES.values())
                    else ".image"
                )
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
                prompt=prompt,
            )
        finally:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)

    def generate_chunk(
        self,
        trajectory_c2w: np.ndarray,
        seed: int,
        prompt: str,
    ) -> np.ndarray:
        """Generate one chunk for camera poses and the active prompt."""
        trajectory = _validate_camera_trajectory(trajectory_c2w)
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Matrix prompt must not be empty")
        intrinsics = np.repeat(
            self._intrinsics[None],
            int(trajectory.shape[0]),
            axis=0,
        )
        camera_path = self._root / f"camera_{self._request_id + 1}.npz"
        np.savez_compressed(
            camera_path,
            extrinsics_c2w=np.ascontiguousarray(trajectory, dtype=np.float32),
            intrinsics=np.ascontiguousarray(intrinsics, dtype=np.float32),
        )
        output: Path | None = None
        try:
            response = self._request(
                "generate",
                camera=str(camera_path),
                seed=int(seed),
                prompt=prompt,
            )
            output = Path(str(response["output"]))
            frames = np.load(output, allow_pickle=False)
            if frames.shape[:1] != (_RGB_FRAMES_PER_CHUNK,):
                raise RuntimeError(
                    f"Matrix generated {int(frames.shape[0])} frames; "
                    f"expected {_RGB_FRAMES_PER_CHUNK}"
                )
            return np.ascontiguousarray(frames, dtype=np.uint8)
        finally:
            camera_path.unlink(missing_ok=True)
            if output is not None:
                output.unlink(missing_ok=True)

    def end_session(self) -> None:
        """Release rollout caches while keeping model weights resident."""
        self._request("end_session")

    def close(self) -> None:
        """Stop the worker and remove its temporary request workspace."""
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
                process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self._temporary.cleanup()

    def _request(self, command: str, **payload: object) -> dict[str, Any]:
        """Send one request and return its correlated worker response."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                detail = "\n".join(self._recent_output)
                raise RuntimeError(f"Matrix worker is not running\n{detail}")
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
                        f"Matrix worker exited with code {process.poll()}\n{detail}"
                    )
                line = line.rstrip()
                if not line.startswith(_RESPONSE_PREFIX):
                    self._recent_output.append(line)
                    if line:
                        logger.info("Matrix worker", output=line[-1000:])
                    continue
                response = json.loads(line.removeprefix(_RESPONSE_PREFIX))
                if int(response.get("id", -2)) != request_id:
                    continue
                if not bool(response.get("ok")):
                    detail = str(response.get("error", "Matrix worker request failed"))
                    logs = "\n".join(self._recent_output)
                    raise RuntimeError(f"{detail}\n{logs}")
                return response


def _normalize_intrinsics(value: np.ndarray) -> np.ndarray:
    """Return one packed ``[fx, fy, cx, cy]`` intrinsic vector."""
    intrinsic = np.asarray(value, dtype=np.float32)
    if intrinsic.ndim == 2 and intrinsic.shape[1] == 4:
        intrinsic = intrinsic[0]
    if intrinsic.shape == (3, 3):
        intrinsic = np.asarray(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
            dtype=np.float32,
        )
    if intrinsic.shape != (4,):
        raise ValueError(
            f"Matrix intrinsics must have shape (4,), (N, 4), or (3, 3); got {intrinsic.shape}"
        )
    if not np.isfinite(intrinsic).all() or bool(np.any(intrinsic[:2] <= 0)):
        raise ValueError("Matrix intrinsics must be finite with positive focal lengths")
    return np.ascontiguousarray(intrinsic)


def _validate_camera_trajectory(value: np.ndarray) -> np.ndarray:
    """Validate one anchor plus one chunk of generated camera poses."""
    trajectory = np.asarray(value, dtype=np.float32)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError(
            f"camera trajectory must have shape (N, 4, 4), got {trajectory.shape}"
        )
    expected = _RGB_FRAMES_PER_CHUNK + 1
    if int(trajectory.shape[0]) != expected:
        raise ValueError(
            f"camera trajectory must contain anchor plus {expected - 1} poses, "
            f"got {int(trajectory.shape[0])}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("camera trajectory must contain only finite values")
    return np.ascontiguousarray(trajectory)
