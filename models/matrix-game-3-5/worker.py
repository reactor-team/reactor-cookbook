#!/usr/bin/env python3
"""Run public Matrix-Game-3.5 inference behind a small JSON-line protocol."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import queue
import random
import shutil
import sys
import threading
import traceback
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_RESPONSE_PREFIX = "REACTOR_MATRIX_RESPONSE "
_RGB_FRAMES_PER_CHUNK = 12


class _InteractiveStopError(Exception):
    """Signal normal termination while the rollout waits for another chunk."""


@dataclass(frozen=True)
class _CameraChunk:
    """Carry chunk conditions into the rollout thread."""

    index: int
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    prompt: str


@dataclass(frozen=True)
class _GeneratedChunk:
    """Carry decoded frames back to the worker request thread."""

    index: int
    frames: np.ndarray


class _Stop:
    """Mark the end of an interactive camera stream."""


class _InteractiveSession:
    """Bridge synchronous worker requests into one persistent GPU rollout."""

    def __init__(self, max_chunks: int) -> None:
        self.max_chunks = max_chunks
        self._cameras: queue.Queue[_CameraChunk | _Stop] = queue.Queue(maxsize=1)
        self._results: queue.Queue[_GeneratedChunk | BaseException] = queue.Queue(maxsize=1)
        self._stop = _Stop()

    def camera_for_chunk(
        self,
        chunk_index: int,
        *,
        frames_per_chunk: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Wait for the camera and prompt conditions for the next model chunk."""
        if frames_per_chunk * 4 != _RGB_FRAMES_PER_CHUNK:
            raise RuntimeError("Matrix interactive rollout expects three latent frames per chunk")
        value = self._cameras.get()
        if isinstance(value, _Stop):
            raise _InteractiveStopError
        if value.index != chunk_index:
            raise RuntimeError(
                f"interactive camera index {value.index} does not match chunk {chunk_index}"
            )
        return value.intrinsics, value.extrinsics, value.prompt

    def publish_chunk(self, chunk_index: int, frames: np.ndarray) -> None:
        """Make one decoded chunk available to the waiting JSON request."""
        self._results.put(
            _GeneratedChunk(
                chunk_index,
                np.ascontiguousarray(frames, dtype=np.uint8),
            )
        )

    def generate(
        self,
        chunk_index: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Submit one conditioned chunk and wait for its decoded RGB output."""
        self._cameras.put(_CameraChunk(chunk_index, intrinsics, extrinsics, prompt))
        result = self._results.get()
        if isinstance(result, BaseException):
            raise RuntimeError("Matrix interactive rollout failed") from result
        if result.index != chunk_index:
            raise RuntimeError(
                f"interactive output index {result.index} does not match chunk {chunk_index}"
            )
        return result.frames

    def fail(self, error: BaseException) -> None:
        """Wake a request waiting on a failed rollout."""
        self._results.put(error)

    def stop(self) -> None:
        """Wake a rollout waiting for camera input and request termination."""
        with suppress(queue.Full):
            self._cameras.put_nowait(self._stop)


class MatrixRuntime:
    """Keep the upstream model and causal rollout state resident."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings = settings
        self._runtime_root = Path(settings["runtime_root"]).resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._default_anchor = Path(settings["anchor_image"]).resolve()
        self._prompt_file = Path(settings["prompt_file"]).resolve()
        self._default_prompt = self._prompt_file.read_text(encoding="utf-8").strip()
        if not self._default_prompt:
            raise ValueError(f"prompt file is empty: {self._prompt_file}")
        self._counter = 0
        self._max_chunks = int(settings["max_chunks"])
        self._session: _InteractiveSession | None = None
        self._session_thread: threading.Thread | None = None
        self._session_root: Path | None = None
        self._session_dataset: Any = None
        self._chunk_index = 0
        self._active_seed = int(settings["seed"])

        os.environ["DA3_MODEL_PATH"] = str(Path(settings["da3_dir"]).resolve())
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        accelerate = importlib.import_module("accelerate")
        torch = importlib.import_module("torch")
        load_inference_config = importlib.import_module("distilled_config").load_inference_config
        infer = importlib.import_module("infer")
        build_model = importlib.import_module("tools.run_distilled_inference").build_model
        build_runtime_args = importlib.import_module(
            "examples.wanvideo.pipeline.mosaic.causal_config"
        ).build_runtime_args
        run_distilled_inference = importlib.import_module(
            "examples.wanvideo.pipeline.mosaic.causal_inference"
        ).run_distilled_inference
        build_mosaic_validation_dataset = importlib.import_module(
            "examples.wanvideo.pipeline.mosaic.datasets"
        ).build_mosaic_validation_dataset

        self._np = np
        self._torch = torch
        self._build_index = infer.build_index
        self._build_input_workspace = infer.build_input_workspace
        self._build_runtime_args = build_runtime_args
        self._build_dataset = build_mosaic_validation_dataset
        self._run_inference = run_distilled_inference
        if "interactive_session" not in inspect.signature(run_distilled_inference).parameters:
            raise RuntimeError(
                "Matrix source is missing the Reactor stateful rollout patch; "
                "apply the patch documented in models/matrix-game-3-5/README.md"
            )
        self._config = load_inference_config(Path(settings["inference_config"]))
        if int(self._config.num_blocks) != 1:
            raise ValueError("Reactor Matrix inference config must set num_blocks: 1")
        self._accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[
                accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
            ],
        )
        initial_args, _initial_dataset, initial_root = self._prepare_request(
            Path(settings["default_camera"]),
            seed=int(self._config.seed),
            anchor_image=self._default_anchor,
            prompt=self._default_prompt,
        )
        try:
            self._seed(int(self._config.seed))
            self._model = build_model(initial_args, self._accelerator)
            self._model.eval()
        finally:
            shutil.rmtree(initial_root, ignore_errors=True)

    def reset(self, seed: int, anchor_image: Path, prompt: str) -> None:
        """Start a fresh causal session without reloading model weights."""
        anchor_image = anchor_image.resolve()
        prompt = prompt.strip()
        if not anchor_image.is_file():
            raise FileNotFoundError(f"Matrix anchor image does not exist: {anchor_image}")
        if not prompt:
            raise ValueError("Matrix prompt must not be empty")
        self._stop_session()
        self._active_seed = int(seed)
        args, dataset, request_root = self._prepare_request(
            Path(self._settings["default_camera"]),
            seed=self._active_seed,
            anchor_image=anchor_image,
            prompt=prompt,
        )
        session = _InteractiveSession(self._max_chunks)
        self._session = session
        self._session_root = request_root
        self._session_dataset = dataset
        self._chunk_index = 0
        self._seed(self._active_seed)
        output = self._runtime_root / "interactive-unused.mp4"

        def run() -> None:
            try:
                self._run_inference(
                    self._accelerator,
                    dataset,
                    self._model,
                    output,
                    args,
                    interactive_session=session,
                )
            except _InteractiveStopError:
                return
            except BaseException as error:
                traceback.print_exc()
                session.fail(error)

        self._session_thread = threading.Thread(
            target=run,
            name="matrix-stateful-rollout",
            daemon=True,
        )
        self._session_thread.start()

    def generate(self, camera: Path, seed: int, prompt: str) -> Path:
        """Generate one chunk while preserving the active causal state."""
        if self._session is None or self._session_thread is None:
            self.reset(seed, self._default_anchor, self._default_prompt)
        if seed != self._active_seed:
            raise ValueError("Matrix seed can change only at reset")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Matrix prompt must not be empty")
        session = self._session
        thread = self._session_thread
        if session is None or thread is None or not thread.is_alive():
            raise RuntimeError("Matrix interactive rollout is not running")
        intrinsics, extrinsics = self._prepare_camera_chunk(camera)
        frames = session.generate(self._chunk_index, intrinsics, extrinsics, prompt)
        self._chunk_index += 1
        self._counter += 1
        output = self._runtime_root / "outputs" / f"chunk_{self._counter:06d}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, frames, allow_pickle=False)
        return output

    def _prepare_request(
        self,
        camera: Path,
        *,
        seed: int,
        anchor_image: Path,
        prompt: str,
    ) -> tuple[Any, Any, Path]:
        """Build the one-scene public Matrix dataset for a rollout session."""
        request_root = self._runtime_root / "requests" / uuid.uuid4().hex
        workspace = Path(
            self._build_input_workspace(
                SimpleNamespace(
                    person="first",
                    image=str(anchor_image),
                    camera=str(camera),
                    caption="",
                    prompt=prompt,
                    prompt_file="",
                    refs="",
                    num_blocks=1,
                    camera_convention="c2w",
                ),
                str(request_root),
            )
        )
        environment = os.environ.copy()
        index = Path(self._build_index(str(workspace), sys.executable, environment))
        args = self._build_runtime_args(
            self._config,
            checkpoint=Path(self._settings["checkpoint"]),
            dataset_index=index,
            workspace=workspace,
            wan_dir=Path(self._settings["wan_dir"]),
            tokenizer_dir=Path(self._settings["tokenizer_dir"]),
            memory_cache_dir=self._runtime_root / "memory_latents",
        )
        args.validation_seed = int(seed)
        args.dataset_pass_id = 0
        args.start_epoch = 0
        args.start_global_step = 0
        args.rank = self._accelerator.process_index
        dataset = self._build_dataset(args, train_dataset=None)
        if len(dataset) != 1:
            raise RuntimeError(
                f"Matrix rollout workspace must contain one sample, got {len(dataset)}"
            )
        return args, dataset, request_root

    def close(self) -> None:
        """Stop the active rollout and release its request workspace."""
        self._stop_session()

    def end_session(self) -> None:
        """Release active rollout state while keeping loaded model weights resident."""
        self._stop_session()

    def _stop_session(self) -> None:
        """Stop one rollout at its next camera-input boundary."""
        session = self._session
        thread = self._session_thread
        self._session = None
        self._session_thread = None
        self._session_dataset = None
        if session is not None:
            session.stop()
        if thread is not None:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("Matrix interactive rollout did not stop")
        if self._session_root is not None:
            shutil.rmtree(self._session_root, ignore_errors=True)
            self._session_root = None

    def _prepare_camera_chunk(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Convert one c2w trajectory to Matrix's normalized PRoPE inputs."""
        with np.load(path) as archive:
            c2w = np.asarray(archive["extrinsics_c2w"], dtype=np.float32)
            packed = np.asarray(archive["intrinsics"], dtype=np.float32)
        if c2w.shape != (_RGB_FRAMES_PER_CHUNK + 1, 4, 4):
            raise ValueError(
                "interactive camera trajectory must contain anchor plus "
                f"{_RGB_FRAMES_PER_CHUNK} poses, got {c2w.shape}"
            )
        w2c = np.linalg.inv(c2w[1:].astype(np.float64)).astype(np.float32)
        matrices = _intrinsic_matrices(packed, int(c2w.shape[0]))[1:]
        temporal_mean = bool(self._session_dataset._intrinsics_temporal_mean_enabled())
        matrices = self._session_dataset.normalize_and_scale_intrinsics(
            matrices,
            H_img=int(self._config.height),
            W_img=int(self._config.width),
            temporal_mean=temporal_mean,
        )
        return (
            np.ascontiguousarray(matrices, dtype=np.float32),
            np.ascontiguousarray(w2c, dtype=np.float32),
        )

    def _seed(self, seed: int) -> None:
        """Seed every upstream generator used by the causal rollout."""
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)
        self._np.random.seed(seed & 0xFFFFFFFF)
        random.seed(seed)


def _intrinsic_matrices(value: np.ndarray, frame_count: int) -> np.ndarray:
    """Expand packed or matrix intrinsics to one matrix per camera pose."""
    intrinsics = np.asarray(value, dtype=np.float32)
    if intrinsics.shape == (4,):
        intrinsics = np.repeat(intrinsics[None], frame_count, axis=0)
    if intrinsics.ndim == 2 and intrinsics.shape[1:] == (4,):
        matrices = np.zeros((int(intrinsics.shape[0]), 3, 3), dtype=np.float32)
        matrices[:, 0, 0] = intrinsics[:, 0]
        matrices[:, 1, 1] = intrinsics[:, 1]
        matrices[:, 0, 2] = intrinsics[:, 2]
        matrices[:, 1, 2] = intrinsics[:, 3]
        matrices[:, 2, 2] = 1.0
        intrinsics = matrices
    elif intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], frame_count, axis=0)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"unsupported Matrix intrinsics shape: {intrinsics.shape}")
    if int(intrinsics.shape[0]) < frame_count:
        tail = np.repeat(intrinsics[-1:], frame_count - int(intrinsics.shape[0]), axis=0)
        intrinsics = np.concatenate([intrinsics, tail], axis=0)
    return np.ascontiguousarray(intrinsics[:frame_count], dtype=np.float32)


def _respond(request_id: int, *, ok: bool, **payload: object) -> None:
    """Write one correlated protocol response."""
    print(
        _RESPONSE_PREFIX + json.dumps({"id": request_id, "ok": ok, **payload}, ensure_ascii=False),
        flush=True,
    )


def main() -> int:
    """Serve requests from stdin until the parent closes or asks to stop."""
    runtime: MatrixRuntime | None = None
    for line in sys.stdin:
        request = json.loads(line)
        request_id = int(request["id"])
        command = str(request["command"])
        if command == "shutdown":
            if runtime is not None:
                runtime.close()
            _respond(request_id, ok=True)
            return 0
        try:
            if command == "initialize":
                runtime = MatrixRuntime(request)
                _respond(request_id, ok=True)
            elif command == "reset":
                if runtime is None:
                    raise RuntimeError("Matrix worker is not initialized")
                runtime.reset(
                    int(request["seed"]),
                    Path(request["anchor_image"]),
                    str(request["prompt"]),
                )
                _respond(request_id, ok=True)
            elif command == "generate":
                if runtime is None:
                    raise RuntimeError("Matrix worker is not initialized")
                output = runtime.generate(
                    Path(request["camera"]),
                    int(request["seed"]),
                    str(request["prompt"]),
                )
                _respond(request_id, ok=True, output=str(output))
            elif command == "end_session":
                if runtime is None:
                    raise RuntimeError("Matrix worker is not initialized")
                runtime.end_session()
                _respond(request_id, ok=True)
            else:
                raise ValueError(f"unknown Matrix worker command: {command}")
        except Exception as error:
            traceback.print_exc()
            _respond(request_id, ok=False, error=f"{type(error).__name__}: {error}")
    if runtime is not None:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
