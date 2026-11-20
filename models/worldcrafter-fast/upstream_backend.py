"""Isolate Python 3.11/NumPy 1 inference from Python 3.12/NumPy 2 Reactor."""
import atexit
from contextlib import suppress
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np

from worldcrafter_model import WorldCrafterInput, WorldCrafterResult

WORKER = Path(__file__).with_name("worker.py")
# A warmed-up chunk takes a few seconds; the bound leaves room for a shape the
# warmup did not compile and covers a collective that never returns.
CALL_TIMEOUT_SECONDS = 180.0
worker_logger = logging.getLogger("worldcrafter.worker")


class InferenceWorkerError(RuntimeError):
    """The inference worker failed, timed out or disconnected; it serves no further calls."""


class WorldCrafterBackend:
    def __init__(self, settings):
        self._settings = settings
        self._lock = threading.Lock()
        self._process = None
        self._relay = None
        self._connection = None
        self._stream = None
        self._temporary = tempfile.TemporaryDirectory(prefix="wc-", dir=settings.scratch)
        self._root = Path(self._temporary.name)
        self._sequence = 0
        self._closed = False
        self._listener = socket.socket(socket.AF_UNIX)
        endpoint = str(self._root / "rpc.sock")
        if len(endpoint.encode()) >= 104:
            raise ValueError("WORLDCRAFTER_SCRATCH path is too long for a Unix socket")
        self._listener.bind(endpoint)
        self._listener.listen(1)
        self._listener.settimeout(1)
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(settings.source), PYTHONUNBUFFERED="1",
            WORLDCRAFTER_DEMO_MODEL=str(settings.weights),
            WORLDCRAFTER_DEMO_DATA=str(self._root / "engine"),
            WORLDCRAFTER_DEMO_BACKEND="context" if settings.gpu_count == 8 else "single",
            WORLDCRAFTER_DEMO_GPUS=str(settings.gpu_count),
            WORLDCRAFTER_DEMO_COMPILE=str(int(settings.compile)),
            WORLDCRAFTER_DEMO_OPTIMIZED=str(int(settings.gpu_count == 8)),
            WORLDCRAFTER_VAE_MODE="pipeline8" if settings.gpu_count == 8 else "single",
            WORLDCRAFTER_KV_FUSION="rope",
            WORLDCRAFTER_CONTROL_SOCKET=str(self._root / "control.sock"),
            TORCH_HOME=str(settings.torch_home),
        )
        command = [str(settings.python)]
        if settings.gpu_count == 8:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"]
        command += [str(WORKER), "--socket", endpoint, "--source", str(settings.source),
                    "--root", str(self._root), "--max-chunks", str(settings.max_chunks),
                    "--warmup-chunks", str(settings.warmup_chunks)]
        try:
            self._process = subprocess.Popen(command, cwd=settings.source, env=environment,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             start_new_session=True)
            self._relay = threading.Thread(target=_relay, args=(self._process.stdout,),
                                           name="worldcrafter-worker-log", daemon=True)
            self._relay.start()
            deadline = time.monotonic() + settings.timeout
            while True:
                if self._process.poll() is not None:
                    raise InferenceWorkerError(f"Inference worker exited with code {self._process.returncode} during startup")
                if time.monotonic() >= deadline:
                    raise InferenceWorkerError("Inference worker startup timed out")
                try:
                    self._connection, _ = self._listener.accept()
                    break
                except socket.timeout:
                    continue
            # Startup loads the model and runs the warmup before the worker reports ready.
            self._connection.settimeout(max(1.0, deadline - time.monotonic()))
            self._stream = self._connection.makefile("rwb")
            ready = self._receive()
            if ready.get("gpu_count") != settings.gpu_count:
                raise InferenceWorkerError("Loaded GPU count does not match configuration")
            self._connection.settimeout(CALL_TIMEOUT_SECONDS)
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    @property
    def healthy(self) -> bool:
        return not self._closed

    def _receive(self):
        try:
            line = self._stream.readline(4 * 1024 * 1024)
        except OSError as exc:
            raise InferenceWorkerError(f"Inference worker did not answer: {exc}") from exc
        if not line or not line.endswith(b"\n"):
            raise InferenceWorkerError("Inference worker disconnected or sent an invalid response")
        result = json.loads(line)
        if "error" in result:
            raise InferenceWorkerError(result["error"])
        return result

    def _request(self, command, **arguments):
        if self._closed:
            raise InferenceWorkerError("Inference worker is closed")
        self._sequence += 1
        request_id = self._sequence
        try:
            try:
                self._stream.write((json.dumps(dict(id=request_id, command=command, **arguments)) + "\n").encode())
                self._stream.flush()
            except OSError as exc:
                raise InferenceWorkerError(f"Inference worker is unreachable: {exc}") from exc
            reply = self._receive()
            if reply.get("id") != request_id:
                raise InferenceWorkerError("Inference worker response identity mismatch")
            return reply
        except BaseException:
            self.close()
            raise

    def generate(self, input: WorldCrafterInput) -> WorldCrafterResult:
        with self._lock:
            if input.image is not None:
                (self._root / "anchor.image").write_bytes(input.image)
            if input.path_poses is not None:
                np.savez(self._root / "poses.npz", path=input.path_poses)
            else:
                np.savez(self._root / "poses.npz", local=input.local_poses, world=input.world_poses)
            reply = self._request("generate", world_id=input.world_id,
                                  has_image=input.image is not None, prompt=input.prompt,
                                  seed=input.seed, gravity_align=input.gravity_align)
            output = self._root / "frames.npy"
            try:
                frames = np.load(output, allow_pickle=False)
                if frames.dtype != np.uint8 or frames.shape != (33, 384, 640, 3):
                    raise ValueError(f"Unexpected frames: {frames.shape}, {frames.dtype}")
                frames = np.ascontiguousarray(frames)
            finally:
                output.unlink(missing_ok=True)
            return WorldCrafterResult(frames=frames, **{key: reply[key] for key in
                ("world_id", "chunk_index", "camera_end", "gravity_aligned", "gpu_count", "world_up")})

    def reset(self):
        with self._lock:
            self._request("reset")
            (self._root / "anchor.image").unlink(missing_ok=True)
            (self._root / "poses.npz").unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        # Buffered socket close can flush into a dead worker. Cleanup must
        # still reap the process and preserve the original inference error.
        for resource in (self._stream, self._connection, self._listener):
            if resource is not None:
                with suppress(OSError):
                    resource.close()
        if self._process is not None and self._process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=10)
        if self._relay is not None:
            self._relay.join(timeout=5)
        self._temporary.cleanup()


def _relay(stream):
    """Forward each line the worker prints to the Runtime's logging."""
    with stream:
        for raw in iter(stream.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if line:
                worker_logger.info(line)
