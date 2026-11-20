"""Drive the worker protocol, health and log relay against a scripted worker."""
import logging
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

import upstream_backend
from upstream_backend import InferenceWorkerError, WorldCrafterBackend
from worldcrafter_assets import Settings

FAKE_WORKER = textwrap.dedent('''
    import argparse, json, os, socket, sys, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket")
    args, _ = parser.parse_known_args()
    print("worker loaded the model", flush=True)
    connection = socket.socket(socket.AF_UNIX)
    connection.connect(args.socket)
    stream = connection.makefile("rwb")
    stream.write(b'{"gpu_count": 1}\\n'); stream.flush()
    for line in stream:
        request = json.loads(line)
        mode = os.environ["FAKE_WORKER_MODE"]
        if mode == "hang":
            time.sleep(60)
        if mode == "error":
            print("Traceback: boom", file=sys.stderr, flush=True)
            stream.write((json.dumps(dict(id=request["id"], error="RuntimeError: boom")) + "\\n").encode())
            stream.flush()
            break
        stream.write((json.dumps(dict(id=request["id"])) + "\\n").encode()); stream.flush()
''')


@pytest.fixture
def backend(monkeypatch):
    # A short directory keeps the worker's socket path under the Unix limit.
    with tempfile.TemporaryDirectory(prefix="wc-test-") as directory:
        root = Path(directory)
        worker = root / "fake_worker.py"
        worker.write_text(FAKE_WORKER)
        monkeypatch.setattr(upstream_backend, "WORKER", worker)
        settings = Settings(python=Path(sys.executable), source=root, weights=root, scratch=root,
                            torch_home=root, gpu_count=1, timeout=30)
        created = []

        def start(mode):
            monkeypatch.setenv("FAKE_WORKER_MODE", mode)
            created.append(WorldCrafterBackend(settings))
            return created[-1]

        yield start
        for instance in created:
            instance.close()


def wait_for(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)
    return predicate()


def test_worker_output_reaches_runtime_logging(backend, caplog):
    caplog.set_level(logging.INFO, logger="worldcrafter.worker")
    worker = backend("ok")
    worker.reset()
    assert worker.healthy
    assert wait_for(lambda: any(r.getMessage() == "worker loaded the model" for r in caplog.records))


def test_worker_error_reply_leaves_backend_unhealthy(backend, caplog):
    caplog.set_level(logging.INFO, logger="worldcrafter.worker")
    worker = backend("error")
    with pytest.raises(InferenceWorkerError, match="boom"):
        worker.reset()
    assert not worker.healthy
    assert wait_for(lambda: any(r.getMessage() == "Traceback: boom" for r in caplog.records))
    with pytest.raises(InferenceWorkerError, match="closed"):
        worker.reset()


def test_hung_worker_times_out_and_is_stopped(backend, monkeypatch):
    monkeypatch.setattr(upstream_backend, "CALL_TIMEOUT_SECONDS", 0.5)
    worker = backend("hang")
    started = time.monotonic()
    with pytest.raises(InferenceWorkerError, match="did not answer"):
        worker.reset()
    assert time.monotonic() - started < 10
    assert not worker.healthy and worker._process.poll() is not None


def test_dead_worker_cleanup_preserves_error_and_reaps_process(backend):
    worker = backend("ok")
    original = worker._stream

    class BrokenStream:
        def write(self, data):
            raise BrokenPipeError("worker disappeared")

        def close(self):
            original.close()
            raise BrokenPipeError("buffered close also failed")

    worker._stream = BrokenStream()
    root = worker._root
    with pytest.raises(InferenceWorkerError, match="worker is unreachable"):
        worker.reset()
    assert not worker.healthy
    assert worker._process.poll() is not None
    assert worker._connection.fileno() == -1
    assert worker._listener.fileno() == -1
    assert not root.exists()
    worker.close()


def test_exited_worker_disconnect_is_recoverable(backend):
    worker = backend("ok")
    worker._process.kill()
    worker._process.wait(timeout=5)
    with pytest.raises(InferenceWorkerError):
        worker.reset()
    assert not worker.healthy and not worker._root.exists()
