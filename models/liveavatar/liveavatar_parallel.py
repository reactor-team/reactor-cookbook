"""Native TPP behind the same demand-driven Runtime backend interface.

Spawns five workers for the released path, or three when ``LIVEAVATAR_TURBO=1``
selects the opt-in three-GPU turbo mode (see ``liveavatar_turbo``)."""

from __future__ import annotations

import atexit
import os
import queue
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from liveavatar_assets import WORK, prepare_assets
from liveavatar_audio import OUTPUT_SAMPLE_RATE, playback_audio


class ParallelBackend:
    def __init__(self):
        self.base, self.lora = prepare_assets()
        self.processes = []
        self.directory = None
        self.active = False
        self.pending_ack = False
        self.audio = np.zeros(0, np.float32)
        self.offset = 0
        self._launch()
        atexit.register(self.shutdown)

    def _launch(self):
        import multiprocessing as mp

        from liveavatar_parallel_worker import run_worker
        from liveavatar_turbo import turbo_plan

        world_size = turbo_plan()["world_size"]
        context = mp.get_context("spawn")
        self.directory = tempfile.TemporaryDirectory(prefix="tpp-", dir=WORK)
        self.commands = [context.Queue(maxsize=1) for _ in range(world_size)]
        self.results = context.Queue(maxsize=2)
        self.ack = context.Queue(maxsize=1)
        self.processes = [
            context.Process(
                target=run_worker,
                args=(
                    rank,
                    os.getpid(),
                    self.directory.name,
                    str(self.base),
                    str(self.lora),
                    self.commands[rank],
                    self.results,
                    self.ack,
                ),
                daemon=True,
            )
            for rank in range(world_size)
        ]
        try:
            for process in self.processes:
                process.start()
            message = self._receive(timeout=300)
            if message[0] != "ready":
                raise RuntimeError(f"TPP startup failed: {message}")
        except BaseException:
            self.shutdown()
            raise

    def _receive(self, timeout=300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self.results.get(timeout=1)
                if message[0] == "error":
                    raise RuntimeError(message[1])
                return message
            except queue.Empty:
                if any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("A TPP worker exited; see the service log")
        raise TimeoutError("Timed out waiting for the native TPP workers")

    def start(self, *, image, audio, pose, prompt, negative_prompt, seed, max_chunks):
        self.close()
        if not self.processes:
            self._launch()
        waveform, rate = sf.read(audio, dtype="float32")
        self.audio = playback_audio(waveform, rate)
        self.offset = 0
        self.pending_ack = False
        job = {
            "image": str(image),
            "audio": str(audio),
            "pose": str(pose) if pose else None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "max_chunks": max_chunks,
        }
        for commands in self.commands:
            commands.put(job)
        self.active = True

    def next(self):
        if not self.active:
            raise RuntimeError("No active take")
        if self.pending_ack:
            self.ack.put(True)
            self.pending_ack = False
        message = self._receive()
        if message[0] == "end":
            self.active = False
            return None
        if message[0] != "chunk":
            raise RuntimeError(f"Unexpected worker message: {message[0]}")
        video = np.load(Path(self.directory.name) / "chunk.npy", allow_pickle=False)
        self.pending_ack = True
        count = len(video) * OUTPUT_SAMPLE_RATE // 25
        audio = self.audio[self.offset : self.offset + count]
        self.offset += count
        return video, np.pad(audio, (0, count - len(audio)))[None, :]

    def close(self):
        # Normal completion retains weights. Cancellation terminates only this
        # backend's workers to unblock native NCCL send/recv safely.
        if self.active:
            self.shutdown()
        self.audio = np.zeros(0, np.float32)
        self.offset = 0

    def shutdown(self):
        self.active = False
        self.pending_ack = False
        for process in self.processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
        self.processes = []
        for channel in (*self.commands, self.results, self.ack):
            channel.cancel_join_thread()
            channel.close()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
