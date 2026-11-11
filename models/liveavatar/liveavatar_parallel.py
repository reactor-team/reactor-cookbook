"""Demand-driven native TPP continuation inside Runtime-owned GPU ranks.

A bounded thread queue preserves the upstream generator's live CUDA/KV state.
A separate CPU process group carries results without mixing its collectives
with the native NCCL pipeline. This module never starts or kills processes.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import timedelta

import numpy as np
import soundfile as sf

from liveavatar_audio import OUTPUT_SAMPLE_RATE, playback_audio
from liveavatar_model import LiveAvatarSettings


class ParallelBackend:
    def __init__(self, settings: LiveAvatarSettings, rank: int, world_size: int):
        import torch.distributed as dist
        from liveavatar_parallel_worker import load_rank
        from liveavatar_turbo import turbo_plan

        plan = turbo_plan(settings.turbo)
        if world_size != plan["world_size"]:
            raise ValueError("Runtime rank count does not match the native TPP profile")
        self.rank = rank
        self.output_rank = plan["output_rank"]
        self._dist = dist
        self._group = dist.new_group(backend="gloo", timeout=timedelta(seconds=600))
        self._model, self._plan = load_rank(rank, settings)
        self._thread: threading.Thread | None = None
        self._results = queue.Queue(maxsize=1)
        self._ack = queue.Queue(maxsize=1)
        self.active = False
        self.pending_ack = False
        self.audio = np.zeros(0, np.float32)
        self.offset = 0

    def start(self, *, image, audio, pose, prompt, negative_prompt, seed, max_chunks):
        self.close()
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
        self._results = queue.Queue(maxsize=1)
        self._ack = queue.Queue(maxsize=1)
        self.active = True
        self._thread = threading.Thread(target=self._produce, args=(job,), daemon=True)
        self._thread.start()

    def _produce(self, job):
        from liveavatar_parallel_worker import generate_take

        try:
            generate_take(self._model, self._plan, self.rank, job, self._deliver)
            # All native sends/receives must finish before a new take can start.
            self._dist.barrier()
            if self.rank == self.output_rank:
                self._results.put(("end", None))
        except Exception as error:
            self._results.put(("error", str(error) or type(error).__name__))

    def _deliver(self, frames):
        self._results.put(("chunk", frames))
        self._ack.get()

    def next(self):
        if not self.active:
            raise RuntimeError("No active take")
        payload = [None]
        if self.rank == self.output_rank:
            if self.pending_ack:
                self._ack.put(True)
                self.pending_ack = False
        # A failed DiT producer must reach the app even while the VAE producer
        # is blocked in native NCCL recv. Only small status objects are polled.
        while True:
            try:
                payload[0] = self._results.get_nowait()
            except queue.Empty:
                payload[0] = ("wait", None)
            kind, value = payload[0]
            statuses = [None] * self._dist.get_world_size(self._group)
            self._dist.all_gather_object(
                statuses, (kind, value if kind == "error" else None), group=self._group
            )
            errors = [message for status, message in statuses if status == "error"]
            if errors:
                raise RuntimeError("; ".join(errors))
            if statuses[self.output_rank][0] != "wait":
                break
            time.sleep(0.05)
        self._dist.broadcast_object_list(
            payload, src=self.output_rank, group=self._group
        )
        kind, video = payload[0]
        if kind == "end":
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Native producer did not finish")
            self.active = False
            return None
        if kind != "chunk":
            raise RuntimeError(f"Unexpected native result: {kind}")
        self.pending_ack = self.rank == self.output_rank
        count = len(video) * OUTPUT_SAMPLE_RATE // 25
        audio = self.audio[self.offset : self.offset + count]
        self.offset += count
        return video, np.pad(audio, (0, count - len(audio)))[None, :]

    def close(self):
        if self.active:
            raise RuntimeError("Stop an active take through the Runtime runner")
        self.audio = np.zeros(0, np.float32)
        self.offset = 0
        self.pending_ack = False
