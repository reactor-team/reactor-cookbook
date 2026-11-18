"""Lossless disk-backed latent history for native midpoint recomputation."""

import errno
import tempfile
from pathlib import Path

import numpy as np

from movieroll_model import HistoryFullError


class LatentHistory:
    """Native latent chunks without retaining an ever-growing GPU/CPU list.

    MovieRoll revisits the midpoint of every event, including older events.
    A byte archive preserves those exact latents; only selected chunks return
    to the device. The directory is removed on reset and session teardown.
    """

    def __init__(self, root: Path, *, max_bytes: int):
        self._temporary = tempfile.TemporaryDirectory(prefix="movieroll-", dir=root)
        self.path = Path(self._temporary.name) / "history.bin"
        self._file = self.path.open("w+b")
        self._limit = max_bytes
        self._count = 0
        self._shape: tuple[int, ...] | None = None
        self._first_bytes = self._regular_bytes = 0
        self._first_dtype: np.dtype | None = None
        self._regular_dtype: np.dtype | None = None

    def __len__(self):
        return self._count

    def append(self, chunk: np.ndarray) -> None:
        if chunk.dtype not in (np.uint16, np.float32) or chunk.ndim != 5:
            raise ValueError("History requires BF16 bits or FP32 [B,C,T,H,W] chunks")
        shape = tuple(chunk.shape)
        if self._shape is None:
            if shape[2] not in (1, 5):
                raise ValueError(
                    "The first chunk must contain one or five latent frames"
                )
            self._shape = shape
            self._first_bytes = chunk.nbytes
            self._first_dtype = chunk.dtype
        elif shape != (*self._shape[:2], 4, *self._shape[3:]):
            raise ValueError(
                "Continuation chunks must contain four matching latent frames"
            )
        if self._count == 1:
            self._regular_bytes = chunk.nbytes
            self._regular_dtype = chunk.dtype
        elif self._count > 1 and chunk.dtype != self._regular_dtype:
            raise ValueError("Continuation chunks must retain their original dtype")
        payload = np.ascontiguousarray(chunk)
        offset = (
            0
            if self._count == 0
            else self._first_bytes + (self._count - 1) * self._regular_bytes
        )
        if offset + payload.nbytes > self._limit:
            raise HistoryFullError(
                "Temporary video history reached its configured storage budget"
            )
        try:
            self._file.seek(offset)
            self._file.write(payload.tobytes())
            self._file.flush()
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                raise HistoryFullError(
                    "Temporary video history storage is full"
                ) from exc
            raise
        self._count += 1

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        assert self._shape is not None
        offset = (
            0 if index == 0 else self._first_bytes + (index - 1) * self._regular_bytes
        )
        length = self._first_bytes if index == 0 else self._regular_bytes
        self._file.seek(offset)
        data = self._file.read(length)
        if len(data) != length:
            raise OSError("Incomplete latent history")
        shape = self._shape if index == 0 else (*self._shape[:2], 4, *self._shape[3:])
        dtype = self._first_dtype if index == 0 else self._regular_dtype
        array = np.frombuffer(data, dtype=dtype).copy().reshape(shape)
        return array

    def close(self) -> None:
        self._file.close()
        self._temporary.cleanup()
