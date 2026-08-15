# ──────────────────────────────────────────────────────────────────────────
# msgpack codec with numpy support, wire-compatible with openpi-client.
#
# The encoding matches openpi-client's own `msgpack_numpy` and RoboLab's
# `MsgPackNumpy` byte for byte:
#
#   np.ndarray  -> {b"__ndarray__": True, b"data": <raw bytes>,
#                   b"dtype": <dtype.str>, b"shape": <tuple>}
#   np.generic  -> {b"__npgeneric__": True, b"data": <python scalar>,
#                   b"dtype": <dtype.str>}
#
# The marker keys are BYTES on purpose: msgpack round-trips bytes as bin and
# str as str, so bytes keys cannot collide with a caller's own string keys.
# Decoding uses strict_map_key=False to match the client.
#
# Included here rather than imported from openpi-client so this package
# installs with plain `msgpack`; see README.md "Why the protocol server is
# in this package". Derived from the openpi project (Physical Intelligence),
# Apache-2.0.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _encode(obj: Any) -> Any:
    """``default=`` hook: make numpy values msgpack-representable."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _decode(obj: dict) -> Any:
    """``object_hook=``: rebuild numpy values from the marker dicts."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    """Serialise *obj* (numpy-aware) to msgpack bytes."""
    # No strict_types: it breaks tuple serialisation, and shapes are tuples.
    return msgpack.packb(obj, default=_encode)


def unpackb(data: bytes) -> Any:
    """Deserialise msgpack *data* (numpy-aware)."""
    return msgpack.unpackb(data, object_hook=_decode, strict_map_key=False)
