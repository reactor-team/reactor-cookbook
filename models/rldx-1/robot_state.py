# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Robot proprio state: how it arrives, and how it becomes model input.

The pure seam between the wire and the observation the policy sees — state JSON
in, the ``{"state.<key>": (1, 1, D)}`` vectors ``RLDXSimPolicyWrapper.get_action``
expects out. No torch, no runtime imports, so it is testable on a bare checkout
(same role ``model_schema.py`` plays for the handshake).

The state arrives as **video frame metadata**: the client tags every view's frame
with the JSON it used to send as the ``state_json`` field, so the proprio is
attached to the exact frames it was read with instead of being a separate stream
a receiver has to pair up. The pipeline aligns frames first, then
:class:`FrameStateTags` ranks only tag candidates from that commit; the
``state_json`` field stays as the fallback for a client whose SDK cannot tag
frames (the browser harness's JS SDK has no per-frame metadata API), and
:meth:`RLDXPipeline._resolve_state` prefers the committed tag.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import numpy as np

# Reserved optional keys a tagging client may embed in the state-JSON object
# alongside the state vectors. ``capture_us`` is the microsecond, on the
# client's own clock, of the snapshot the state and the frames were read from;
# ``seq`` is the client's tick counter. Both are integers and both optional —
# embedding either is what buys stamp-ordered freshness across the view streams
# (see :class:`FrameStateTags`). ``parse_state`` reads only the ``state_dims``
# keys, so these ride along in the same object without disturbing it.
CAPTURE_US_KEY = "capture_us"
SEQ_KEY = "seq"
STATE_TAG_KEYS = (CAPTURE_US_KEY, SEQ_KEY)

# Distinct tag byte-strings whose embedded stamp is remembered. In rldx all
# three views of a tick carry byte-identical tags, so a handful of entries turns
# the per-view re-decode into one decode per tick; past that the memo is dropped
# whole rather than evicted one by one — a client that never repeats its bytes
# would only pay for the bookkeeping.
_STAMP_MEMO_MAX = 8


def _as_int(val: object) -> int | None:
    """A JSON number as an ``int``, or ``None`` for anything else."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return int(val)


def _extract_stamp(tag: bytes) -> tuple[int | None, int | None]:
    """The ``(capture_us, seq)`` a tag embeds; either or both may be ``None``.

    Decoding is as lenient as :func:`parse_state`'s: bytes that are not the
    state JSON simply embed no stamp, which drops the tag to the transport's
    ordering rather than raising out of the inference loop.
    """
    try:
        raw = json.loads(tag.decode("utf-8", "replace"))
    except (ValueError, TypeError):
        return (None, None)
    if not isinstance(raw, dict):
        return (None, None)
    return (_as_int(raw.get(CAPTURE_US_KEY)), _as_int(raw.get(SEQ_KEY)))


def zero_state(state_dims: Mapping[str, int]) -> dict[str, np.ndarray]:
    """Zero-filled state vectors, one ``(1, 1, D)`` array per ``state_dims`` key."""
    return {
        f"state.{key}": np.zeros((1, 1, dim), dtype=np.float32)
        for key, dim in state_dims.items()
    }


def parse_state(state_json: str, state_dims: Mapping[str, int]) -> dict[str, np.ndarray] | None:
    """Parse the robot proprio-state JSON into model state vectors.

    Returns ``None`` (state invalid) if the payload is empty, is not valid JSON,
    is not an object, or is missing any required ``state_dims`` key / has a key
    of the wrong length or non-numeric contents. On success returns
    ``{"state.<key>": np.ndarray(1, 1, dim) float32}``.

    This never silently substitutes zeros — the caller owns the fallback
    decision (see :meth:`RLDXPipeline._resolve_state`).
    """
    if not state_json:
        return None
    try:
        raw = json.loads(state_json)
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, np.ndarray] = {}
    for key, dim in state_dims.items():
        val = raw.get(key)
        if val is None:
            return None
        try:
            arr = np.asarray(val, dtype=np.float32).reshape(-1)
        except (ValueError, TypeError):
            return None
        if arr.shape[0] != dim:
            return None
        out[f"state.{key}"] = arr.reshape(1, 1, dim)
    return out


class FrameStateTags:
    """The freshest proprio tag across the inbound view streams.

    Freshest is an ordering key per tag, in three tiers, because the answers to
    "when was this read?" are not equally trustworthy:

    2. a stamp the tag itself embeds — ``seq`` when present, else ``capture_us``
       (:data:`STATE_TAG_KEYS`). The client's own tick counter or its own clock,
       so it orders the views by the snapshot they were read from no matter what
       order the three streams deliver them in, and it is the only "when" that
       names a moment on the client's timeline.
    1. the frame's ``capture_time_us`` — the value the sender passed to
       ``push_frame``, carried verbatim from reactor-sdk 1.1.1 on (1.1.0 and
       earlier discarded it and stamped the engine's own send instant, so a
       tick's frames landed a few hundred microseconds apart instead of sharing
       one number). A tagging client that stamps a tick once therefore has every
       view of that tick agreeing exactly, which is what lets the window be
       assembled from one declared instant rather than from three nearby ones.
       Still a reading of the sender's clock, so differences are meaningful and
       the absolute value is only interpretable to whoever minted it.
    0. arrival order — the tag says nothing about when and the transport carried
       no stamp, so the last tagged frame offered wins.

    A higher tier always outranks a lower one: a client that embeds a stamp is
    never overruled by a wire stamp of a staler snapshot. Within a tier the
    larger value wins, and equal values go to the later arrival.

    The parse is cached against the bytes: three views at the control rate means
    the same JSON arrives ~60 times a second, and only a change in the tag can
    change the state it decodes to. A tag that fails to parse is cached the same
    way — a broken tag stays broken, so re-parsing it every tick buys nothing.
    The embedded stamp is memoized against the bytes for the same reason.
    """

    def __init__(self) -> None:
        self._tag: bytes | None = None
        self._key: tuple[int, int, int] | None = None
        self._arrivals = 0
        self._stamps: dict[bytes, tuple[int | None, int | None]] = {}
        self._parsed_tag: bytes | None = None
        self._parsed: dict[str, np.ndarray] | None = None

    def offer(self, tag: bytes | None, capture_time_us: int | None = None) -> None:
        """Take one frame's metadata as a candidate for freshest tag.

        ``capture_time_us`` is that frame's transport stamp, used only when the
        tag embeds no stamp of its own.

        An untagged candidate leaves another tagged candidate in the same
        selection standing: a frame with no metadata says nothing about the
        robot. The pipeline clears the selection between aligned commits so an
        older commit's tag cannot leak into a newer one.
        """
        if not tag:
            return
        self._arrivals += 1
        capture_us, seq = self._stamp_of(tag)
        if seq is not None or capture_us is not None:
            key = (2, seq if seq is not None else capture_us, self._arrivals)
        elif capture_time_us is not None:
            key = (1, int(capture_time_us), self._arrivals)
        else:
            key = (0, self._arrivals, self._arrivals)
        if self._key is None or key > self._key:
            self._tag = tag
            self._key = key

    @property
    def stamp(self) -> tuple[int | None, int | None]:
        """The freshest tag's embedded ``(capture_us, seq)``.

        ``(None, None)`` when no tag has been seen, or when the one standing
        embeds neither — the client is then not asking for anything to be tied
        back to its timeline.
        """
        if self._tag is None:
            return (None, None)
        return self._stamp_of(self._tag)

    def _stamp_of(self, tag: bytes) -> tuple[int | None, int | None]:
        if tag not in self._stamps:
            if len(self._stamps) >= _STAMP_MEMO_MAX:
                self._stamps.clear()
            self._stamps[tag] = _extract_stamp(tag)
        return self._stamps[tag]

    def parse(self, state_dims: Mapping[str, int]) -> dict[str, np.ndarray] | None:
        """The freshest tag as model state vectors.

        ``None`` when there is nothing usable — no tag seen yet, or the newest
        one is not the state JSON this checkpoint expects. Decoding is lenient
        (``errors="replace"``) so non-UTF-8 bytes fail the JSON parse and land in
        the caller's fallback instead of raising out of the inference loop.
        """
        if self._tag is None:
            return None
        if self._tag != self._parsed_tag:
            self._parsed_tag = self._tag
            self._parsed = parse_state(self._tag.decode("utf-8", "replace"), state_dims)
        return self._parsed

    def clear(self) -> None:
        """Forget the current tag selection and its parse/stamp caches."""
        self._tag = None
        self._key = None
        self._arrivals = 0
        self._stamps.clear()
        self._parsed_tag = None
        self._parsed = None
