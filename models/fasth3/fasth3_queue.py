"""The clip queue: every generation a client has asked for, in order.

Pure bookkeeping — no torch, no fastvideo, no runtime imports — so the queue's
behaviour is testable on any machine. ``fasth3.py`` owns when entries move
(builds are submitted and applied on the model's event loop); this module owns
what an entry is and what the queue guarantees: bounded capacity, stable order,
and one wire form (`ClipInfo` in ``fasth3_types.py``) reported everywhere a
clip is mentioned.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import fasth3_clip_plan as clip_plan


@dataclass
class ClipEntry:
    """One enqueued generation, from request to built payload.

    The client-facing fields are frozen at enqueue time: the prompt and
    metadata as the client sent them, and the frame count and seed as the
    session's conditions stood. ``video`` and ``audio`` are filled in when the
    build completes; ``ready`` is derived from their presence.
    """

    clip_id: str
    prompt: str
    metadata: str
    frames: int
    seed: int
    # Set while a build for this entry is in flight, so the scheduler never
    # submits the same entry twice.
    building: bool = False
    # The built payload: decoded RGB frames and the wire-ready waveform.
    video: list[Any] | None = None
    audio: Any = None

    @property
    def ready(self) -> bool:
        """Whether the clip is built and can be played."""
        return self.video is not None

    @property
    def seconds(self) -> float:
        """Exact playout length, derived from the frame count."""
        return clip_plan.seconds_for_frames(self.frames)

    def snapshot(self) -> dict[str, Any]:
        """The clip's wire form — the `ClipInfo` structure, as a plain mapping.

        Every message that references a clip carries this whole structure, so a
        client never has to join an id against an earlier message. A mapping
        rather than a dataclass instance, because the wire encoder accepts only
        JSON-representable values; ``ClipInfo`` in ``fasth3_types.py`` is the
        schema-side declaration of this exact shape.
        """
        return {
            "clip_id": self.clip_id,
            "prompt": self.prompt,
            "metadata": self.metadata,
            "frames": self.frames,
            "seconds": round(self.seconds, 3),
            "seed": self.seed,
            "ready": self.ready,
        }


@dataclass
class ClipQueue:
    """A bounded, ordered queue of :class:`ClipEntry`.

    Order is enqueue order and never changes; builds run through it front to
    back, so ready entries always form a prefix of the pending ones. Capacity
    comes from the deployment config (``inference.queue_size``) — every entry
    can hold a fully built clip in host memory, so the bound is also the memory
    budget.
    """

    capacity: int
    _entries: list[ClipEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"queue capacity must be positive, got {self.capacity}")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry: ClipEntry) -> bool:
        return any(existing is entry for existing in self._entries)

    @property
    def full(self) -> bool:
        """Whether another enqueue would exceed the capacity."""
        return len(self._entries) >= self.capacity

    def enqueue(self, *, prompt: str, metadata: str, frames: int, seed: int) -> ClipEntry:
        """Append one generation request and return its entry.

        Raises:
            ValueError: If the queue is already at capacity.
        """
        if self.full:
            raise ValueError(f"the queue is full ({self.capacity} clips)")
        entry = ClipEntry(
            clip_id=str(uuid.uuid4()),
            prompt=prompt,
            metadata=metadata,
            frames=frames,
            seed=seed,
        )
        self._entries.append(entry)
        return entry

    def get(self, clip_id: str) -> ClipEntry | None:
        """The entry with *clip_id*, or ``None`` when no queued clip has it."""
        for entry in self._entries:
            if entry.clip_id == clip_id:
                return entry
        return None

    def next_to_build(self) -> ClipEntry | None:
        """The oldest entry that is neither built nor being built."""
        for entry in self._entries:
            if not entry.ready and not entry.building:
                return entry
        return None

    def next_ready(self) -> ClipEntry | None:
        """The oldest entry that is built and playable."""
        for entry in self._entries:
            if entry.ready:
                return entry
        return None

    def ready_count(self) -> int:
        """How many queued clips are built and playable right now."""
        return sum(1 for entry in self._entries if entry.ready)

    def remove(self, entry: ClipEntry) -> None:
        """Take *entry* out of the queue; playing a clip consumes its entry."""
        self._entries = [existing for existing in self._entries if existing is not entry]

    def clear(self) -> int:
        """Drop every entry, built payloads included, and return how many."""
        cleared = len(self._entries)
        self._entries = []
        return cleared

    def snapshot(self) -> list[dict[str, Any]]:
        """Every entry's wire form, oldest first — the `queue_update` payload."""
        return [entry.snapshot() for entry in self._entries]


__all__ = ["ClipEntry", "ClipQueue"]
