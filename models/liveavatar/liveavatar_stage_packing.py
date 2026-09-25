"""Stage ownership and independent random streams for the streaming adapter."""

import random


class StageRandomStreams:
    """Preserve native per-rank Python RNG used by conditional RoPE offsets."""

    def __init__(self, stages):
        self.states = dict.fromkeys(stages, random.getstate())

    def call(self, stage, forward, *args, **kwargs):
        outside = random.getstate()
        random.setstate(self.states[stage])
        try:
            return forward(*args, **kwargs)
        finally:
            self.states[stage] = random.getstate()
            random.setstate(outside)


def stage_groups(world_size):
    if world_size not in range(1, 6):
        raise ValueError("Expected one through five GPUs")
    ranks = max(1, world_size - 1)
    # Contiguous stages, balanced as evenly as possible.
    return [
        list(range((rank * 4) // ranks, ((rank + 1) * 4) // ranks))
        for rank in range(ranks)
    ]
