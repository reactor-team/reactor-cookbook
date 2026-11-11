import random

import pytest

from liveavatar_stage_packing import (
    StageRandomStreams,
    stage_groups,
)


def test_stage_rng_matches_separate_native_processes():
    random.seed(420)
    outside = random.getstate()
    streams = StageRandomStreams(range(4))
    reference = random.Random(420)
    expected = [reference.randint(4, 30) for _ in range(100)]
    observed = [[] for _ in range(4)]
    for _ in range(100):
        for stage in range(4):
            observed[stage].append(streams.call(stage, random.randint, 4, 30))
    assert observed == [expected] * 4
    assert random.getstate() == outside


@pytest.mark.parametrize("gpus", range(1, 6))
def test_all_four_stages_are_preserved(gpus):
    groups = stage_groups(gpus)
    assert [step for group in groups for step in group] == [0, 1, 2, 3]
    assert len(groups) == max(1, gpus - 1)
