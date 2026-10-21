from itertools import pairwise

import numpy as np
import pytest

from liveavatar_audio import playback_audio


def test_playback_preserves_pitch_duration_and_chunk_continuity():
    source = (0.3 * np.sin(2 * np.pi * 1000 * np.arange(16000 * 6) / 16000)).astype(
        np.float32
    )
    output = playback_audio(source, 16000)
    assert output.dtype == np.float32
    assert len(output) == len(source) * 3
    assert np.isfinite(output).all()
    frequency = np.fft.rfftfreq(len(output), 1 / 48000)[
        np.argmax(abs(np.fft.rfft(output)))
    ]
    assert frequency == pytest.approx(1000)
    boundaries = np.cumsum([0, 45 * 1920, 48 * 1920, 48 * 1920])
    chunks = [output[a:b] for a, b in pairwise(boundaries)]
    np.testing.assert_array_equal(np.concatenate(chunks), output[: boundaries[-1]])


@pytest.mark.parametrize(
    "audio,rate",
    [(np.zeros(10), 48000), (np.zeros((2, 10)), 16000), (np.array([np.nan]), 16000)],
)
def test_reject_invalid_audio(audio, rate):
    with pytest.raises(ValueError):
        playback_audio(audio, rate)
