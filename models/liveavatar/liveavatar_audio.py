"""Keep native inference audio separate from the WebRTC playback waveform."""

import numpy as np
from scipy.signal import resample_poly

OUTPUT_SAMPLE_RATE = 48000


def playback_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample once per take, avoiding filter discontinuities at clip boundaries."""
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000 or audio.ndim != 1 or not len(audio):
        raise ValueError("Driving audio must be nonempty mono 16 kHz")
    if not np.isfinite(audio).all():
        raise ValueError("Driving audio contains non-finite samples")
    return np.clip(resample_poly(audio, 3, 1), -1, 1).astype(np.float32)
