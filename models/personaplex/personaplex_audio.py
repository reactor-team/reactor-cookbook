"""Rate conversion between Reactor's wire audio and PersonaPlex's model audio.

Reactor's WebRTC transport carries mono 16-bit PCM at 48 kHz in both
directions, and PersonaPlex works on mono float32 at 24 kHz — Mimi's native
rate. The ratio is exactly two, so both directions are a fixed half-band
resampling: decimate by two on the way in, interpolate by two on the way out.

Both use the same linear-phase FIR, applied with retained history so the
filter is continuous across frame boundaries. That continuity is the reason
these are objects rather than functions: a conversation is a long stream cut
into 80 ms frames, and a filter restarted on every frame would stamp a
discontinuity into the audio 12.5 times a second.

The constant group delay is ``(taps - 1) / 2`` samples at 48 kHz — 0.65 ms
with the filter below, on each leg. That is well inside the latency budget of
a spoken turn and does not accumulate: it is a fixed offset, not a drift.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

WIRE_SAMPLE_RATE = 48_000
"""Sample rate Reactor's transport delivers and accepts, in Hz."""

MODEL_SAMPLE_RATE = 24_000
"""Sample rate Mimi encodes and decodes at, in Hz."""

_RATIO = WIRE_SAMPLE_RATE // MODEL_SAMPLE_RATE

_INT16_SCALE = 32768.0
_INT16_PEAK = 32767

# 63 taps of windowed sinc, cut off at 11 kHz of the 48 kHz stream. The stop
# band starts below the 12 kHz Nyquist of the 24 kHz side, so decimation
# folds nothing audible back into the speech band; a Kaiser window at
# beta 8.6 puts the stop-band ripple near -90 dB, far under the noise floor
# of a 16-bit wire. Speech carries nothing above 11 kHz worth preserving.
_NUM_TAPS = 63
_CUTOFF_HZ = 11_000.0
_KAISER_BETA = 8.6


def _lowpass_taps() -> npt.NDArray[np.float32]:
    """Return the shared anti-alias / anti-image FIR, normalised to unity DC gain."""
    positions = np.arange(_NUM_TAPS, dtype=np.float64)
    centre = (_NUM_TAPS - 1) / 2.0
    ratio = _CUTOFF_HZ / WIRE_SAMPLE_RATE
    taps = 2.0 * ratio * np.sinc(2.0 * ratio * (positions - centre))
    taps *= np.kaiser(_NUM_TAPS, _KAISER_BETA)
    taps /= taps.sum()
    return taps.astype(np.float32)


_TAPS = _lowpass_taps()


class _StreamingFir:
    """The shared FIR, applied across frames without restarting.

    Holds the ``taps - 1`` trailing samples of the previous call so the next
    one convolves against real history instead of zeros. Feeding ``n`` samples
    always returns ``n`` samples.
    """

    def __init__(self) -> None:
        self._history = np.zeros(_TAPS.size - 1, dtype=np.float32)

    def reset(self) -> None:
        """Forget the retained history, for a stream that starts over."""
        self._history[:] = 0.0

    def process(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Filter *samples*, returning as many samples as were given."""
        if samples.size == 0:
            return samples
        padded = np.concatenate([self._history, samples])
        self._history = padded[padded.size - self._history.size :].copy()
        return np.convolve(padded, _TAPS, mode="valid").astype(np.float32)


class WireToModel:
    """Turns 48 kHz int16 wire audio into 24 kHz float32 model audio.

    Low-passes at the shared cutoff, then keeps every second sample. Input
    length must be even, which the caller guarantees by working in whole
    frames.
    """

    def __init__(self) -> None:
        self._fir = _StreamingFir()

    def reset(self) -> None:
        """Drop the filter history, for a conversation that starts over."""
        self._fir.reset()

    def process(self, wire: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
        """Convert one block of wire samples into model samples.

        Args:
            wire: Mono int16 samples at 48 kHz. The length must be even.

        Returns:
            Mono float32 samples at 24 kHz in ``[-1, 1)``, half as many.

        Raises:
            ValueError: If the block length is not a multiple of the rate ratio.
        """
        if wire.size % _RATIO:
            raise ValueError(
                f"expected a multiple of {_RATIO} wire samples, got {wire.size}"
            )
        scaled = wire.astype(np.float32) / _INT16_SCALE
        return self._fir.process(scaled)[::_RATIO]


class ModelToWire:
    """Turns 24 kHz float32 model audio into 48 kHz int16 wire audio.

    Inserts a zero between successive samples and low-passes to remove the
    images that zero-stuffing creates. Zero-stuffing halves the average
    amplitude, so the filtered result is scaled back by the ratio.
    """

    def __init__(self) -> None:
        self._fir = _StreamingFir()

    def reset(self) -> None:
        """Drop the filter history, for a conversation that starts over."""
        self._fir.reset()

    def process(self, model: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
        """Convert one block of model samples into wire samples.

        Args:
            model: Mono float32 samples at 24 kHz, nominally in ``[-1, 1]``.

        Returns:
            Mono int16 samples at 48 kHz, twice as many, clipped to the int16
            range rather than wrapped — a generated frame can overshoot, and a
            wrap would read as a click.
        """
        stuffed = np.zeros(model.size * _RATIO, dtype=np.float32)
        stuffed[::_RATIO] = model.astype(np.float32) * _RATIO
        filtered = self._fir.process(stuffed)
        return np.clip(filtered * _INT16_SCALE, -_INT16_PEAK - 1, _INT16_PEAK).astype(
            np.int16
        )


class WireFrameAssembler:
    """Cuts a stream of arbitrary inbound blocks into fixed-size frames.

    Inbound audio arrives in whatever blocks the transport decoded — 10 ms at a
    time in practice — while the model consumes exactly one 80 ms frame per
    step. This holds the remainder between reads so no sample is dropped or
    duplicated at a block boundary.

    Args:
        frame_samples: How many samples one frame holds.
    """

    def __init__(self, frame_samples: int) -> None:
        self._frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.int16)

    def reset(self) -> None:
        """Discard whatever is held short of a full frame."""
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def buffered(self) -> int:
        """How many samples are held, short of a full frame."""
        return int(self._pending.size)

    def push(self, block: npt.NDArray[np.int16]) -> None:
        """Append one decoded inbound block, flattening its channel dimension."""
        samples = np.asarray(block, dtype=np.int16).reshape(-1)
        if samples.size:
            self._pending = np.concatenate([self._pending, samples])

    def take(self) -> npt.NDArray[np.int16] | None:
        """Return one whole frame, or ``None`` while fewer samples are held."""
        if self._pending.size < self._frame_samples:
            return None
        frame = self._pending[: self._frame_samples].copy()
        self._pending = self._pending[self._frame_samples :]
        return frame
