"""Measure the model half offline: render chunks from a speech file, one call per chunk, and time them.

python bench.py --weights /path/to/aura-weights --portrait portrait.jpg --audio speech.wav --chunks 30
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from aura_assets import prepare_weights
from aura_model import BLOCK_SAMPLES, SAMPLE_RATE, AuraModel
from avatar_contract import AvatarInput


def read_speech(path: Path) -> np.ndarray:
    """Mono int16 at 24 kHz, the rate the avatar receives speech at."""
    import librosa

    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--portrait", type=Path, default=Path("portrait.jpg"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    portrait = args.portrait.read_bytes()
    model = AuraModel()
    start = time.perf_counter()
    model.load(
        prepare_weights(args.weights),
        compile_dit=not args.no_compile,
        warmup_avatar=portrait,
    )
    print(f"loaded and warmed up in {time.perf_counter() - start:.1f} s")

    speech = read_speech(args.audio)
    times = []
    for n in range(args.chunks):
        audio = speech[n * BLOCK_SAMPLES : (n + 1) * BLOCK_SAMPLES]
        audio = np.pad(audio, (0, BLOCK_SAMPLES - audio.size))
        input = AvatarInput(
            audio=audio, silent=False, listening=False, avatar=portrait, avatar_id=1
        )
        start = time.perf_counter()
        model.generate(input)
        times.append(time.perf_counter() - start)
    steady = times[1:]
    median = statistics.median(steady)
    print(f"per chunk: median {median * 1000:.0f} ms, max {max(steady) * 1000:.0f} ms")
    print(
        f"real-time factor: {median / (BLOCK_SAMPLES / SAMPLE_RATE):.2f} (a chunk plays for 1.0 s)"
    )


if __name__ == "__main__":
    main()
