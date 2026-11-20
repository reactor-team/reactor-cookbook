"""Transport-only test fixture. Never represents learned LiveAvatar output."""

import numpy as np
import soundfile as sf
from reactor_runtime.serve import main

from liveavatar_audio import OUTPUT_SAMPLE_RATE, playback_audio
from liveavatar_model import LiveAvatarModel
from liveavatar_pipeline import LiveAvatar


class Surrogate:
    def start(self, **kwargs):
        self.index = 0
        self.limit = kwargs["max_chunks"]
        audio, rate = sf.read(kwargs["audio"], dtype="float32")
        self.audio = playback_audio(audio, rate)
        self.offset = 0

    def next(self):
        if self.index >= self.limit:
            return None
        count = 45 if self.index == 0 else 48
        self.index += 1
        samples = count * OUTPUT_SAMPLE_RATE // 25
        audio = self.audio[self.offset : self.offset + samples]
        self.offset += samples
        audio = np.pad(audio, (0, samples - len(audio)))
        return np.full((count, 64, 64, 3), self.index * 30 % 256, np.uint8), audio[
            None, :
        ]

    def close(self):
        pass


class CPURunner(LiveAvatarModel):
    def shutdown(self):
        self.reset()


def start_cpu_engine(self):
    if self._engine is None:
        self._engine = CPURunner()
        self._engine._backend = Surrogate()


def load(self, config_path=None):
    from liveavatar_assets import configure_cache_environment

    self._work = configure_cache_environment()
    self._start_engine()


if __name__ == "__main__":
    LiveAvatar._start_engine = start_cpu_engine
    LiveAvatar.load = load
    main()
