"""Step the pinned FastFrames sampler and Apple's streaming pixel decoder."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from apple_fastframes_model import FOLLOWUP_FRAMES, LATENT_STEPS, VIDEO_FRAMES, Settings


@dataclass
class Rollout:
    noise: Any
    transition: Callable
    visual_kv: list
    text_kv: list | None = None


class NativeBackend:
    def __init__(self, settings: Settings):
        import torch

        # Apple's profiling decoder uses local imports. This path is confined
        # to the isolated model process, where the pinned Apple code is loaded.
        sys.path[:0] = [
            str(settings.source),
            str(settings.source / "evaluation/latency"),
        ]
        from decoders import StreamingPixelDecoder
        from fastframes.pipeline import MODELS, FastFrames, configure_backends, to_uint8
        from fastframes.wan.causal_model import allocate_kv_cache

        spec = MODELS[settings.variant]
        if (spec.config.num_frames, spec.config.num_pixel_frames) != (
            LATENT_STEPS,
            VIDEO_FRAMES,
        ):
            raise ValueError("Unexpected FastFrames rollout geometry")
        configure_backends(spec.config)
        self._pipe = FastFrames.from_pretrained(
            spec.config,
            settings.wan,
            settings.generator,
            settings.apple / spec.refiner,
            settings.apple / spec.decoder,
        )
        h, w = spec.config.latent_size
        self._decoder = StreamingPixelDecoder(self._pipe.decoder, (h // 2, w // 2))
        self._allocate_kv_cache = allocate_kv_cache
        self._to_uint8 = to_uint8
        self._torch = torch
        self._rollout: Rollout | None = None

    def start(self, seed: int) -> None:
        self.reset()
        pipe = self._pipe
        with self._torch.inference_mode(), pipe.autocast():
            noise, transition = pipe._noise(seed)
            visual_kv = self._allocate_kv_cache(
                pipe.dit, pipe.config.num_frames, pipe.frame_tokens, device=pipe.device
            )
        self._rollout = Rollout(noise=noise, transition=transition, visual_kv=visual_kv)

    def set_prompt(self, prompt: str) -> None:
        pipe = self._pipe
        assert self._rollout is not None
        embeds = pipe.encode_prompt(prompt)
        with self._torch.inference_mode(), pipe.autocast():
            self._rollout.text_kv = pipe.dit.text_cache(embeds)

    def step(self, index: int) -> np.ndarray:
        run = self._rollout
        assert run is not None and run.text_kv is not None
        pipe = self._pipe
        torch = self._torch
        with torch.inference_mode(), pipe.autocast():
            # Preserve the order used by FastFrames.rollout, including the
            # first-frame clean context pass and the learned KV refiner.
            config = pipe.config
            schedule = config.first_frame_timesteps if index == 0 else config.timesteps
            noisy = run.noise[:, index : index + 1]
            for step, timestep in enumerate(schedule):
                clean, features, modulation = pipe._predict(
                    noisy, timestep, index, run.text_kv, run.visual_kv
                )
                if step + 1 < len(schedule):
                    next_t = torch.full(
                        (1,),
                        schedule[step + 1],
                        device=pipe.device,
                        dtype=torch.float32,
                    )
                    noisy = pipe.scheduler.add_noise(
                        clean.flatten(0, 1), run.transition(step, index), next_t
                    ).unflatten(0, clean.shape[:2])
            if index == 0:
                pipe._predict(clean, 0.0, index, run.text_kv, run.visual_kv)
            else:
                pipe._refine(clean, index, run.visual_kv)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pixels = self._decoder(features, modulation)[0]
            # Apple aligns 84 decoder frames to 81 by dropping the leading 3.
            if index == 0:
                pixels = pixels[FOLLOWUP_FRAMES - 1 :]
            return self._to_uint8(pixels).cpu().numpy()

    def reset(self) -> None:
        self._rollout = None
        with self._torch.inference_mode():
            self._decoder.reset()
