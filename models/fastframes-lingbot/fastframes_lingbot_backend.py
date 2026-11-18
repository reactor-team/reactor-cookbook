"""Drive the published LingBot pipeline, camera and pixel decoder."""

import sys

import numpy as np
from PIL import Image
from torch import inference_mode

from fastframes_lingbot_conditioning import anchor_condition
from fastframes_lingbot_model import CHUNKS, HEIGHT, WIDTH, Anchor, CameraAction, Settings


class NativeBackend:
    def __init__(self, settings: Settings) -> None:
        sys.path.insert(0, str(settings.fastframes_source))
        from apps.lingbot.conditioning import CHUNK, KeyboardCamera, LiveCamera, text_context
        from apps.lingbot.pipeline import MAX_LATENTS, FastFramesLingBot, configure_backends

        class PitchCamera(KeyboardCamera):
            def _apply(self, action):
                super()._apply(action)
                if action in ("pitch_up", "pitch_down"):
                    # Give the vertical arrows a more visible turn per block.
                    super()._apply(action)

        configure_backends()
        self.pipe = FastFramesLingBot.from_pretrained(
            "lingbot-1.3b",
            lingbot_root=settings.lingbot_source,
            lingbot_checkpoint=settings.dit,
            checkpoint_dir=settings.networks,
            assets_dir=settings.shared,
        )
        self.keyboard = PitchCamera()
        self.camera = LiveCamera()
        self.idle_rays = self.camera.rays(KeyboardCamera().chunk("idle"))
        self.text_context = text_context
        self.latents = CHUNKS * CHUNK
        if self.latents > MAX_LATENTS:
            raise ValueError("The configured world length exceeds the native rotary position table")
        self.warmup_latents = 8 * CHUNK

    @property
    def chunk(self) -> int:
        return self.pipe.chunk

    def warmup(self) -> None:
        # Visit the native cache lengths and both decoder shapes before serving
        # user images. The warmup frames never leave the worker.
        self.pipe.start(Image.new("RGB", (WIDTH, HEIGHT)), "", seed=0, latents=self.warmup_latents, noise="chunk")
        for _ in range(2):
            self.keyboard.reset()
            self.camera.reset()
            for _ in range(8):
                self.pipe.step(self.idle_rays)
            self.pipe.restart(seed=0)
        self.reset()

    def start(self, anchor: Anchor, prompt: str) -> None:
        self.set_prompt(prompt)
        vae = self.pipe.upstream.vae
        self.pipe.condition = anchor_condition(vae, Image.fromarray(anchor.image), self.latents, self.pipe.device)
        self.pipe.latents = self.latents
        self.pipe.restart(seed=anchor.seed, noise="chunk")
        self.keyboard.reset()
        self.camera.reset()

    @inference_mode()
    def set_prompt(self, prompt: str) -> None:
        if prompt != self.pipe.prompt:
            context = self.text_context(self.pipe.upstream.text_encoder, prompt, self.pipe.device)
            self.pipe.dit.set_context(context)
            self.pipe.prompt = prompt

    def step(self, action: CameraAction) -> np.ndarray:
        poses = self.keyboard.chunk(action)
        if action == "idle":
            rays = self.idle_rays
        elif action in ("yaw_left", "yaw_right", "pitch_up", "pitch_down"):
            # Shift the origin to the fixed camera position during a rotation.
            # Otherwise TF32 roundoff creates tiny translations that the native
            # relative-pose code normalizes to full motion.
            position = self.camera.previous[:3, 3].clone()
            self.camera.previous[:3, 3].zero_()
            poses[:, :3, 3].zero_()
            try:
                rays = self.camera.rays(poses)
            finally:
                self.camera.previous[:3, 3].copy_(position)
        else:
            rays = self.camera.rays(poses)
        frames = self.pipe.step(rays)
        return frames.cpu().numpy()

    def reset(self) -> None:
        self.pipe.condition = None
        self.pipe.prompt = None
        self.pipe.generator = None
        self.keyboard.reset()
        self.camera.reset()
