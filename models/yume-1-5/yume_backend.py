"""Faithful single-GPU extraction of YUME-5B's rolling-latent sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from yume_assets import YumeConfig
from yume_types import Movement, View

MOVEMENT_TEXT: dict[Movement, str] = {
    "none": "The camera's movement direction remains stationary (·).",
    "forward": "The camera pushes forward (W).",
    "backward": "The camera pulls back (S).",
    "left": "The camera moves to the left (A).",
    "right": "The camera moves to the right (D).",
    "forward_left": "The camera pushes forward and moves to the left (W+A).",
    "forward_right": "The camera pushes forward and moves to the right (W+D).",
    "backward_left": "The camera pulls back and moves to the left (S+A).",
    "backward_right": "The camera pulls back and moves to the right (S+D).",
}
VIEW_TEXT: dict[View, str] = {
    "none": "The rotation direction of the camera remains stationary (·).",
    "pan_left": "The camera pans to the left (←).",
    "pan_right": "The camera pans to the right (→).",
    "tilt_up": "The camera tilts up (↑).",
    "tilt_down": "The camera tilts down (↓).",
    "tilt_up_left": "The camera tilts up and pans to the left (↑←).",
    "tilt_up_right": "The camera tilts up and pans to the right (↑→).",
    "tilt_down_left": "The camera tilts down and pans to the left (↓←).",
    "tilt_down_right": "The camera tilts down and pans to the right (↓→).",
}


def conditioned_prompt(prompt: str, movement: Movement, view: View) -> str:
    """Encode controls in the caption format used to train and sample YUME."""
    distance = 0 if movement == "none" else 4
    rotation = 0 if view == "none" else 4
    return " ".join(
        (
            "First-person perspective.",
            MOVEMENT_TEXT[movement],
            VIEW_TEXT[view],
            f"Actual distance moved:{distance} at 100 meters per second.",
            f"Angular change rate (turn speed):{rotation}.",
            f"View rotation speed:{rotation}.",
            prompt.strip(),
        )
    )


class YumeBackend:
    """Own YUME weights and its clean rolling latent/pixel continuation state."""

    def __init__(self, config: YumeConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("YUME-5B requires CUDA")
        from wan23 import Yume
        from wan23.configs import WAN_CONFIGS

        self.config = config
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.wan = Yume(
            config=WAN_CONFIGS["ti2v-5B"],
            checkpoint_dir=str(config.checkpoint_path),
            device_id=torch.cuda.current_device(),
        )
        self.transformer = (
            self.wan.model.to(self.device, dtype=self.dtype)
            .eval()
            .requires_grad_(False)
        )
        self.wan.vae.model.to(self.device)
        for parameter in self.wan.vae.model.parameters():
            parameter.data = parameter.data.to(self.dtype)
        self.model_input_latent: torch.Tensor | None = None
        self.model_input_pixels: torch.Tensor | None = None
        self.mode: str | None = None
        self.chunk_index = 0
        self.generator = torch.Generator(device=self.device)

    @torch.inference_mode()
    def reset(
        self,
        *,
        image: Path | None,
        video: Path | None = None,
        prompt: str,
        seed: int,
        movement: Movement,
        view: View,
    ) -> None:
        """Initialize image or text mode while preserving upstream 32/8 context semantics."""
        self.generator.manual_seed(seed)
        self.model_input_latent = None
        self.model_input_pixels = None
        self.mode = (
            "video_to_video"
            if video is not None
            else ("image_to_video" if image is not None else "text_to_video")
        )
        self.chunk_index = 0
        if image is None and video is None:
            return
        if video is not None:
            visible = self._load_video(video)
        else:
            assert image is not None
            rgb = (
                Image.open(image)
                .convert("RGB")
                .resize(
                    (self.config.width, self.config.height), Image.Resampling.BILINEAR
                )
            )
            pixels = (
                torch.from_numpy(np.asarray(rgb).copy())
                .permute(2, 0, 1)
                .float()
                .div(127.5)
                .sub(1)
                .to(self.device)
            )
            visible = torch.zeros(
                (3, 33, self.config.height, self.config.width), device=self.device
            )
            visible[:, 0] = pixels
        model_pixels = torch.cat(
            [visible[:, :1].repeat(1, 16, 1, 1), visible[:, :33]], dim=1
        )
        with torch.autocast("cuda", dtype=self.dtype):
            first = self.wan.vae.encode(
                [model_pixels[:, : -self.config.frames_per_chunk]]
            )[0]
            second = self.wan.vae.encode(
                [model_pixels[:, -self.config.frames_per_chunk :]]
            )[0]
        self.model_input_pixels = model_pixels
        self.model_input_latent = torch.cat([first, second], dim=1)

    def _load_video(self, path: Path) -> torch.Tensor:
        """Match sample_5b.py's first 33 frames at its assumed 30 FPS."""
        import av

        with av.open(str(path)) as container:
            images = [frame.to_image() for frame in container.decode(video=0)]
        if len(images) < 33:
            raise ValueError("YUME video conditioning requires at least 33 frames")
        tensors = []
        for image in images[:33]:
            rgb = image.convert("RGB").resize(
                (self.config.width, self.config.height), Image.Resampling.BICUBIC
            )
            tensors.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1))
        return (
            torch.stack(tensors)
            .permute(1, 0, 2, 3)
            .float()
            .div(127.5)
            .sub(1)
            .to(self.device)
        )

    @torch.inference_mode()
    def generate_chunk(
        self, *, prompt: str, movement: Movement, view: View
    ) -> tuple[np.ndarray, str]:
        """Denoise only the newest eight latents, decode one 32-frame chunk, and retain clean context."""
        from wan23.utils.utils import masks_like

        text = conditioned_prompt(prompt, movement, view)
        latent_frames = self.config.latent_frames_per_chunk
        if self.mode == "text_to_video" and self.model_input_latent is None:
            arg_c, _arg_null, noise = self.wan.generate(
                text,
                frame_num=self.config.frames_per_chunk,
                max_area=self.config.width * self.config.height,
                latent_frame_zero=latent_frames,
                sampling_steps=self.config.sample_steps,
                shift=self.config.shift,
                seed=self.generator.initial_seed(),
                offload_model=False,
            )
            latent = noise.to(self.dtype)
            mask2 = None
        else:
            if self.model_input_latent is None:
                raise RuntimeError("YUME rollout was not initialized")
            continuing = self.chunk_index > 0
            frame_num = (
                (self.model_input_latent.shape[1] - 1) * 4
                + 1
                + self.config.frames_per_chunk
                if continuing
                else self.model_input_pixels.shape[1]
            )
            img = (
                self.model_input_latent
                if continuing
                else self.model_input_latent[:, :-latent_frames]
            )
            arg_c, _arg_null, noise, mask2, _img = self.wan.generate(
                text,
                img=img,
                frame_num=frame_num,
                max_area=self.config.width * self.config.height,
                latent_frame_zero=latent_frames,
                sampling_steps=self.config.sample_steps,
                shift=self.config.shift,
                seed=self.generator.initial_seed(),
                offload_model=False,
            )
            if continuing:
                fresh = torch.randn(
                    (
                        self.wan.vae.model.z_dim,
                        self.model_input_latent.shape[1] + latent_frames,
                        self.model_input_latent.shape[2],
                        self.model_input_latent.shape[3],
                    ),
                    generator=self.generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                latent = torch.cat(
                    [self.model_input_latent, fresh[:, -latent_frames:]], dim=1
                )
                _mask1, mask2 = masks_like(
                    [latent], zero=True, latent_frame_zero=latent_frames
                )
            else:
                latent = torch.cat(
                    [
                        self.model_input_latent[:, :-latent_frames],
                        noise[:, -latent_frames:],
                    ],
                    dim=1,
                ).to(self.dtype)

        sigmas = _sampling_sigmas(self.config.sample_steps, self.config.shift)
        for index, sigma in enumerate(sigmas):
            timestep = torch.tensor([sigma * 1000], device=self.device)
            if mask2 is None:
                tvec = timestep
            else:
                clean_ts = mask2[0][0][:-latent_frames, ::2, ::2].flatten()
                tvec = torch.cat(
                    [
                        clean_ts,
                        clean_ts.new_ones(arg_c["seq_len"] - clean_ts.numel())
                        * timestep,
                    ]
                ).unsqueeze(0)
            with torch.autocast("cuda", dtype=self.dtype):
                kwargs = {"flag": False} if mask2 is None else {}
                prediction = self.transformer(
                    [latent], t=tvec, latent_frame_zero=latent_frames, **kwargs, **arg_c
                )[0]
            next_sigma = 0.0 if index + 1 == len(sigmas) else sigmas[index + 1]
            tail = (
                latent[:, -latent_frames:]
                + (next_sigma - sigma) * prediction[:, -latent_frames:]
            )
            latent = torch.cat([latent[:, :-latent_frames], tail], dim=1)

        with torch.autocast("cuda", dtype=self.dtype):
            video = self.wan.vae.decode([latent[:, -latent_frames:]])[0][
                :, -self.config.frames_per_chunk :
            ]
        if self.model_input_latent is None:
            self.model_input_latent = latent[:, -latent_frames:]
            self.model_input_pixels = video[:, -self.config.frames_per_chunk :]
        elif self.chunk_index == 0:
            self.model_input_latent = torch.cat(
                [
                    self.model_input_latent[:, :-latent_frames],
                    latent[:, -latent_frames:],
                ],
                dim=1,
            )
            self.model_input_pixels = torch.cat(
                [
                    self.model_input_pixels[:, : -self.config.frames_per_chunk],
                    video[:, -self.config.frames_per_chunk :],
                ],
                dim=1,
            )
        else:
            self.model_input_latent = torch.cat(
                [self.model_input_latent, latent[:, -latent_frames:]], dim=1
            )
            self.model_input_pixels = torch.cat(
                [self.model_input_pixels, video[:, -self.config.frames_per_chunk :]],
                dim=1,
            )
        self.chunk_index += 1
        frames = (
            video.clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .byte()
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )
        return frames, text

    def end_session(self) -> None:
        self.model_input_latent = None
        self.model_input_pixels = None
        self.mode = None
        self.chunk_index = 0
        torch.cuda.empty_cache()


def _sampling_sigmas(steps: int, shift: float) -> np.ndarray:
    sigma = np.linspace(1, 0, steps + 1)[:steps]
    return shift * sigma / (1 + (shift - 1) * sigma)
