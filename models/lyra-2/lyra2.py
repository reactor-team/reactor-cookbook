"""Expose Lyra 2.0's native autoregressive video step through Reactor SDK."""

from __future__ import annotations

import io
import secrets
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError
from reactor_runtime import (ClientInfo, CommandError, InputField, ReactorPipeline,
                             UploadedFile, connected, disconnected, event,
                             session_ended, session_started)

from lyra2_backend import Lyra2Backend
from lyra2_camera import Lyra2CameraPlanner
from lyra2_schema import (CameraChanged, ChunkCompleted, ImageSelected, Lyra2Output,
                          Lyra2State, PromptQueued, ResetQueued, StateUpdate)


class Lyra2(ReactorPipeline):
    """Explore an image-conditioned world through native 80-frame AR updates."""

    state: Lyra2State
    buffer_size = 80

    def __init__(self) -> None:
        super().__init__()
        self.config: dict | None = None
        self.backend: Lyra2Backend | None = None
        self.planner: Lyra2CameraPlanner | None = None
        self.image: UploadedFile | Path | None = None
        self.image_name: str | None = None
        self.active_prompt: str | None = None
        self.seed = 1
        self.chunk = 0
        self.generating = False

    def load(self, config_path: Path | None) -> None:
        if config_path is None:
            raise ValueError("Lyra-2 requires lyra2.yaml")
        self.config = yaml.safe_load(config_path.read_text())
        for key in ("source_path", "output_path", "cache_path"):
            self.config[key] = str(Path(self.config[key]).expanduser().resolve())
            Path(self.config[key]).mkdir(parents=True, exist_ok=True) if key != "source_path" else None
        import os
        cache = self.config["cache_path"]
        for name, value in {
            "HF_HOME": cache, "HUGGINGFACE_HUB_CACHE": f"{cache}/hub",
            "TORCH_HOME": f"{cache}/torch", "XDG_CACHE_HOME": f"{cache}/xdg",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }.items(): os.environ.setdefault(name, value)
        self.backend = Lyra2Backend(self.config)
        self.planner = Lyra2CameraPlanner(
            translation_per_frame=self.config["translation_per_frame"],
            rotation_degrees_per_frame=self.config["rotation_degrees_per_frame"],
        )

    @session_started
    def started(self) -> None:
        self.state.prompt = ""
        self.state._reset_requested = False
        self._clear_motion()

    @session_ended
    def ended(self) -> None:
        if self.backend: self.backend.clear()
        self.image = None
        self.image_name = None

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._state())

    @disconnected
    async def on_disconnected(self) -> None:
        self._clear_motion()

    @event(name="set_image", description=(
        "Select an uploaded image and begin a fresh continuous rollout from it. Valid at any "
        "time; the selected image, prompt, and seed replace the current world when the next "
        "chunk begins. Emits `image_selected` and broadcasts `state_update` on success, or "
        "`command_error` when the upload is missing, unsupported, empty, or larger than 25 MiB."
    ))
    async def set_image(self, image: UploadedFile = InputField(moderate=True, description=(
        "Anchor image uploaded through Reactor. JPEG, PNG, WebP, or BMP up to 25 MiB; it is "
        "resized to 768x448 and becomes active when the fresh rollout begins."
    )), prompt: str = InputField(default="", max_length=4096, moderate=True, description=(
        "Scene description for the fresh rollout, up to 4096 characters. Whitespace is trimmed; "
        "an empty value selects Lyra's default prompt."
    )), seed: int = InputField(default=-1, ge=-1, le=2147483647, description=(
        "Seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain the active seed; a "
        "non-negative value becomes active when the rollout begins."
    ))) -> ImageSelected:
        self._decode(image)
        if seed >= 0: self.seed = seed
        self.image, self.image_name = image, image.name
        self.state.prompt = prompt.strip() or self._cfg()["default_prompt"]
        self._request_reset()
        await self.send(self._state())
        return ImageSelected(filename=image.name, prompt=self.state.prompt, seed=self.seed)

    @event(name="random_image", description=(
        "Select a random built-in image and begin a fresh continuous rollout using its paired "
        "prompt. Valid at any time. Emits `image_selected` and broadcasts `state_update` on "
        "success, or `command_error` when no built-in images are available."
    ))
    async def random_image(self) -> ImageSelected:
        root = Path(self._cfg()["source_path"]) / "assets/samples"
        images = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not images: raise CommandError("image_unavailable", "No bundled Lyra-2 sample images were found.")
        choices = [p for p in images if p != self.image] or images
        self.image = secrets.choice(choices)
        self.image_name = self.image.name
        caption = self.image.with_suffix(".txt")
        self.state.prompt = caption.read_text().strip() if caption.exists() else self._cfg()["default_prompt"]
        self._request_reset()
        await self.send(self._state())
        return ImageSelected(filename=self.image_name, prompt=self.state.prompt, seed=self.seed)

    @event(name="set_prompt", description=(
        "Set the text condition without restarting the current continuous world. Requires a "
        "selected image; the normalized text is sampled when the next chunk begins. Emits "
        "`prompt_queued` and broadcasts `state_update` on success, or `command_error` for empty "
        "text or a missing image."
    ))
    async def set_prompt(self, prompt: str = InputField(max_length=4096, moderate=True, description=(
        "Non-empty scene description, up to 4096 characters. Whitespace is trimmed and the "
        "result is sampled at the next chunk boundary and held for later chunks."
    ))) -> PromptQueued:
        self._require_image()
        value = prompt.strip()
        if not value: raise CommandError("prompt_required", "Lyra-2 requires a non-empty caption.")
        self.state.prompt = value
        applies = self.chunk + 1 + int(self.generating)
        await self.send(self._state())
        return PromptQueued(prompt=value, applies_to_chunk=applies)

    @event(name="set_camera_motion", description=(
        "Set all six held camera axes for the next chunk in one command. Requires a selected "
        "image; values are sampled at the next chunk boundary and held for later chunks. Emits "
        "`camera_changed` and broadcasts `state_update` on success, or `command_error` when no "
        "image is selected."
    ))
    async def set_camera_motion(self,
        forward: float = InputField(default=0, ge=-1, le=1, description="Normalized backward (-1) to forward (1) translation. Zero stops this axis; the value is sampled at the next chunk boundary and held."),
        strafe: float = InputField(default=0, ge=-1, le=1, description="Normalized left (-1) to right (1) translation. Zero stops this axis; the value is sampled at the next chunk boundary and held."),
        vertical: float = InputField(default=0, ge=-1, le=1, description="Normalized down (-1) to up (1) translation. Zero stops this axis; the value is sampled at the next chunk boundary and held."),
        pitch: float = InputField(default=0, ge=-1, le=1, description="Normalized downward (-1) to upward (1) pitch. Zero stops this axis; the value is sampled at the next chunk boundary and held."),
        yaw: float = InputField(default=0, ge=-1, le=1, description="Normalized left (-1) to right (1) yaw. Zero stops this axis; the value is sampled at the next chunk boundary and held."),
        roll: float = InputField(default=0, ge=-1, le=1, description="Normalized counterclockwise (-1) to clockwise (1) roll. Zero stops this axis; the value is sampled at the next chunk boundary and held."),
    ) -> CameraChanged:
        self._require_image()
        for name, value in locals().copy().items():
            if name not in {"self"}: setattr(self.state, f"_{name}", value)
        await self.send(self._state())
        return self._camera_message()

    @event(name="release_camera", description=(
        "Stop all held camera translation and rotation. Requires a selected image; neutral values "
        "are sampled at the next chunk boundary and held. Emits `camera_changed` and broadcasts "
        "`state_update` on success, or `command_error` when no image is selected."
    ))
    async def release_camera(self) -> CameraChanged:
        self._require_image(); self._clear_motion(); await self.send(self._state()); return self._camera_message()

    @event(name="reset", description=(
        "Restart from the selected image and prompt with continuous generation from chunk one. "
        "Valid when an image exists; progress and camera axes reset. Emits `reset_queued` and "
        "broadcasts `state_update` on success, or `command_error` when no image is selected."
    ))
    async def reset(self, seed: int = InputField(default=-1, ge=-1, le=2147483647, description=(
        "Seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain the active seed; a "
        "non-negative value becomes active when reset begins."
    ))) -> ResetQueued:
        self._require_image()
        if seed >= 0: self.seed = seed
        old = self.chunk; self._request_reset(); await self.send(self._state())
        return ResetQueued(seed=self.seed, replaced_chunks=old)

    async def inference(self) -> AsyncGenerator[Lyra2Output | None, None]:
        backend, planner = self.backend, self.planner
        if backend is None or planner is None: raise RuntimeError("Lyra-2 not loaded")
        while True:
            if self.state._reset_requested:
                self.generating = True; await self.send(self._state())
                try:
                    c2w, _ = backend.reset(self._image_array(), prompt=self.state.prompt, seed=self.seed)
                    planner.reset(c2w)
                    self.chunk = 0; self.active_prompt = None; self.state._reset_requested = False
                finally: self.generating = False
            if self.image is None:
                yield None; continue
            controls = {name: getattr(self.state, f"_{name}") for name in
                        ("forward", "strafe", "vertical", "pitch", "yaw", "roll")}
            prompt = self.state.prompt
            camera = planner.plan_chunk(**controls, frame_count=80, intrinsics=backend.intrinsics)
            self.generating = True; await self.send(self._state()); started = time.perf_counter()
            try: frames, corrected = backend.generate_chunk(camera.w2c, camera.intrinsics, prompt=prompt, chunk=self.chunk + 1)
            finally: self.generating = False
            if corrected is not None: planner.reset(corrected)
            self.chunk += 1; self.active_prompt = prompt
            await self.send(ChunkCompleted(chunk=self.chunk, video_frames=len(frames),
                generation_seconds=round(time.perf_counter() - started, 3), prompt=prompt))
            await self.send(self._state())
            yield Lyra2Output(main_video=frames)

    def _decode(self, upload: UploadedFile) -> np.ndarray:
        if not upload.mime_type.startswith("image/") or not upload.data or upload.size > 25 * 1024 * 1024:
            raise CommandError("invalid_image", "Upload a non-empty image no larger than 25 MiB.")
        try:
            with Image.open(io.BytesIO(upload.data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}: raise ValueError
                return np.asarray(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise CommandError("invalid_image", "The uploaded image could not be decoded.") from error

    def _image_array(self) -> np.ndarray:
        if isinstance(self.image, Path):
            with Image.open(self.image) as image: return np.asarray(image.convert("RGB"))
        if isinstance(self.image, UploadedFile): return self._decode(self.image)
        raise RuntimeError("No image selected")

    def _request_reset(self) -> None:
        self.output.flush(); self.state._reset_requested = True; self._clear_motion()

    def _clear_motion(self) -> None:
        for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll"): setattr(self.state, f"_{name}", 0.0)

    def _camera_message(self) -> CameraChanged:
        values = {name: getattr(self.state, f"_{name}") for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll")}
        return CameraChanged(**values, applies_to_chunk=None if self.image is None else self.chunk + 1 + int(self.generating))

    def _state(self) -> StateUpdate:
        values = {name: getattr(self.state, f"_{name}") for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll")}
        return StateUpdate(image_name=self.image_name, prompt=self.state.prompt, active_prompt=self.active_prompt,
            seed=self.seed, generating=self.generating, completed_chunks=self.chunk, **values)

    def _require_image(self) -> None:
        if self.image is None: raise CommandError("image_required", "Select an image before this command.")

    def _cfg(self) -> dict:
        if self.config is None: raise RuntimeError("Lyra-2 not loaded")
        return self.config
