"""Expose Lyra 2.0's native autoregressive video step through Reactor SDK."""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path

from reactor_runtime.distributed import DistributedRunner
from lyra2_assets import prepare_config
from lyra2_images import prepare_image
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    disconnected,
    event,
    get_weights_path,
    session_ended,
    session_started,
)

from lyra2_model import Lyra2Input, Lyra2Model, Lyra2Result
from lyra2_camera import Lyra2CameraPlanner
from lyra2_types import (
    CameraChanged,
    ChunkCompleted,
    ImageSelected,
    Lyra2Output,
    Lyra2State,
    PromptQueued,
    ResetQueued,
    StateUpdate,
)


class Lyra2(ReactorApp):
    """Explore an image-conditioned world through native 80-frame AR updates."""

    state: Lyra2State
    buffer_size = 80

    def __init__(self) -> None:
        super().__init__()
        self.config: dict | None = None
        self.engine: DistributedRunner | None = None

    def load(self, config_path: Path | None) -> None:
        if config_path is None:
            raise ValueError("Lyra-2 requires lyra2.yaml")
        self.config = prepare_config(config_path, get_weights_path())
        # Isolate upstream's cwd and module changes in a Runtime-owned worker.
        self.engine = DistributedRunner(
            Lyra2Model,
            world_size=1,
            load_kwargs={"config": self.config},
            start_timeout=3600,
            call_timeout=600,
        )
        self.engine.start()

    @session_started
    def started(self) -> None:
        config = self._cfg()
        self.state._seed = config["seed"]
        self.state._planner = Lyra2CameraPlanner(
            translation_per_frame=config["translation_per_frame"],
            rotation_degrees_per_frame=config["rotation_degrees_per_frame"],
        )

    @session_ended
    def ended(self) -> None:
        self.engine.reset()
        self.output.flush()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._state())

    @disconnected
    async def on_disconnected(self) -> None:
        self._clear_motion()
        await self.send(self._state())

    @event(
        name="set_image",
        description=(
            "Select an uploaded image and begin a fresh continuous rollout from it. Valid at any "
            "time; the selected image, prompt, and seed replace the current world when the next "
            "chunk begins. Emits `image_selected` and broadcasts `state_update` on success, or "
            "`command_error` when the upload is missing, unsupported, empty, or larger than 25 MiB."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description=(
                "Anchor image uploaded through Reactor. JPEG, PNG, WebP, or BMP up to 25 MiB; it is "
                "resized to 768x448 and becomes active when the fresh rollout begins."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Scene description for the fresh rollout, up to 4096 characters. Whitespace is trimmed; "
                "an empty value selects Lyra's default prompt."
            ),
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2147483647,
            description=(
                "Seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain the active seed; a "
                "non-negative value becomes active when the rollout begins."
            ),
        ),
    ) -> ImageSelected:
        self.state._anchor = await asyncio.to_thread(prepare_image, image)
        if seed >= 0:
            self.state._seed = seed
        self.state._image, self.state._image_name = None, image.name
        self.state.prompt = prompt.strip() or self._cfg()["default_prompt"]
        self._request_reset()
        await self.send(self._state())
        return ImageSelected(
            filename=image.name, prompt=self.state.prompt, seed=self.state._seed
        )

    @event(
        name="random_image",
        description=(
            "Select a random built-in image and begin a fresh continuous rollout using its paired "
            "prompt. Valid at any time. Emits `image_selected` and broadcasts `state_update` on "
            "success, or `command_error` when no built-in images are available."
        ),
    )
    async def random_image(self) -> ImageSelected:
        root = Path(self._cfg()["source_path"]) / "assets/samples"
        images = (
            sorted(
                p
                for p in root.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if root.is_dir()
            else []
        )
        if not images:
            raise CommandError(
                "image_unavailable", "No bundled Lyra-2 sample images were found."
            )
        choices = [p for p in images if p != self.state._image] or images
        selected = secrets.choice(choices)
        upload = UploadedFile(
            name=selected.name, mime_type="image/jpeg", data=selected.read_bytes()
        )
        anchor = await asyncio.to_thread(prepare_image, upload)
        self.state._image, self.state._image_name = selected, selected.name
        self.state._anchor = anchor
        caption = selected.with_suffix(".txt")
        self.state.prompt = (
            caption.read_text().strip()
            if caption.exists()
            else self._cfg()["default_prompt"]
        )
        self._request_reset()
        await self.send(self._state())
        return ImageSelected(
            filename=self.state._image_name,
            prompt=self.state.prompt,
            seed=self.state._seed,
        )

    @event(
        name="set_prompt",
        description=(
            "Set the text condition without restarting the current continuous world. Requires a "
            "selected image; the normalized text is sampled when the next chunk begins. Emits "
            "`prompt_queued` and broadcasts `state_update` on success, or `command_error` for empty "
            "text or a missing image."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene description, up to 4096 characters. Whitespace is trimmed and the "
                "result is sampled at the next chunk boundary and held for later chunks."
            ),
        ),
    ) -> PromptQueued:
        self._require_image()
        value = prompt.strip()
        if not value:
            raise CommandError(
                "prompt_required", "Lyra-2 requires a non-empty caption."
            )
        self.state.prompt = value
        applies = self.state._chunk + 1
        await self.send(self._state())
        return PromptQueued(prompt=value, applies_to_chunk=applies)

    @event(
        name="set_camera_motion",
        description=(
            "Set all six held camera axes for the next chunk in one command. Requires a selected "
            "image; values are sampled at the next chunk boundary and held for later chunks. Emits "
            "`camera_changed` and broadcasts `state_update` on success, or `command_error` when no "
            "image is selected."
        ),
    )
    async def set_camera_motion(
        self,
        forward: float = InputField(
            default=0,
            ge=-1,
            le=1,
            description="Normalized backward (-1) to forward (1) translation. Zero stops this axis; the value is sampled at the next chunk boundary and held.",
        ),
        strafe: float = InputField(
            default=0,
            ge=-1,
            le=1,
            description="Normalized left (-1) to right (1) translation. Zero stops this axis; the value is sampled at the next chunk boundary and held.",
        ),
        vertical: float = InputField(
            default=0,
            ge=-1,
            le=1,
            description="Normalized down (-1) to up (1) translation. Zero stops this axis; the value is sampled at the next chunk boundary and held.",
        ),
        pitch: float = InputField(
            default=0,
            ge=-1,
            le=1,
            description="Normalized downward (-1) to upward (1) pitch. Zero stops this axis; the value is sampled at the next chunk boundary and held.",
        ),
        yaw: float = InputField(
            default=0,
            ge=-1,
            le=1,
            description="Normalized left (-1) to right (1) yaw. Zero stops this axis; the value is sampled at the next chunk boundary and held.",
        ),
        roll: float = InputField(
            default=0,
            ge=-1,
            le=1,
            description="Normalized counterclockwise (-1) to clockwise (1) roll. Zero stops this axis; the value is sampled at the next chunk boundary and held.",
        ),
    ) -> CameraChanged:
        self._require_image()
        self.state.forward = forward
        self.state.strafe = strafe
        self.state.vertical = vertical
        self.state.pitch = pitch
        self.state.yaw = yaw
        self.state.roll = roll
        await self.send(self._state())
        return self._camera_message()

    @event(
        name="release_camera",
        description=(
            "Stop all held camera translation and rotation. Requires a selected image; neutral values "
            "are sampled at the next chunk boundary and held. Emits `camera_changed` and broadcasts "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def release_camera(self) -> CameraChanged:
        self._require_image()
        self._clear_motion()
        await self.send(self._state())
        return self._camera_message()

    @event(
        name="reset",
        description=(
            "Restart from the selected image and prompt with continuous generation from chunk one. "
            "Valid when an image exists; progress and camera axes reset. Emits `reset_queued` and "
            "broadcasts `state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2147483647,
            description=(
                "Seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain the active seed; a "
                "non-negative value becomes active when reset begins."
            ),
        ),
    ) -> ResetQueued:
        self._require_image()
        if seed >= 0:
            self.state._seed = seed
        old = self.state._chunk
        self._request_reset()
        await self.send(self._state())
        return ResetQueued(seed=self.state._seed, replaced_chunks=old)

    async def process_input(self) -> Lyra2Input:
        if self.state._anchor is None:
            raise ApplicationError("No image selected")
        fresh = self.state._world_id != self.state._applied_world_id
        camera = None
        if not fresh:
            controls = {
                name: getattr(self.state, name)
                for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll")
            }
            camera = self.state._planner.plan_chunk(
                **controls, frame_count=80, intrinsics=self.state._intrinsics
            )
        return Lyra2Input(
            world_id=self.state._world_id,
            anchor=self.state._anchor if fresh else None,
            prompt=self.state.prompt,
            seed=self.state._seed,
            w2c=None if camera is None else camera.w2c,
            intrinsics=None if camera is None else camera.intrinsics,
        )

    def generate(self, input: Lyra2Input) -> Lyra2Result:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> Lyra2Output | None:
        if outcome.error is not None:
            # Unexpected model failures end the session; no success hides a discontinuity.
            raise outcome.error
        result: Lyra2Result = outcome.result
        self.state._applied_world_id = result.world_id
        if result.corrected_c2w is not None:
            self.state._planner.reset(result.corrected_c2w)
        if result.intrinsics is not None:
            self.state._intrinsics = result.intrinsics
        self.state._chunk = result.chunk
        if result.frames is None:
            self.state._active_prompt = None
            await self.send(self._state())
            return None
        self.state._active_prompt = self.state.prompt
        await self.send(
            ChunkCompleted(
                chunk=result.chunk,
                video_frames=len(result.frames),
                generation_seconds=round(outcome.elapsed, 3),
                prompt=self.state.prompt,
            )
        )
        await self.send(self._state())
        return Lyra2Output(main_video=result.frames)

    def _request_reset(self) -> None:
        self.output.flush()
        self.state._chunk = 0
        self.state._active_prompt = None
        self.state._intrinsics = None
        self.state._world_id += 1
        self.state._applied_world_id = None
        self._clear_motion()

    def _clear_motion(self) -> None:
        for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll"):
            setattr(self.state, name, 0.0)

    def _camera_message(self) -> CameraChanged:
        values = {
            name: getattr(self.state, name)
            for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll")
        }
        return CameraChanged(
            **values,
            applies_to_chunk=None
            if self.state._anchor is None
            else self.state._chunk + 1,
        )

    def _state(self) -> StateUpdate:
        values = {
            name: getattr(self.state, name)
            for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll")
        }
        return StateUpdate(
            image_name=self.state._image_name,
            prompt=self.state.prompt,
            active_prompt=self.state._active_prompt,
            seed=self.state._seed,
            completed_chunks=self.state._chunk,
            **values,
        )

    def _require_image(self) -> None:
        if self.state._anchor is None:
            raise CommandError("image_required", "Select an image before this command.")

    def _cfg(self) -> dict:
        if self.config is None:
            raise RuntimeError("Lyra-2 not loaded")
        return self.config
