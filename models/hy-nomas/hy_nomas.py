"""Hy-Nomas streaming world generation with live text and camera controls."""

from __future__ import annotations

import asyncio
from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from hy_nomas_assets import resolve_assets
from hy_nomas_camera import CAMERAS, KEYS, decode_trajectory, plan_camera, plan_keyboard
from hy_nomas_model import (
    MAX_WORLD_FRAMES,
    HyNomasInput,
    HyNomasModel,
    HyNomasResult,
    WorldStart,
)
from hy_nomas_types import HyNomasOutput, HyNomasState, StateUpdate


class HyNomas(ReactorApp):
    """Create a world from text, then change its description and camera between chunks."""

    state: HyNomasState

    def load(self, config_path: Path | None) -> None:
        settings, serving = resolve_assets(config_path, Path(get_weights_path()))
        app_type = type(self)
        if serving.playback_fps is not None:
            app_type.fps = serving.playback_fps
        elif "fps" in vars(app_type):
            delattr(app_type, "fps")
        warmup = None
        if serving.warmup_frames is not None:
            poses, intrinsics = plan_keyboard((), serving.warmup_frames)
            warmup = HyNomasInput(
                world_id=0,
                start=WorldStart(seed=42, segment_frames=serving.warmup_frames),
                prompt="A quiet stone courtyard with a fountain and green ivy.",
                c2ws=poses,
                intrinsics=intrinsics,
                translation_strength=1.0,
                rotation_strength=2.5,
            )
        self.engine = DistributedRunner(
            HyNomasModel,
            world_size=serving.num_gpus,
            load_kwargs={"settings": settings, "warmup": warmup},
            call_timeout=1800,
            start_timeout=1800,
        )
        self.engine.start()

    async def process_input(self) -> HyNomasInput:
        state = self.state
        if state._start is None or state.paused or state._complete:
            raise ApplicationError("Press Start or resume the active world.")
        if not state.prompt.strip():
            raise ApplicationError("Enter a prompt before generating.")
        count = min(state._start.segment_frames, MAX_WORLD_FRAMES - state._generated)
        if state.camera == "trajectory":
            if state._camera_poses is None or state._camera_cursor + count > len(
                state._camera_poses
            ):
                raise ApplicationError(
                    "Upload a longer camera path or choose another camera motion."
                )
            start = state._camera_cursor
            poses = state._camera_poses[start : start + count].copy()
            intrinsics = state._camera_intrinsics[start : start + count].copy()
        elif state.camera == "keyboard":
            keys = (
                state._pending_keys
                if state._pending_keys is not None
                else state._held_keys
            )
            poses, intrinsics = plan_keyboard(keys, count)
            state._pending_keys = None
        else:
            poses, intrinsics = plan_camera(state.camera, count)
        return HyNomasInput(
            world_id=state._world_id,
            start=state._start if state._applied_world_id != state._world_id else None,
            prompt=state.prompt,
            c2ws=poses,
            intrinsics=intrinsics,
            translation_strength=state.translation_strength,
            rotation_strength=state.rotation_strength,
        )

    def generate(self, input: HyNomasInput) -> HyNomasResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> HyNomasOutput | None:
        if outcome.error is not None:
            # Client readiness is checked before inference. Native/worker errors
            # stop the session instead of silently replacing the current world.
            raise outcome.error
        result: HyNomasResult = outcome.result
        state = self.state
        if result.world_id != state._world_id:
            return None
        if state.camera == "trajectory":
            state._camera_cursor += result.generated_frames - state._generated
        state._applied_world_id = result.world_id
        state._completed = result.chunk_index
        state._generated = result.generated_frames
        state._emitted = result.emitted_frames
        state._last_seconds = outcome.elapsed
        state._complete = result.complete
        if (
            result.complete
            or state.camera == "trajectory"
            and len(state._camera_poses) - state._camera_cursor
            < min(state._start.segment_frames, MAX_WORLD_FRAMES - state._generated)
        ):
            state.paused = True
        await self.send(self._status())
        return HyNomasOutput(main_video=result.frames)

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        self.engine.reset()
        self.output.flush()

    @event(
        name="start",
        description="Start a fresh world from text, using the staged camera settings and replacing previous playback. Fails with `invalid_prompt` for empty text, `invalid_chunk_frames` unless frames are a multiple of 24, or `camera_path_too_short` for an insufficient selected trajectory.",
    )
    async def start(
        self,
        prompt: str = InputField(
            default="",
            max_length=8192,
            moderate=True,
            description="Initial scene description. Empty uses a previously accepted prompt.",
        ),
        seed: int = InputField(
            default=42,
            ge=0,
            le=2147483647,
            description="Random seed for the new world.",
        ),
        chunk_frames: int = InputField(
            default=120,
            ge=48,
            le=240,
            description="Frames per chunk at the 24 FPS content timebase. Multiple of 24; fixed for this world.",
        ),
    ) -> StateUpdate:
        text = prompt.strip() or self.state.prompt.strip()
        if not text:
            raise CommandError("invalid_prompt", "Enter a scene description.")
        if chunk_frames % 24:
            raise CommandError(
                "invalid_chunk_frames", "Choose a multiple of 24 frames."
            )
        state = self.state
        if state.camera == "trajectory":
            self._require_camera_frames(chunk_frames, rewind=True)
        state.prompt = text
        state._start = WorldStart(seed=seed, segment_frames=chunk_frames)
        self._fresh_world()
        return await self._broadcast()

    @event(
        name="reset",
        description="Restart the active world with its current prompt and chunk length, optionally choosing a new seed. Rewinds the camera path. Fails with `no_world` before Start.",
    )
    async def reset(
        self,
        seed: int | None = InputField(
            default=None,
            ge=0,
            le=2147483647,
            description="Seed for the restarted world. Omit to keep the current seed.",
        ),
    ) -> StateUpdate:
        self._require_world()
        if seed is not None:
            self.state._start = WorldStart(
                seed=seed, segment_frames=self.state._start.segment_frames
            )
        self._fresh_world()
        return await self._broadcast()

    @event(
        name="set_prompt",
        description="Update the next chunk's description while continuing the world. Frames already generated keep the previous prompt. Does not resume a paused world; can stage text before Start. Fails with `invalid_prompt` for empty text, or `rollout_complete` after the 120-second duration.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            max_length=8192,
            moderate=True,
            description="Scene/action text, or a JSON object with one segments item, for the next chunk.",
        ),
    ) -> StateUpdate:
        self._require_unfinished()
        if not prompt.strip():
            raise CommandError("invalid_prompt", "Enter a non-empty scene description.")
        self.state.prompt = prompt.strip()
        return await self._broadcast()

    @event(
        name="set_paused",
        description="Pause or resume generation between chunks. Frames already generated can keep playing. Fails with `no_world` before Start, `rollout_complete` on completed worlds, or `camera_path_too_short` when resuming an exhausted selected trajectory.",
    )
    async def set_paused(
        self, paused: bool = InputField(description="True pauses; false resumes.")
    ) -> StateUpdate:
        self._require_world()
        self._require_unfinished()
        if not paused and self.state.camera == "trajectory":
            self._require_camera_frames()
        self.state.paused = paused
        return await self._broadcast()

    @event(
        name="set_camera",
        description="Select camera motion for the next chunk while continuing the world and keeping its pause state. Can stage settings for the next Start after a world completes. Fails with `camera_path_too_short` for a missing/exhausted trajectory.",
    )
    async def set_camera(
        self,
        camera: str = InputField(
            choices=CAMERAS, description="Camera motion to hold until changed."
        ),
    ) -> StateUpdate:
        if camera == "trajectory":
            self._require_camera_frames()
        if camera != self.state.camera:
            self.state._held_keys = ()
            self.state._pending_keys = None
        self.state.camera = camera
        return await self._broadcast()

    @event(
        name="set_keyboard",
        description="Hold WASD and arrow-key camera motion while continuing the world and keeping its prompt and pause state. A brief tap can guide one chunk; use `release_controls` to cancel movement. Fails with `no_world` before Start, `world_changed` if a supplied world ID is stale, or `rollout_complete` after completion.",
    )
    async def set_keyboard(
        self,
        world_id: int | None = InputField(
            default=None,
            ge=1,
            description="Optional world identity to reject delayed controls after another viewer restarts.",
        ),
        w: bool = InputField(default=False, description="Move forward."),
        s: bool = InputField(default=False, description="Move backward."),
        a: bool = InputField(default=False, description="Strafe left."),
        d: bool = InputField(default=False, description="Strafe right."),
        up: bool = InputField(default=False, description="Look up."),
        down: bool = InputField(default=False, description="Look down."),
        left: bool = InputField(default=False, description="Look left."),
        right: bool = InputField(default=False, description="Look right."),
        latch: bool = InputField(
            default=True,
            description="Keep a brief press until the next generated chunk. Set false for key-release updates.",
        ),
    ) -> StateUpdate:
        self._require_control_world(world_id)
        keys = tuple(
            key for key, held in zip(KEYS, (w, s, a, d, up, down, left, right)) if held
        )
        self.state.camera = "keyboard"
        self.state._held_keys = keys
        if keys and latch:
            self.state._pending_keys = keys
        return await self._broadcast()

    @event(
        name="release_controls",
        description="Cancel held and brief-tap camera movement without changing the selected camera or pause state. Fails with `no_world` before Start, `world_changed` if a supplied world ID is stale, or `rollout_complete` after completion.",
    )
    async def release_controls(
        self,
        world_id: int | None = InputField(
            default=None,
            ge=1,
            description="Optional world identity to avoid stopping a replacement world.",
        ),
    ) -> StateUpdate:
        self._require_control_world(world_id)
        self.state._held_keys = ()
        self.state._pending_keys = None
        return await self._broadcast()

    @event(
        name="set_translation_strength",
        description="Change camera movement strength for the next chunk without resuming or restarting the world. Can stage settings for the next Start after completion.",
    )
    async def set_translation_strength(
        self,
        translation_strength: float = InputField(
            ge=0.1, le=3.0, description="Camera translation gain."
        ),
    ) -> StateUpdate:
        self.state.translation_strength = translation_strength
        return await self._broadcast()

    @event(
        name="set_rotation_strength",
        description="Change camera look strength for the next chunk without resuming or restarting the world. Can stage settings for the next Start after completion.",
    )
    async def set_rotation_strength(
        self,
        rotation_strength: float = InputField(
            ge=0.1, le=3.0, description="Camera rotation gain."
        ),
    ) -> StateUpdate:
        self.state.rotation_strength = rotation_strength
        return await self._broadcast()

    @event(
        name="set_camera_path",
        description="Upload a 24 FPS NPZ camera path and select it for future chunks without restarting or resuming the world. Fails with `invalid_camera_path` for malformed/oversized data, `camera_path_too_short` for less than one chunk. Can stage a path for the next Start after completion.",
    )
    async def set_camera_path(
        self,
        trajectory: UploadedFile = InputField(  # noqa: B008 - Runtime field declaration
            moderate=True,
            description="NPZ up to 4 MiB containing c2ws [T,4,4] and intrinsics [T,3,3]; 48–2880 frames at 24 FPS.",
        ),
    ) -> StateUpdate:
        poses, intrinsics = await self._decode_camera_path(trajectory)
        needed = self.state._start.segment_frames if self.state._start else 48
        if len(poses) < needed:
            raise CommandError(
                "camera_path_too_short",
                f"This chunk requires at least {needed} camera frames.",
            )
        self.state._camera_poses = poses
        self.state._camera_intrinsics = intrinsics
        self.state._camera_name = Path(trajectory.name).name
        self.state._camera_cursor = 0
        self.state.camera = "trajectory"
        return await self._broadcast()

    async def _decode_camera_path(self, trajectory: UploadedFile):
        try:
            return await asyncio.to_thread(decode_trajectory, trajectory.data)
        except ValueError as exc:
            raise CommandError("invalid_camera_path", str(exc)) from exc

    def _fresh_world(self) -> None:
        state = self.state
        state._world_id += 1
        state._completed = state._generated = state._emitted = 0
        state._complete = False
        state._last_seconds = None
        state._camera_cursor = 0
        state._held_keys = ()
        state._pending_keys = None
        state.paused = False
        self.output.flush()

    def _require_world(self) -> None:
        if self.state._start is None:
            raise CommandError("no_world", "Press Start to create a world first.")

    def _require_unfinished(self) -> None:
        if self.state._complete:
            raise CommandError(
                "rollout_complete", "Press Start or Reset to create a fresh world."
            )

    def _require_control_world(self, world_id: int | None) -> None:
        self._require_world()
        if world_id is not None and world_id != self.state._world_id:
            raise CommandError(
                "world_changed", "The world changed; control the current world."
            )
        self._require_unfinished()

    def _require_camera_frames(
        self, count: int | None = None, *, rewind: bool = False
    ) -> None:
        state = self.state
        needed = (
            count
            if count is not None
            else (
                min(state._start.segment_frames, MAX_WORLD_FRAMES - state._generated)
                if state._start
                else 48
            )
        )
        if (
            state._camera_poses is None
            or len(state._camera_poses) - (0 if rewind else state._camera_cursor)
            < needed
        ):
            raise CommandError(
                "camera_path_too_short",
                "Upload enough camera frames for the next segment.",
            )

    async def _broadcast(self) -> StateUpdate:
        status = self._status()
        await self.send(status)
        return status

    def _status(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            pressed_keys=list(state._held_keys),
            world_id=state._world_id,
            prompt=state.prompt,
            paused=state.paused,
            camera=state.camera,
            translation_strength=state.translation_strength,
            rotation_strength=state.rotation_strength,
            completed_chunks=state._completed,
            emitted_frames=state._emitted,
            last_chunk_seconds=state._last_seconds,
            active=state._start is not None,
            limit_reached=state._complete,
            seed=state._start.seed if state._start else None,
            chunk_frames=state._start.segment_frames if state._start else None,
            camera_path=state._camera_name,
            camera_frames_remaining=0
            if state._camera_poses is None
            else len(state._camera_poses) - state._camera_cursor,
        )
