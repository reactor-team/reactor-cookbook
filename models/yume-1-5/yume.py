"""Serve one native YUME-1.5 rolling-latent video chunk per Reactor turn."""

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
    disconnected,
    event,
    get_weights_path,
    session_ended,
    session_started,
)
from yume_assets import (
    YumeConfig,
    configure_environment,
    prepare_assets,
    read_config,
)
from yume_images import (
    prepare_image,
    prepare_video,
)
from yume_model import YumeAnchor, YumeInput, YumeModel, YumeResult
from yume_types import (
    ActionChanged,
    ChunkCompleted,
    Movement,
    PromptChanged,
    RolloutResetQueued,
    RolloutLimitReached,
    SceneQueued,
    StateUpdate,
    View,
    YumeOutput,
    YumeState,
)

_KEYS = ["w", "a", "s", "d", "arrow_left", "arrow_right", "arrow_up", "arrow_down"]
_MOVEMENT_KEYS = {
    frozenset(): "none",
    frozenset({"w"}): "forward",
    frozenset({"s"}): "backward",
    frozenset({"a"}): "left",
    frozenset({"d"}): "right",
    frozenset({"w", "a"}): "forward_left",
    frozenset({"w", "d"}): "forward_right",
    frozenset({"s", "a"}): "backward_left",
    frozenset({"s", "d"}): "backward_right",
}
_VIEW_KEYS = {
    frozenset(): "none",
    frozenset({"arrow_left"}): "pan_left",
    frozenset({"arrow_right"}): "pan_right",
    frozenset({"arrow_up"}): "tilt_up",
    frozenset({"arrow_down"}): "tilt_down",
    frozenset({"arrow_up", "arrow_left"}): "tilt_up_left",
    frozenset({"arrow_up", "arrow_right"}): "tilt_up_right",
    frozenset({"arrow_down", "arrow_left"}): "tilt_down_left",
    frozenset({"arrow_down", "arrow_right"}): "tilt_down_right",
}


class Yume15(ReactorApp):
    """Explore a continuous YUME-1.5 world from text or an uploaded first frame."""

    state: YumeState
    buffer_size = 29

    def __init__(self) -> None:
        super().__init__()
        self._config: YumeConfig | None = None
        self._engine: YumeModel

    def load(self, config_path: Path | None) -> None:
        """Prepare pinned public assets and load the 5B model on one GPU."""
        config = read_config(config_path, get_weights_path())
        configure_environment(config)
        self._config = config
        prepare_assets(config)
        self._engine = YumeModel()
        self._engine.load(config)

    @session_started
    def on_session_started(self) -> None:
        self.state._seed = self._require_config().seed
        self.state.prompt = ""
        self.state._pressed_keys = frozenset()
        self.state._world_id += 1
        self.state._applied_world_id = None
        self.state._mode = None
        self.state._image = None
        self.state._video = None
        self.state._image_name = None
        self.state._chunk_index = 0
        self.state._complete = False

    @session_ended
    def on_session_ended(self) -> None:
        self._engine.reset()
        self.output.flush()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        self._clear_controls()
        await self.send(self._state_update())

    @event(
        name="set_image",
        description="Start a new world from an uploaded image. The replacement starts at the next chunk boundary. A blank `prompt` uses a neutral continuation description. Emits `scene_queued` and `state_update` on success, or `command_error` if the image is invalid.",
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description="Reference frame uploaded through the Reactor file-upload protocol. Accepts JPEG, PNG, WebP, BMP, or TIFF up to 25 MiB and 100 million pixels; YUME fits it to the output frame.",
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Optional description of the scene and events to preserve or introduce. It conditions the first and subsequent chunks; blank selects the server's neutral image-continuation description.",
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Seed used to initialize this rollout. Use `-1` to retain the session's current seed.",
        ),
    ) -> SceneQueued:
        pixels = await asyncio.to_thread(prepare_image, image)
        config = self._require_config()
        normalized = prompt.strip() or config.default_upload_prompt
        if seed >= 0:
            self.state._seed = seed
        (
            self.state._mode,
            self.state._image,
            self.state._video,
            self.state._image_name,
        ) = (
            "image_to_video",
            pixels,
            None,
            image.name,
        )
        self.state.prompt = normalized
        self._request_reset()
        await self.send(self._state_update())
        return SceneQueued(
            mode=self.state._mode,
            conditioning_name=image.name,
            prompt=normalized,
            seed=self.state._seed,
        )

    @event(
        name="set_video_scene",
        description="Start a new world from an uploaded video and a prompt. The replacement starts at the next chunk boundary. Emits `scene_queued` and `state_update` on success, or `command_error` if the video or prompt is invalid.",
    )
    async def set_video_scene(
        self,
        video: UploadedFile = InputField(  # noqa: B008
            moderate=True,
            description="Reference video uploaded through the Reactor file-upload protocol. It must be decodable, contain at least 33 frames, and be no larger than 500 MiB; the first 33 frames anchor the rollout.",
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Non-empty description of the scene and forthcoming events. It conditions the first generated continuation and remains active for later chunks.",
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Seed used to initialize this rollout. Use `-1` to retain the session's current seed.",
        ),
    ) -> SceneQueued:
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required",
                "YUME video continuation requires a non-empty prompt.",
            )
        frames = await asyncio.to_thread(prepare_video, video)
        if seed >= 0:
            self.state._seed = seed
        (
            self.state._mode,
            self.state._image,
            self.state._video,
            self.state._image_name,
        ) = (
            "video_to_video",
            None,
            frames,
            video.name,
        )
        self.state.prompt = normalized
        self._request_reset()
        await self.send(self._state_update())
        return SceneQueued(
            mode=self.state._mode,
            conditioning_name=video.name,
            prompt=normalized,
            seed=self.state._seed,
        )

    @event(
        name="set_text_scene",
        description="Start a new world from a text description without reference media. The replacement starts at the next chunk boundary. Emits `scene_queued` and `state_update` on success, or `command_error` if `prompt` is blank.",
    )
    async def set_text_scene(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Non-empty description of the initial scene and forthcoming events. It conditions the first and subsequent chunks until changed by `set_prompt`.",
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Seed used to initialize this rollout. Use `-1` to retain the session's current seed.",
        ),
    ) -> SceneQueued:
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "YUME text-to-video requires a non-empty prompt."
            )
        if seed >= 0:
            self.state._seed = seed
        (
            self.state._mode,
            self.state._image,
            self.state._video,
            self.state._image_name,
        ) = (
            "text_to_video",
            None,
            None,
            None,
        )
        self.state.prompt = normalized
        self._request_reset()
        await self.send(self._state_update())
        return SceneQueued(
            mode=self.state._mode,
            conditioning_name=None,
            prompt=normalized,
            seed=self.state._seed,
        )

    @event(
        name="set_prompt",
        description="Change the scene and event description without restarting the world. It is valid after a scene is selected and applies from the next chunk boundary. Emits `prompt_changed` and `state_update` on success, or `command_error` if no scene exists or `prompt` is blank.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Non-empty description used to condition forthcoming chunks. It does not alter a chunk already being generated or discard visual history.",
        ),
    ) -> PromptChanged:
        self._require_scene()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "YUME requires a non-empty prompt.")
        self.state.prompt = normalized
        result = PromptChanged(prompt=normalized, applies_to_chunk=self._next_chunk())
        await self.send(self._state_update())
        return result

    @event(
        name="set_key_state",
        description="Press or release one persistent movement or view key. It is valid after a scene is selected and affects the next chunk boundary. Compatible keys may be held together. Emits `action_changed` and `state_update` on success, or `command_error` for no scene or an unsupported combination.",
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=_KEYS,
            description="Translation key (`w`, `a`, `s`, or `d`) or camera-view key (`arrow_left`, `arrow_right`, `arrow_up`, or `arrow_down`) whose held state will change.",
        ),
        pressed: bool = InputField(
            default=True,
            description="Set to `true` to hold `key`, or `false` to release it. The resulting held-key set applies from the next chunk boundary.",
        ),
    ) -> ActionChanged:
        self._require_scene()
        updated = (
            self.state._pressed_keys.union((key,))
            if pressed
            else self.state._pressed_keys.difference((key,))
        )
        self._resolve_controls(updated)
        self.state._pressed_keys = updated
        result = ActionChanged(
            key=key,
            pressed=pressed,
            pressed_keys=self._ordered_keys(),
            applies_to_chunk=self._next_chunk(),
        )
        await self.send(self._state_update())
        return result

    @event(
        name="release_controls",
        description="Release every held movement and view key. It is valid after a scene is selected and restores stationary controls from the next chunk boundary. Emits `action_changed` and `state_update` on success, or `command_error` if no scene exists.",
    )
    async def release_controls(self) -> ActionChanged:
        self._require_scene()
        self._clear_controls()
        result = ActionChanged(
            key="all",
            pressed=False,
            pressed_keys=[],
            applies_to_chunk=self._next_chunk(),
        )
        await self.send(self._state_update())
        return result

    @event(
        name="reset",
        description="Restart the selected scene and discard its generated history. It is valid after a scene is selected and begins at the next chunk boundary with all controls released. Emits `rollout_reset_queued` and `state_update` on success, or `command_error` if no scene exists.",
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Seed used to restart the rollout. Use `-1` to retain the session's current seed; the value is read when the reset begins.",
        ),
    ) -> RolloutResetQueued:
        self._require_scene()
        if seed >= 0:
            self.state._seed = seed
        replaced = self.state._chunk_index
        self._request_reset()
        await self.send(self._state_update())
        return RolloutResetQueued(seed=self.state._seed, replaced_chunks=replaced)

    async def process_input(self) -> YumeInput:
        if self.state._mode is None:
            raise ApplicationError("no scene selected")
        if self.state._complete:
            raise ApplicationError("rollout complete; reset or select a scene")
        movement, view = self._map_controls(self.state._pressed_keys)
        anchor = None
        if self.state._world_id != self.state._applied_world_id:
            media = (
                self.state._image
                if self.state._image is not None
                else self.state._video
            )
            anchor = YumeAnchor(
                mode=self.state._mode,
                media=media,
                seed=self.state._seed,
            )
        return YumeInput(
            world_id=self.state._world_id,
            anchor=anchor,
            prompt=self.state.prompt,
            movement=movement,
            view=view,
        )

    def generate(self, input: YumeInput) -> YumeResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> YumeOutput | None:
        if outcome.error is not None:
            # Invalid model state and GPU failures cannot be repaired by a silent reset.
            raise outcome.error
        result: YumeResult = outcome.result
        self.state._applied_world_id = result.world_id
        self.state._chunk_index = result.chunk_index
        self.state._complete = result.complete
        movement, view = self._map_controls(self.state._pressed_keys)
        await self.send(
            ChunkCompleted(
                chunk=result.chunk_index,
                frames=int(result.frames.shape[0]),
                generation_seconds=round(outcome.elapsed, 3),
                prompt=self.state.prompt,
                movement=movement,
                view=view,
            )
        )
        if result.complete:
            await self.send(RolloutLimitReached(completed_chunks=result.chunk_index))
        await self.send(self._state_update())
        return YumeOutput(main_video=result.frames)

    def _request_reset(self) -> None:
        self.output.flush()
        self._clear_controls()
        self.state._world_id += 1
        self.state._chunk_index = 0
        self.state._complete = False

    def _require_scene(self) -> None:
        if self.state._mode is None:
            raise CommandError("scene_required", "Select an image or text scene first.")

    def _require_config(self) -> YumeConfig:
        if self._config is None:
            raise RuntimeError("YUME was not loaded")
        return self._config

    def _next_chunk(self) -> int:
        return (
            1
            if self.state._world_id != self.state._applied_world_id
            else self.state._chunk_index + 1
        )

    def _clear_controls(self) -> None:
        self.state._pressed_keys = frozenset()

    def _ordered_keys(self) -> list[str]:
        return [key for key in _KEYS if key in self.state._pressed_keys]

    def _resolve_controls(self, keys: frozenset[str]) -> tuple[Movement, View]:
        movement_keys = frozenset(key for key in keys if key in {"w", "a", "s", "d"})
        view_keys = keys.difference(movement_keys)
        if movement_keys not in _MOVEMENT_KEYS or view_keys not in _VIEW_KEYS:
            raise CommandError(
                "unsupported_key_combination",
                "YUME cannot hold opposite movement keys or opposite view keys together.",
            )
        return _MOVEMENT_KEYS[movement_keys], _VIEW_KEYS[view_keys]

    def _map_controls(self, keys: frozenset[str]) -> tuple[Movement, View]:
        """Map state already validated atomically by the command handler."""
        movement = keys.intersection({"w", "a", "s", "d"})
        return _MOVEMENT_KEYS[movement], _VIEW_KEYS[keys.difference(movement)]

    def _state_update(self) -> StateUpdate:
        return StateUpdate(
            mode=self.state._mode or "uninitialized",
            conditioning_name=self.state._image_name,
            prompt=self.state.prompt,
            pressed_keys=self._ordered_keys(),
            seed=self.state._seed,
            reset_queued=self.state._mode is not None
            and self.state._world_id != self.state._applied_world_id,
            limit_reached=self.state._complete,
            completed_chunks=self.state._chunk_index,
            next_chunk=None
            if self.state._mode is None or self.state._complete
            else self._next_chunk(),
        )
