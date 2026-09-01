"""Serve one native YUME-1.5 rolling-latent video chunk per Reactor turn."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorPipeline,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)

from yume_assets import (
    YumeConfig,
    activate_source,
    configure_environment,
    prepare_assets,
    read_config,
)
from yume_images import (
    materialized_image,
    materialized_video,
    validate_image,
    validate_video,
)
from yume_types import (
    ActionChanged,
    ChunkCompleted,
    Movement,
    PromptChanged,
    RolloutResetQueued,
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


class Backend(Protocol):
    def reset(
        self,
        *,
        image: Path | None,
        video: Path | None = None,
        prompt: str,
        seed: int,
        movement: Movement,
        view: View,
    ) -> None: ...
    def generate_chunk(
        self, *, prompt: str, movement: Movement, view: View
    ) -> tuple[np.ndarray, str]: ...
    def end_session(self) -> None: ...


class Yume15(ReactorPipeline):
    """Explore a continuous YUME-1.5 world from text or an uploaded first frame."""

    state: YumeState
    buffer_size = 29

    def __init__(self) -> None:
        super().__init__()
        self._config: YumeConfig | None = None
        self._backend: Backend | None = None
        self._mode: (
            Literal["image_to_video", "video_to_video", "text_to_video"] | None
        ) = None
        self._image: UploadedFile | None = None
        self._video: UploadedFile | None = None
        self._image_name: str | None = None
        self._seed = 42
        self._chunk_index = 0
        self._generating = False

    def load(self, config_path: Path | None) -> None:
        """Prepare pinned public assets and load the 5B model on one GPU."""
        config = read_config(config_path)
        configure_environment(config)
        prepare_assets(config)
        activate_source(config)
        from yume_backend import YumeBackend

        self._config = config
        self._seed = config.seed
        self._backend = YumeBackend(config)

    @session_started
    def on_session_started(self) -> None:
        self.state.prompt = ""
        self.state._pressed_keys = frozenset()
        self.state._reset_requested = False
        self._mode = None
        self._image = None
        self._video = None
        self._image_name = None
        self._chunk_index = 0
        self._generating = False

    @session_ended
    def on_session_ended(self) -> None:
        if self._backend is not None:
            self._backend.end_session()
        self._mode = None
        self._image = None
        self._video = None
        self._image_name = None
        self._chunk_index = 0

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
        validate_image(image)
        config = self._require_loaded()[1]
        normalized = prompt.strip() or config.default_upload_prompt
        if seed >= 0:
            self._seed = seed
        self._mode, self._image, self._video, self._image_name = (
            "image_to_video",
            image,
            None,
            image.name,
        )
        self.state.prompt = normalized
        self._request_reset()
        await self.send(self._state_update())
        return SceneQueued(
            mode=self._mode,
            conditioning_name=image.name,
            prompt=normalized,
            seed=self._seed,
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
        validate_video(video)
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required",
                "YUME video continuation requires a non-empty prompt.",
            )
        if seed >= 0:
            self._seed = seed
        self._mode, self._image, self._video, self._image_name = (
            "video_to_video",
            None,
            video,
            video.name,
        )
        self.state.prompt = normalized
        self._request_reset()
        await self.send(self._state_update())
        return SceneQueued(
            mode=self._mode,
            conditioning_name=video.name,
            prompt=normalized,
            seed=self._seed,
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
            self._seed = seed
        self._mode, self._image, self._video, self._image_name = (
            "text_to_video",
            None,
            None,
            None,
        )
        self.state.prompt = normalized
        self._request_reset()
        await self.send(self._state_update())
        return SceneQueued(
            mode=self._mode,
            conditioning_name=None,
            prompt=normalized,
            seed=self._seed,
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
            self._seed = seed
        replaced = self._chunk_index
        self._request_reset()
        await self.send(self._state_update())
        return RolloutResetQueued(seed=self._seed, replaced_chunks=replaced)

    async def inference(self) -> AsyncGenerator[YumeOutput | None, None]:
        """Generate exactly one native eight-latent continuation per turn."""
        backend, config = self._require_loaded()
        while True:
            if self._mode is None:
                yield None
                continue
            if self.state._reset_requested:
                self.state._reset_requested = False
                if self._mode == "image_to_video":
                    assert self._image is not None
                    with materialized_image(
                        self._image, config.runtime_dir
                    ) as image_path:
                        backend.reset(
                            image=image_path,
                            video=None,
                            prompt=self.state.prompt,
                            seed=self._seed,
                            movement="none",
                            view="none",
                        )
                elif self._mode == "video_to_video":
                    assert self._video is not None
                    with materialized_video(
                        self._video, config.runtime_dir
                    ) as video_path:
                        backend.reset(
                            image=None,
                            video=video_path,
                            prompt=self.state.prompt,
                            seed=self._seed,
                            movement="none",
                            view="none",
                        )
                else:
                    backend.reset(
                        image=None,
                        video=None,
                        prompt=self.state.prompt,
                        seed=self._seed,
                        movement="none",
                        view="none",
                    )
                self._chunk_index = 0
            movement, view = self._resolve_controls(self.state._pressed_keys)
            prompt = self.state.prompt
            self._generating = True
            await self.send(self._state_update())
            started = time.perf_counter()
            try:
                frames, exact_prompt = backend.generate_chunk(
                    prompt=prompt, movement=movement, view=view
                )
            finally:
                self._generating = False
            self._chunk_index += 1
            await self.send(
                ChunkCompleted(
                    chunk=self._chunk_index,
                    frames=int(frames.shape[0]),
                    generation_seconds=round(time.perf_counter() - started, 3),
                    prompt=prompt,
                    conditioned_prompt=exact_prompt,
                    movement=movement,
                    view=view,
                )
            )
            await self.send(self._state_update())
            yield YumeOutput(main_video=np.ascontiguousarray(frames))

    def _request_reset(self) -> None:
        self.output.flush()
        self._clear_controls()
        self.state._reset_requested = True
        self._chunk_index = 0

    def _require_scene(self) -> None:
        if self._mode is None:
            raise CommandError("scene_required", "Select an image or text scene first.")

    def _require_loaded(self) -> tuple[Backend, YumeConfig]:
        if self._backend is None or self._config is None:
            raise RuntimeError("YUME was not loaded")
        return self._backend, self._config

    def _next_chunk(self) -> int:
        return (
            1
            if self.state._reset_requested
            else self._chunk_index + 1 + int(self._generating)
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

    def _state_update(self) -> StateUpdate:
        return StateUpdate(
            mode=self._mode or "uninitialized",
            conditioning_name=self._image_name,
            prompt=self.state.prompt,
            pressed_keys=self._ordered_keys(),
            seed=self._seed,
            reset_queued=self.state._reset_requested,
            generating=self._generating,
            completed_chunks=self._chunk_index,
            next_chunk=None if self._mode is None else self._next_chunk(),
        )
