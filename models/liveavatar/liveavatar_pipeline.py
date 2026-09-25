"""Audio-driven avatar takes; module name avoids upstream's liveavatar package."""

# ruff: noqa: B008 -- InputField defaults declare the Reactor wire schema.
from __future__ import annotations

import asyncio
import io
import subprocess
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

from PIL import Image
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorPipeline,
    UploadedFile,
    connected,
    event,
    session_ended,
    session_started,
)

from liveavatar_assets import WORK, configure_cache_environment
from liveavatar_types import (
    ChunkComplete,
    GenerationEnded,
    InputAccepted,
    LiveAvatarOutput,
    LiveAvatarState,
    StateUpdate,
    TakeChanged,
)


class LiveAvatar(ReactorPipeline):
    state: LiveAvatarState
    fps = 25
    buffer_size = 48

    def __init__(self):
        super().__init__()
        self._backend = None
        self._directory = None
        self._image: Path | None = None
        self._audio: Path | None = None
        self._pose: Path | None = None
        self._pending = False

    def load(self, config_path: Path | None = None):
        configure_cache_environment()
        from liveavatar_parallel import ParallelBackend

        self._backend = ParallelBackend()

    @session_started
    async def on_session_started(self):
        configure_cache_environment()
        self._directory = tempfile.TemporaryDirectory(prefix="session-", dir=WORK)
        self._image = self._audio = self._pose = None
        self._pending = False

    @session_ended
    async def on_session_ended(self):
        if self._backend is not None:
            await asyncio.to_thread(self._backend.close)
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        self._image = self._audio = self._pose = None
        self._pending = False

    @connected
    async def on_connected(self, client: ClientInfo):
        await client.send(StateUpdate.from_state(self.state))

    def _require_idle(self):
        if self.state._running:
            raise CommandError(
                "take_running", "Use `stop` before replacing take conditions."
            )

    def _path(self, name: str) -> Path:
        if self._directory is None:
            raise CommandError("session_required", "A session must be active.")
        return Path(self._directory.name) / name

    async def _send_state_update(self):
        await self.send(StateUpdate.from_state(self.state))

    @event(
        name="set_avatar_image",
        description="Select the reference image for the next avatar take while idle. Requires an active session and a completed upload. Returns `input_accepted` and broadcasts `state_update`; generation begins after a separate `start`. Rejects invalid images with `invalid_image` and changes during a take with `take_running` through `command_error`.",
    )
    async def set_avatar_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description="Uploaded reference image, such as PNG, JPEG or WebP; nonempty, at most 25 MiB and 40 million pixels. Converted to RGB and fitted to the output aspect ratio. Required alongside `set_audio` before `start`; replaces the selected image for the next take and is cleared by `reset`.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        try:
            if not image.data or image.size > 25 * 1024**2:
                raise ValueError("Image must be nonempty and at most 25 MiB")
            with Image.open(io.BytesIO(image.data)) as decoded:
                if decoded.width * decoded.height > 40_000_000:
                    raise ValueError("Image exceeds 40 million pixels")
                decoded.convert("RGB").save(self._path("reference-new.png"))
                self._path("reference-new.png").replace(self._path("reference.png"))
        except Exception as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        self._image = self._path("reference.png")
        self.state._image_name = image.name
        await self._send_state_update()
        return InputAccepted(field="avatar_image")

    @event(
        name="set_audio",
        description="Select the speech audio for the next avatar take while idle. Requires an active session and a completed upload. Returns `input_accepted` and broadcasts `state_update`; `start` also requires an accepted reference image. Rejects empty, oversized or unreadable audio with `invalid_audio`, audio shorter than 1.92 seconds with `audio_too_short`, and changes during a take with `take_running` through `command_error`.",
    )
    async def set_audio(
        self,
        audio: UploadedFile = InputField(
            moderate=True,
            description="Uploaded speech audio, such as WAV, FLAC or MP3; nonempty, at most 100 MiB and at least 1.92 seconds long after decoding. Drives lip and body motion from the next `start` and is played as mono 48 kHz on `main_audio`. Audio duration and `max_chunks` limit the take; `reset` clears this selection.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        if not audio.data or audio.size > 100 * 1024**2:
            raise CommandError(
                "invalid_audio", "Audio must be nonempty and at most 100 MiB"
            )
        raw, target = self._path("audio-upload"), self._path("driving-new.wav")
        raw.write_bytes(audio.data)
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(raw),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(target),
            ],
            capture_output=True,
        )
        if result.returncode:
            raise CommandError("invalid_audio", "Cannot decode the uploaded audio")
        import soundfile as sf

        info = sf.info(target)
        if info.duration < 48 / 25:
            raise CommandError(
                "audio_too_short",
                "Supply at least 1.92 seconds of audio for one native clip",
            )
        target.replace(self._path("driving.wav"))
        self._audio = self._path("driving.wav")
        self.state._audio_name = audio.name
        await self._send_state_update()
        return InputAccepted(field="audio")

    @event(
        name="set_pose_video",
        description="Select or clear optional pose guidance for the next take while idle. Requires an active session. Returns `input_accepted` and broadcasts `state_update`; reference image and speech audio remain required before `start`. Rejects empty, oversized or unreadable video with `invalid_pose` and changes during a take with `take_running` through `command_error`.",
    )
    async def set_pose_video(
        self,
        pose_video: UploadedFile | None = InputField(
            default=None,
            moderate=True,
            description="Uploaded MP4 containing a prepared pose sequence for motion guidance; nonempty, at most 100 MiB and containing a readable video track. Applies at the next `start`. Omit or pass null to clear the selection and use audio-driven motion. Prepare the pose sequence before upload; this command selects the supplied sequence.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        if pose_video is None:
            self._pose = None
            self.state._pose_name = None
        else:
            if not pose_video.data or pose_video.size > 100 * 1024**2:
                raise CommandError(
                    "invalid_pose", "Pose video must be nonempty and at most 100 MiB"
                )
            path = self._path("pose-new.mp4")
            path.write_bytes(pose_video.data)
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
            )
            if result.returncode or not result.stdout.strip():
                raise CommandError(
                    "invalid_pose", "Upload must contain a readable video track"
                )
            path.replace(self._path("pose.mp4"))
            self._pose = self._path("pose.mp4")
            self.state._pose_name = pose_video.name
        await self._send_state_update()
        return InputAccepted(field="pose_video")

    @event(
        name="set_prompt",
        description="Set the optional appearance, scene and performance description for the next take while idle. Returns `input_accepted` and broadcasts `state_update`; invalid field values or an active take return `command_error`. Applies at the next `start`, with speech supplied through `set_audio`. `negative_prompt` is retained as compatibility text and has no effect on video in this serving profile.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Scene, appearance and performance description, up to 4096 characters. Read at the next `start`; an empty string clears additional scene text. Speech content comes from `set_audio`. Retained by `stop` and cleared by `reset`.",
        ),
        negative_prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Compatibility text, up to 4096 characters, retained for the next `start`. This serving profile applies no negative conditioning, so changing this value has no effect on video. Empty selects the model's default text; `reset` clears the stored value.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        self.state._prompt = prompt
        self.state._negative_prompt = negative_prompt
        await self._send_state_update()
        return InputAccepted(field="prompt")

    @event(
        name="set_generation_options",
        description="Select the sampling seed and clip limit for the next take while idle. Returns `input_accepted` and broadcasts `state_update`; invalid field values or an active take return `command_error`. Both values apply at the next explicit `start`; omitted fields take their declared defaults.",
    )
    async def set_generation_options(
        self,
        seed: int = InputField(
            default=420,
            ge=0,
            le=2147483647,
            description="Non-negative sampling seed from 0 through 2147483647, default 420. Read at the next `start` and retained by `stop`; `reset` restores 420. Selecting a seed leaves the take idle until `start`.",
        ),
        max_chunks: int = InputField(
            default=10000,
            ge=1,
            le=10000,
            description="Maximum generated clips per take, from 1 through 10000, default 10000. Read at the next `start`; audio duration can finish the take earlier. The first clip has 45 frames and later clips have 48 at 25 FPS. Retained by `stop`; `reset` restores 10000.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        self.state._seed, self.state._max_chunks = seed, max_chunks
        await self._send_state_update()
        return InputAccepted(field="generation_options")

    @event(
        name="start",
        description="Begin an avatar take from the selected image, speech audio and optional conditions. Valid while idle after `set_avatar_image` and `set_audio` succeed. Returns `take_changed` and broadcasts `state_update`, resetting progress and the generation error. Rejects missing inputs with `inputs_required` and an active take with `take_running` through `command_error`. Configure every desired input before calling this parameter-free command.",
    )
    async def start(self) -> TakeChanged:
        self._require_idle()
        if self._image is None or self._audio is None:
            raise CommandError(
                "inputs_required", "Upload reference image and driving audio first"
            )
        self.state._running = True
        self.state._chunks = self.state._frames = 0
        self.state._error = None
        self._pending = True
        await self._send_state_update()
        return TakeChanged(action="start")

    @event(
        name="stop",
        description="End the take and clear queued playback while retaining selected inputs, options and progress for inspection or another `start`. Valid while idle or generating in an active session. Returns `take_changed` and broadcasts `state_update`. Handled between inference turns, so an in-flight clip can delay the reply. Explicit stopping is reported by this reply; `generation_ended` reports automatic completion or failure.",
    )
    async def stop_take(self) -> TakeChanged:
        if self._backend is not None:
            await asyncio.to_thread(self._backend.close)
        self.state._running = self._pending = False
        self.output.flush()
        await self._send_state_update()
        return TakeChanged(action="stop")

    @event(
        name="reset",
        description="End the take, clear queued playback and selected inputs, and restore the session defaults. Valid while idle or generating in an active session; an in-flight clip can delay the reply. Returns `take_changed` and broadcasts `state_update` with cleared image, audio, pose, text, progress and error, seed 420 and clip limit 10000. Select image and audio again before `start`.",
    )
    async def reset(self) -> TakeChanged:
        await self.stop_take()
        self._image = self._audio = self._pose = None
        self.state = LiveAvatarState()
        await self._send_state_update()
        return TakeChanged(action="reset")

    async def inference(self) -> AsyncGenerator[LiveAvatarOutput | None, None]:
        while True:
            if not self.state._running:
                yield None
                continue
            try:
                if self._pending:
                    self._pending = False
                    await asyncio.to_thread(
                        self._backend.start,
                        image=self._image,
                        audio=self._audio,
                        pose=self._pose,
                        prompt=self.state._prompt,
                        negative_prompt=self.state._negative_prompt,
                        seed=self.state._seed,
                        max_chunks=self.state._max_chunks,
                    )
                # Runtime calls handlers between inference turns, not within GPU work.
                result = await asyncio.to_thread(self._backend.next)
                if result is None:
                    await asyncio.to_thread(self._backend.close)
                    self.state._running = False
                    await self.send(GenerationEnded(reason="complete"))
                    await self._send_state_update()
                    yield None
                    continue
                video, audio = result
                self.state._chunks += 1
                self.state._frames += len(video)
                await self.send(
                    ChunkComplete(chunk=self.state._chunks, frames=len(video))
                )
                await self._send_state_update()
                yield LiveAvatarOutput(main_video=video, main_audio=audio)
            except Exception as exc:  # noqa: BLE001 - report upstream failure and release take state
                await asyncio.to_thread(self._backend.close)
                self.state._running = False
                self.state._error = str(exc) or type(exc).__name__
                await self.send(GenerationEnded(reason=self.state._error))
                await self._send_state_update()
                yield None
