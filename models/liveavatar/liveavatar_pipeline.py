"""Audio-driven avatar takes; module name avoids upstream's liveavatar package."""

# ruff: noqa: B008 -- InputField defaults declare the Reactor wire schema.
from __future__ import annotations

import asyncio
import io
import subprocess
import tempfile
from pathlib import Path

import liveavatar_assets as assets
from liveavatar_model import (
    LiveAvatarInput,
    LiveAvatarModel,
    LiveAvatarResult,
    TakeConditions,
    TakeFailed,
)
from liveavatar_types import (
    ChunkComplete,
    GenerationEnded,
    InputAccepted,
    LiveAvatarOutput,
    LiveAvatarState,
    StateUpdate,
    TakeChanged,
)
from PIL import Image
from reactor_runtime.distributed import DistributedRunner
from liveavatar_turbo import turbo_plan
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
    session_ended,
    session_started,
)


def _save_reference(data: bytes, temporary: Path, destination: Path) -> None:
    """Validate and convert on the command's CPU preparation thread."""
    with Image.open(io.BytesIO(data)) as decoded:
        if decoded.width * decoded.height > 40_000_000:
            raise ValueError("Image exceeds 40 million pixels")
        decoded.convert("RGB").save(temporary)
        temporary.replace(destination)


class LiveAvatar(ReactorApp):
    state: LiveAvatarState
    fps = 25
    buffer_size = 48

    def __init__(self):
        super().__init__()
        self._engine: DistributedRunner | None = None
        self._settings = None
        self._work: Path | None = None

    def load(self, config_path: Path | None = None):
        from reactor_runtime import get_weights_path

        weights_root = get_weights_path()
        settings = assets.prepare_settings(weights_root)
        self._work = settings.work
        self._settings = settings
        self._start_engine()

    def _start_engine(self) -> None:
        if self._engine is None:
            runner = DistributedRunner(
                LiveAvatarModel,
                world_size=turbo_plan(self._settings.turbo)["world_size"],
                load_kwargs={"settings": self._settings},
                start_timeout=3600,
                call_timeout=660,
            )
            runner.start()
            self._engine = runner

    def _release_engine(self) -> None:
        if self._engine is not None:
            # Cancellation must tear down all ranks together: a native TPP
            # continuation can be blocked in CUDA send/recv or awaiting demand.
            self._engine.shutdown()
            self._engine = None

    @session_started
    async def on_session_started(self):
        if self._work is None:
            raise RuntimeError("LiveAvatar was not loaded")
        self.state._directory = tempfile.TemporaryDirectory(
            prefix="session-", dir=self._work
        )
        self.state._image = self.state._audio = self.state._pose = None
        self.state._applied_take_id = None

    @session_ended
    async def on_session_ended(self):
        try:
            await asyncio.to_thread(self._release_engine)
        finally:
            if self.state._directory is not None:
                self.state._directory.cleanup()
                self.state._directory = None
            self.state._image = self.state._audio = self.state._pose = None
            self.state._applied_take_id = None

    @connected
    async def on_connected(self, client: ClientInfo):
        await client.send(StateUpdate.from_state(self.state))

    def _require_idle(self):
        if self.state._running:
            raise CommandError(
                "take_running", "Use `stop` before replacing take conditions."
            )

    def _path(self, name: str) -> Path:
        if self.state._directory is None:
            raise CommandError("session_required", "A session must be active.")
        return Path(self.state._directory.name) / name

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
            await asyncio.to_thread(
                _save_reference,
                image.data,
                self._path("reference-new.png"),
                self._path("reference.png"),
            )
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        self.state._image = self._path("reference.png")
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
        try:
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
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CommandError(
                "invalid_audio", "Audio conversion failed or timed out"
            ) from error
        finally:
            raw.unlink(missing_ok=True)
        if result.returncode:
            raise CommandError("invalid_audio", "Cannot decode the uploaded audio")
        import soundfile as sf

        try:
            info = sf.info(target)
        except (RuntimeError, OSError) as error:
            raise CommandError(
                "invalid_audio", "Converted audio is unreadable"
            ) from error
        if info.duration < 48 / 25:
            raise CommandError(
                "audio_too_short",
                "Supply at least 1.92 seconds of audio for one native clip",
            )
        target.replace(self._path("driving.wav"))
        self.state._audio = self._path("driving.wav")
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
            self.state._pose = None
            self.state._pose_name = None
        else:
            if not pose_video.data or pose_video.size > 100 * 1024**2:
                raise CommandError(
                    "invalid_pose", "Pose video must be nonempty and at most 100 MiB"
                )
            path = self._path("pose-new.mp4")
            path.write_bytes(pose_video.data)
            try:
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
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise CommandError(
                    "invalid_pose", "Video inspection failed or timed out"
                ) from error
            if result.returncode or not result.stdout.strip():
                raise CommandError(
                    "invalid_pose", "Upload must contain a readable video track"
                )
            path.replace(self._path("pose.mp4"))
            self.state._pose = self._path("pose.mp4")
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
        if self.state._image is None or self.state._audio is None:
            raise CommandError(
                "inputs_required", "Upload reference image and driving audio first"
            )
        await asyncio.to_thread(self._start_engine)
        self.state._running = True
        self.state._chunks = self.state._frames = 0
        self.state._error = None
        self.state._take_id += 1
        await self._send_state_update()
        return TakeChanged(action="start")

    @event(
        name="stop",
        description="End the take and clear queued playback while retaining selected inputs, options and progress for inspection or another `start`. Valid while idle or generating in an active session. Returns `take_changed` and broadcasts `state_update`. Handled between inference turns, so an in-flight clip can delay the reply. Explicit stopping is reported by this reply; `generation_ended` reports automatic completion or failure.",
    )
    async def stop_take(self) -> TakeChanged:
        await asyncio.to_thread(self._release_engine)
        self.state._running = False
        self.state._applied_take_id = None
        self.output.flush()
        await self._send_state_update()
        return TakeChanged(action="stop")

    @event(
        name="reset",
        description="End the take, clear queued playback and selected inputs, and restore the session defaults. Valid while idle or generating in an active session; an in-flight clip can delay the reply. Returns `take_changed` and broadcasts `state_update` with cleared image, audio, pose, text, progress and error, seed 420 and clip limit 10000. Select image and audio again before `start`.",
    )
    async def reset(self) -> TakeChanged:
        await asyncio.to_thread(self._release_engine)
        self.output.flush()
        directory = self.state._directory
        take_id = self.state._take_id
        self.state = LiveAvatarState()
        self.state._directory = directory
        self.state._take_id = take_id
        await self._send_state_update()
        return TakeChanged(action="reset")

    async def process_input(self) -> LiveAvatarInput:
        if self.state is None or not self.state._running:
            raise ApplicationError("Upload image and audio, then start a take")
        conditions = None
        if self.state._take_id != self.state._applied_take_id:
            if self.state._image is None or self.state._audio is None:
                raise ApplicationError("A take requires an uploaded image and audio")
            conditions = TakeConditions(
                image=self.state._image,
                audio=self.state._audio,
                pose=self.state._pose,
                prompt=self.state._prompt,
                negative_prompt=self.state._negative_prompt,
                seed=self.state._seed,
                max_chunks=self.state._max_chunks,
            )
        return LiveAvatarInput(take_id=self.state._take_id, conditions=conditions)

    def generate(self, input: LiveAvatarInput) -> LiveAvatarResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> LiveAvatarOutput | None:
        if isinstance(outcome.error, TakeFailed):
            await asyncio.to_thread(self._release_engine)
            self.state._running = False
            self.state._error = str(outcome.error)
            await self.send(GenerationEnded(reason=self.state._error))
            await self.send(StateUpdate.from_state(self.state))
            return None
        if outcome.error is not None:
            # Contract or device failures outside the native take are not recoverable.
            raise outcome.error
        result: LiveAvatarResult = outcome.result
        self.state._applied_take_id = result.take_id
        self.state._chunks, self.state._frames = result.chunks, result.frames
        if result.complete:
            await asyncio.to_thread(self._engine.reset)
            self.state._running = False
            await self.send(GenerationEnded(reason="complete"))
            await self.send(StateUpdate.from_state(self.state))
            return None
        await self.send(ChunkComplete(chunk=result.chunks, frames=len(result.video)))
        await self.send(StateUpdate.from_state(self.state))
        return LiveAvatarOutput(main_video=result.video, main_audio=result.audio)
