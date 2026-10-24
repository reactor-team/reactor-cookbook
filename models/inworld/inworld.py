"""InWorld's segment-boundary interaction policy."""

from __future__ import annotations

import asyncio
from pathlib import Path

import av
from PIL import Image
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

from inworld_images import decode_reference, validate_video
from inworld_model import InWorldAnchor, InWorldInput, InWorldModel, InWorldResult
from inworld_types import InputAccepted, InWorldOutput, InWorldState, StateUpdate


class InWorld(ReactorApp):
    """Generate an audio-video story from text, references or a previous video.

    Stage inputs, then Start. Prompt updates continue the same video at the next
    segment boundary. Pause leaves already-buffered video and audio playing.
    """

    # Preserve the native video clock alongside 48 kHz audio.
    fps = 24
    state: InWorldState

    def load(self, config_path: Path | None) -> None:
        from inworld_assets import prepare_assets

        settings = prepare_assets(get_weights_path(), config_path)
        self._engine = DistributedRunner(
            InWorldModel,
            world_size=settings.num_gpus,
            init_process_group=False,
            load_kwargs={"settings": settings},
            call_timeout=1800,
            start_timeout=1800,
        )
        self._engine.start()

    def generate(self, input: InWorldInput) -> InWorldResult:
        return self._engine.generate(input)

    async def process_input(self) -> InWorldInput:
        if self.state.paused or self.state._anchor is None:
            raise ApplicationError(
                "Choose inputs and press Start, or resume generation"
            )
        if not self.state.prompt.strip():
            raise ApplicationError("Enter a prompt before generating")
        return InWorldInput(
            world_id=self.state._world_id,
            prompt=self.state.prompt,
            anchor=self.state._anchor
            if self.state._applied_world_id != self.state._world_id
            else None,
        )

    async def process_output(self, outcome: StepOutcome) -> InWorldOutput:
        if outcome.error is not None:
            # Invalid client inputs are refused before this step. Unexpected
            # native or worker failures must stop the service, not reset history.
            raise outcome.error
        result: InWorldResult = outcome.result
        self.state._applied_world_id = result.world_id
        self.state._completed = result.chunk_index
        self.state._last_seconds = outcome.elapsed
        await self.send(self._status())
        return InWorldOutput(main_video=result.frames, main_audio=result.audio)

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await self.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        self._engine.reset()
        self.output.flush()

    @event(
        name="set_reference",
        description="Stage one ordered reference for the next Start. Slots must be consecutive from 1. Clears the staged parent video. Fails with invalid_image for an unreadable or oversized image.",
    )
    async def set_reference(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description="Reference image, at most 25 MiB and 20 million pixels.",
        ),
        slot: int = InputField(
            default=1,
            ge=1,
            le=5,
            description="Reference position corresponding to Picture 1 through Picture 5.",
        ),
    ) -> InputAccepted:
        try:
            rgb = await asyncio.to_thread(decode_reference, image.data)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        self.state._references = {**self.state._references, slot: rgb}
        self.state._video = None
        await self.send(self._status())
        return InputAccepted(kind="reference")

    @event(
        name="set_video",
        description="Stage a 24 FPS parent video for the next Start, using its last 17 frames. Clears staged images. Fails with invalid_video for unreadable, oversized, too short or incorrectly timed media.",
    )
    async def set_video(
        self,
        video: UploadedFile = InputField(
            moderate=True,
            description="Parent video, at most 256 MiB and 60 seconds, at least 17 frames, exactly 24 FPS.",
        ),
    ) -> InputAccepted:
        try:
            await asyncio.to_thread(validate_video, video.data)
        except (ValueError, av.FFmpegError) as exc:
            raise CommandError("invalid_video", str(exc)) from exc
        self.state._video = video.data
        self.state._references = {}
        await self.send(self._status())
        return InputAccepted(kind="video")

    @event(
        name="clear_media",
        description="Clear staged media for a text-only Start. Does not change a running video.",
    )
    async def clear_media(self) -> InputAccepted:
        self.state._references = {}
        self.state._video = None
        await self.send(self._status())
        return InputAccepted(kind="clear")

    @event(
        name="set_prompt",
        description="Change the next segment's prompt, preserving the 17-frame visual continuation. Leaves a paused video paused. Buffered media keeps playing. Fails with prompt_required when empty.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            moderate=True,
            max_length=8192,
            description="Scene, character, motion, sound and dialogue description.",
        ),
    ) -> InputAccepted:
        if not prompt.strip():
            raise CommandError("prompt_required", "Enter a non-empty prompt")
        self.state.prompt = prompt.strip()
        await self.send(self._status())
        return InputAccepted(kind="prompt")

    @event(
        name="set_paused",
        description="Pause or resume at the next segment boundary. Buffered media keeps playing; before Start, resuming still waits for inputs.",
    )
    async def set_paused(
        self,
        paused: bool = InputField(
            default=True, description="True pauses; false resumes the current video."
        ),
    ) -> InputAccepted:
        self.state.paused = paused
        await self.send(self._status())
        return InputAccepted(kind="paused")

    @event(
        name="start",
        description="Start a fresh continuous video from staged media and the current prompt. Fails with prompt_required when empty or missing_reference_slot when image slots have gaps.",
    )
    async def start(
        self,
        seed: int = InputField(
            default=42,
            ge=0,
            le=2147483647,
            description="Native random seed, reused per segment.",
        ),
        new_frames: int = InputField(
            default=120,
            choices=[24, 48, 72, 120, 240, 360, 464],
            description="New frames per continuation at 24 FPS; first segment adds one opening frame.",
        ),
    ) -> InputAccepted:
        if not self.state.prompt.strip():
            raise CommandError("prompt_required", "Enter a non-empty prompt")
        refs = self.state._references
        if sorted(refs) != list(range(1, len(refs) + 1)):
            raise CommandError(
                "missing_reference_slot",
                "Use consecutive reference slots starting at 1",
            )
        self.state._anchor = InWorldAnchor(
            references=tuple(refs[i] for i in sorted(refs)),
            video=self.state._video,
            seed=seed,
            new_frames=new_frames,
        )
        self.state._world_id += 1
        self.state._completed = 0
        self.state._last_seconds = None
        self.state.paused = False
        self.output.flush()
        await self.send(self._status())
        return InputAccepted(kind="start")

    def _status(self) -> StateUpdate:
        return StateUpdate(
            world_id=self.state._world_id,
            completed_segments=self.state._completed,
            paused=self.state.paused,
            prompt=self.state.prompt,
            reference_count=len(self.state._references),
            has_video=self.state._video is not None,
            last_segment_seconds=self.state._last_seconds,
        )
