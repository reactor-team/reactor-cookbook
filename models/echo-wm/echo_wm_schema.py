"""Define Echo-WM's public Reactor state, media tracks, and messages."""

from __future__ import annotations

from typing import Literal

from reactor_runtime import (
    Audio,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class EchoAudio(Audio):
    """Declare Echo-WM's 48 kHz generated audio track."""

    sample_rate = 48_000


class EchoWMOutput(Output):
    """Carry one synchronized generated video-and-audio chunk."""

    main_video: Video
    main_audio: EchoAudio


class EchoWMState(InputState):
    """Expose shared scene and camera controls for one world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Scene, character, perspective, sound, music, and speech description. "
            "Use `set_prompt` to replace it; a change starts a fresh world from the "
            "selected image because one prompt conditions an entire rollout."
        ),
    )
    _forward: float = 0.0
    _strafe: float = 0.0
    _pitch: float = 0.0
    _yaw: float = 0.0
    _fov_degrees: float = 70.0
    _reset_requested: bool = False


class StateUpdate(ModelMessage):
    """Emitted after connection, state mutation, reset, or chunk completion."""

    image_source: Literal["uploaded", "built_in"] | None = MessageField(
        description=(
            'Source of the first frame: "uploaded", "built_in", or null before an '
            "image is selected."
        )
    )
    image_name: str | None = MessageField(
        description="Selected first-frame filename, or null before image selection."
    )
    prompt: str | None = MessageField(
        description="Prompt for the current or queued world, or null before image selection."
    )
    active_prompt: str | None = MessageField(
        description="Prompt used by the latest completed chunk, or null before generation."
    )
    seed: int = MessageField(description="Random seed for the current or queued world.")
    reset_queued: bool = MessageField(
        description="Whether a fresh world will be initialized before the next chunk."
    )
    generating: bool = MessageField(
        description="Whether image preparation or one audio-video chunk is in progress."
    )
    completed_chunks: int = MessageField(
        description="Completed audio-video chunks since the latest world reset."
    )
    next_chunk: int | None = MessageField(
        description="One-based chunk that accepted camera state will first affect."
    )
    max_chunks: int = MessageField(
        description="Chunk count that automatically starts a fresh world from the same image."
    )
    forward: float = MessageField(
        description="Active backward-to-forward camera motion."
    )
    strafe: float = MessageField(description="Active left-to-right camera motion.")
    pitch: float = MessageField(description="Active downward-to-upward pitch motion.")
    yaw: float = MessageField(description="Active left-to-right yaw motion.")
    fov_degrees: float = MessageField(
        description="Active horizontal field of view in degrees."
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image queues a fresh world."""

    source: Literal["uploaded", "built_in"] = MessageField(
        description="Image source accepted by the command."
    )
    filename: str = MessageField(description="Selected first-frame filename.")
    prompt: str = MessageField(description="Non-empty prompt for the fresh world.")
    seed: int = MessageField(description="Random seed for the fresh world.")


class PromptQueued(ModelMessage):
    """Emitted when a prompt queues a fresh world from the selected image."""

    prompt: str = MessageField(description="Trimmed prompt accepted by `set_prompt`.")
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the prompt; always 1 after the reset."
    )


class CameraMotionChanged(ModelMessage):
    """Emitted when camera motion or field of view changes."""

    forward: float = MessageField(description="Active backward-to-forward motion.")
    strafe: float = MessageField(description="Active left-to-right motion.")
    pitch: float = MessageField(description="Active downward-to-upward pitch motion.")
    yaw: float = MessageField(description="Active left-to-right yaw motion.")
    fov_degrees: float = MessageField(description="Active horizontal field of view.")
    applies_to_chunk: int | None = MessageField(
        description="First chunk expected to sample these values, or null without an image."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when `reset` queues a fresh world from the selected image."""

    seed: int = MessageField(description="Random seed for the fresh world.")
    replaced_chunks: int = MessageField(
        description="Completed chunks discarded by the reset."
    )


class ChunkCompleted(ModelMessage):
    """Emitted once after one synchronized audio-video chunk finishes."""

    chunk: int = MessageField(description="One-based chunk that completed.")
    video_frames: int = MessageField(description="RGB frames carried by `main_video`.")
    audio_samples: int = MessageField(
        description="48 kHz mono samples carried by `main_audio`."
    )
    generation_seconds: float = MessageField(
        description="Wall-clock seconds spent generating and decoding the chunk."
    )
    denoise_seconds: float | None = MessageField(
        description="CUDA seconds spent in the four denoising transformer forwards."
    )
    cache_commit_seconds: float | None = MessageField(
        description="CUDA seconds spent committing clean audio-video KV cache entries."
    )
    video_decode_seconds: float | None = MessageField(
        description="CUDA seconds spent decoding the video latent block."
    )
    audio_decode_seconds: float | None = MessageField(
        description="CUDA seconds spent decoding and vocoding the audio latent block."
    )
    cuda_total_seconds: float | None = MessageField(
        description="Total CUDA seconds from denoising through synchronized media decode."
    )
    prompt: str = MessageField(description="Prompt sampled by the completed chunk.")
    forward: float = MessageField(description="Forward motion sampled by the chunk.")
    strafe: float = MessageField(description="Strafe motion sampled by the chunk.")
    pitch: float = MessageField(description="Pitch motion sampled by the chunk.")
    yaw: float = MessageField(description="Yaw motion sampled by the chunk.")


class AutomaticResetQueued(ModelMessage):
    """Emitted after the final chunk queues a fresh bounded rollout."""

    completed_chunks: int = MessageField(
        description="Chunks completed when the configured bound was reached."
    )
    max_chunks: int = MessageField(
        description="Configured chunk bound that was reached."
    )
    seed: int = MessageField(description="Seed retained by the queued fresh world.")
