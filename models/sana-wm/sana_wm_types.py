"""Define SANA-WM configuration and the public Reactor schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

Control = Literal[
    "forward",
    "back",
    "strafe_left",
    "strafe_right",
    "yaw_left",
    "yaw_right",
    "pitch_up",
    "pitch_down",
]
ImageSource = Literal["uploaded", "built_in"]
IntrinsicsSource = Literal["uploaded", "built_in", "estimated"]
ControlMode = Literal["interactive", "trajectory"]


@dataclass(frozen=True)
class HubAsset:
    """Describe one public Hugging Face asset at an immutable revision."""

    repo_id: str
    revision: str


@dataclass(frozen=True)
class BuiltInScene:
    """Describe one first-frame, prompt, and camera-calibration example."""

    name: str
    image: Path
    prompt: Path
    intrinsics: Path


@dataclass(frozen=True)
class SanaWMConfig:
    """Hold validated source, asset, rollout, and motion settings."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    streaming: HubAsset
    stage1_text_encoder: HubAsset
    pi3x_model: HubAsset
    pi3x_source_url: str
    pi3x_source_revision: str
    scenes: tuple[BuiltInScene, ...]
    seed: int
    max_chunks: int
    num_cached_blocks: int
    refiner_kv_max_frames: int
    translation_speed: float
    rotation_speed_degrees: float
    pitch_limit_degrees: float


class SanaWMOutput(Output):
    """Stream one generated SANA-WM RGB frame on `main_video`."""

    main_video: Video


class SanaWMState(InputState):
    """Expose shared prompt, camera, and playback state for one world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description=(
            "Scene description used for the next fresh rollout. SANA-WM encodes text once "
            "when its autoregressive caches are initialized."
        ),
    )
    _trajectory_exhausted: bool = False
    _held_controls: set[str] = field(default_factory=set)
    _reset_requested: bool = False


class StateUpdate(ModelMessage):
    """Emitted when observable world state changes or a viewer connects."""

    image_source: ImageSource | None = MessageField(
        description="Selected first-frame source, or null before an image is selected."
    )
    image_name: str | None = MessageField(
        description="Selected first-frame filename, or null before an image is selected."
    )
    intrinsics_source: IntrinsicsSource | None = MessageField(
        description=(
            "Camera calibration source for the selected image: uploaded NumPy data, a "
            "built-in calibration, Pi3X estimation, or null before image selection."
        )
    )
    prompt: str | None = MessageField(
        description="Non-empty prompt for the current or queued fresh world."
    )
    active_prompt: str | None = MessageField(
        description="Prompt encoded into the active autoregressive and refiner caches."
    )
    control_mode: ControlMode = MessageField(
        description=(
            "Camera source for the next chunk: live canonical controls or an uploaded "
            "camera-to-world trajectory."
        )
    )
    trajectory_name: str | None = MessageField(
        description=(
            "Uploaded camera-trajectory filename while `control_mode` is `trajectory`, "
            "or null while interactive controls are active."
        )
    )
    trajectory_frames: int | None = MessageField(
        description=(
            "Number of poses in the uploaded camera trajectory, or null while interactive "
            "controls are active."
        )
    )
    held_controls: list[Control] = MessageField(
        description=(
            "Canonical SANA-WM controls currently held. Multiple controls can be active at "
            "the same time and are sampled at the next chunk boundary."
        )
    )
    seed: int = MessageField(
        description="Random seed used by the current or queued rollout."
    )
    trajectory_exhausted: bool = MessageField(
        description=(
            "Whether the uploaded camera trajectory ran out of complete 24-frame chunks "
            "and generation stopped. Selecting an image or calling `reset` starts a "
            "fresh rollout."
        )
    )
    reset_queued: bool = MessageField(
        description="Whether the selected image, prompt, and seed will initialize fresh caches."
    )
    generating: bool = MessageField(
        description="Whether cache initialization or one chunk of GPU inference is running."
    )
    completed_chunks: int = MessageField(
        description="Number of completed 24-frame chunks in the active rollout."
    )
    next_chunk: int | None = MessageField(
        description="One-based chunk that accepted controls will affect, or null without an image."
    )
    max_chunks: int = MessageField(
        description="Chunk count that triggers a fresh bounded rollout from the selected image."
    )


class ImageSelected(ModelMessage):
    """Emitted when an uploaded or built-in image queues a fresh world."""

    source: ImageSource = MessageField(description="Selected first-frame source.")
    filename: str = MessageField(description="Selected first-frame filename.")
    prompt: str = MessageField(description="Effective prompt for the fresh world.")
    intrinsics_source: IntrinsicsSource = MessageField(
        description="Calibration source used for the fresh world."
    )


class PromptChanged(ModelMessage):
    """Emitted when text queues a fresh prompt-conditioned rollout."""

    prompt: str = MessageField(
        description="Trimmed prompt queued for fresh cache initialization."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first use the prompt; always 1 after the reset."
    )


class ControlChanged(ModelMessage):
    """Emitted after one canonical held control changes."""

    control: Control = MessageField(
        description="Canonical SANA-WM control that changed."
    )
    pressed: bool = MessageField(description="Whether the control is now held.")
    held_controls: list[Control] = MessageField(
        description="Complete canonical control set after the change."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk expected to sample the held controls."
    )


class ControlsReleased(ModelMessage):
    """Emitted after all held camera controls return to neutral."""

    applies_to_chunk: int = MessageField(
        description="One-based chunk expected to observe released controls."
    )


class TrajectorySelected(ModelMessage):
    """Emitted when an uploaded native camera trajectory replaces live controls."""

    filename: str = MessageField(description="Uploaded NumPy trajectory filename.")
    frames: int = MessageField(
        description="Number of camera-to-world poses available, including the initial pose."
    )
    available_chunks: int = MessageField(
        description="Complete 24-frame chunks available after the initial pose."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when fresh autoregressive caches are requested."""

    trigger: Literal["manual", "prompt", "automatic_chunk_limit"] = MessageField(
        description="Reason the current rollout will be replaced."
    )
    seed: int = MessageField(description="Seed used by the fresh rollout.")
    replaced_chunks: int = MessageField(
        description="Number of completed chunks in the rollout being replaced."
    )


class TrajectoryExhausted(ModelMessage):
    """Emitted when an uploaded finite camera trajectory has no complete chunk left."""

    completed_chunks: int = MessageField(
        description="Number of chunks generated before the trajectory ended."
    )
    trajectory_frames: int = MessageField(
        description="Number of poses supplied by the uploaded trajectory."
    )
