"""Define Matrix-Game-2.0 configuration and public Reactor schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

KEYBOARD_KEYS = ["w", "a", "s", "d"]


@dataclass(frozen=True)
class MatrixGame2Config:
    """Hold validated source, checkpoint, and rollout settings."""

    checkout_path: Path
    source_path: Path
    source_url: str
    source_revision: str
    model_repo_id: str
    model_revision: str
    model_cache: Path
    checkpoint_file: str
    seed: int
    max_latent_frames: int
    random_images: tuple[Path, ...]

    @property
    def max_chunks(self) -> int:
        """Return the number of native three-latent chunks in one rollout."""
        return self.max_latent_frames // 3


class MatrixGame2Output(Output):
    """Stream one generated Matrix-Game-2.0 RGB frame batch on `main_video`."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted after connection, accepted commands, and completed chunks."""

    image_source: str = MessageField(
        description=(
            "Source of the selected starting image: `built_in`, `uploaded`, or `none` before "
            "an image has been selected."
        )
    )
    image_name: str = MessageField(
        description=(
            "Filename of the selected starting image, or an empty string before selection. "
            "The image becomes active when a queued reset begins."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed used by the current or queued rollout. A non-negative `seed` passed "
            "to `reset` replaces it."
        )
    )
    reset_queued: bool = MessageField(
        description=(
            "Whether the selected image is waiting to initialize a fresh autoregressive "
            "rollout before another chunk is generated."
        )
    )
    chunk_in_flight: bool = MessageField(
        description=(
            "Whether the GPU is currently generating one three-latent chunk. Commands "
            "accepted while true apply at the following chunk boundary."
        )
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether the official 360-latent rollout horizon is exhausted. When true, "
            "`reset`, `set_image`, or `random_image` is required before generation continues."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of native three-latent chunks completed in the active rollout. It returns "
            "to zero when a fresh image-conditioned rollout begins."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk expected to consume controls accepted now, or null before image "
            "selection and after the rollout horizon is exhausted."
        )
    )
    max_chunks: int = MessageField(
        description=(
            "Number of three-latent chunks available in the official 360-latent rollout "
            "horizon. The universal distilled checkpoint uses 120."
        )
    )
    last_chunk_frames: int = MessageField(
        description=(
            "Number of RGB frames emitted by the most recently completed chunk. It is 9 for "
            "the first causal VAE decode, 12 thereafter, and 0 before generation."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Sorted WASD keys held for `next_chunk`. Multiple keys produce the official "
            "multi-hot keyboard condition, such as `w` plus `a` for forward-left."
        )
    )
    pitch: float = MessageField(
        description=(
            "Normalized look-down (-1) to look-up (1) velocity held for `next_chunk`; zero "
            "is neutral."
        )
    )
    yaw: float = MessageField(
        description=(
            "Normalized turn-left (-1) to turn-right (1) velocity held for `next_chunk`; "
            "zero is neutral."
        )
    )


class ChunkComplete(ModelMessage):
    """Emitted after one native autoregressive chunk finishes on the GPU."""

    chunk: int = MessageField(
        description="One-based index of the chunk that finished in the active rollout."
    )
    frames: int = MessageField(
        description=(
            "Number of decoded RGB frames in this chunk: 9 for chunk 1 and 12 for later "
            "chunks."
        )
    )
    inference_seconds: float = MessageField(
        description=(
            "Wall-clock seconds spent generating and causally decoding this chunk, excluding "
            "video playout time."
        )
    )
    pressed_keys: list[str] = MessageField(
        description="Sorted WASD keys sampled as a multi-hot condition for this chunk."
    )
    pitch: float = MessageField(
        description="Normalized pitch velocity sampled for this completed chunk."
    )
    yaw: float = MessageField(
        description="Normalized yaw velocity sampled for this completed chunk."
    )


class ActionChanged(ModelMessage):
    """Emitted when `set_key_state` changes one held WASD key."""

    key: str = MessageField(description="WASD key addressed by the control event.")
    pressed: bool = MessageField(
        description="Whether the addressed key is now held for forthcoming chunks."
    )
    pressed_keys: list[str] = MessageField(
        description="Complete sorted WASD key set held after applying the event."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample the updated multi-hot key state."
    )


class CameraMotionChanged(ModelMessage):
    """Emitted when a continuous camera event changes forthcoming motion."""

    pitch: float = MessageField(
        description="Look-down (-1) to look-up (1) velocity held for forthcoming chunks."
    )
    yaw: float = MessageField(
        description="Turn-left (-1) to turn-right (1) velocity held for forthcoming chunks."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample both camera values."
    )


class RolloutLimitReached(ModelMessage):
    """Emitted when generation idles at the official latent horizon."""

    completed_chunks: int = MessageField(
        description="Number of completed chunks when the rollout reached its horizon."
    )
    max_chunks: int = MessageField(
        description=(
            "Configured chunk horizon now exhausted; select an image or call `reset` to start "
            "a fresh rollout."
        )
    )


class MatrixGame2State(InputState):
    """Expose shared Matrix keyboard and mouse-camera controls."""

    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Normalized look-down (-1) to look-up (1) velocity sampled at chunk boundaries "
            "and held until `set_pitch` changes it or controls are released."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Normalized turn-left (-1) to turn-right (1) velocity sampled at chunk boundaries "
            "and held until `set_yaw` changes it or controls are released."
        ),
    )
    _pressed_keys: frozenset[str] = frozenset()
    _restart_requested: bool = False
    _limit_reached: bool = False
