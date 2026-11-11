"""StreamDiffusionV2 client contract: the session state, the tracks, and every message."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from reactor_runtime import InputState, MediaInput, MessageField, ModelMessage, Output, Video

TRAINED_FPS = 16
PROMPT_TOKEN_BUDGET = 512  # umT5 reads at most 512 tokens
SEED_MAX = 2**31 - 1


class StreamDiffusionV2Media(MediaInput):
    """The video the model rewrites."""

    video_in: Video


class StreamDiffusionV2Output(Output):
    """The rewritten video: 480x832, 4 frames per chunk (5 in a run's first)."""

    main_video: Video


class StreamDiffusionV2State(InputState):
    """Session values the application owns; every client input has its own command."""

    _phase: str = "idle"  # idle | running | paused
    _prompt: str | None = None
    _seed: int | None = None
    _noise_scale: float | None = None
    _run_id: int = 0  # bumped by every start
    _started: bool = False  # the run's first chunk has been rewritten
    _applied_prompt: str | None = None
    _applied_seed: int | None = None
    _applied_noise_scale: float | None = None
    _pending: deque[np.ndarray] | None = None  # input frames not yet rewritten, at 16 fps
    _last_capture_us: int | None = None
    _chunk_index: int = -1


class StateUpdate(ModelMessage):
    """Emitted on connect, as the answer to every command, and when a new prompt comes into use."""

    state: str = MessageField(description="`idle`, `running` or `paused`.")
    settings: dict = MessageField(
        description=(
            "Each input's value last sent (`sent`) and the value applied at the last chunk "
            "(`applied`): `prompt`, `seed`, `noise_scale`. `noise_scale.applied` is the strength the "
            "last chunk used after the model's per-chunk adaptation to motion."
        )
    )
    capabilities: dict = MessageField(
        description="Every command with its kind (serving config, session setup, live control) and accepted values."
    )
    frames_waiting: int = MessageField(
        description="Input frames received and not yet rewritten; the first chunk needs 5, each later one 4."
    )
    chunk_index: int = MessageField(description="The last completed chunk of the run, or -1 before the first.")


class ChunkComplete(ModelMessage):
    """Emitted after every chunk the model ran, before its frames."""

    index: int = MessageField(description="The chunk's index within the run, from 0.")
    frames_out: int = MessageField(
        description="Frames this chunk emitted: 5 for chunk 0, 0 for chunk 1 (the stream batch fills), 4 after."
    )
    generate_seconds: float = MessageField(description="Wall-clock seconds the model took for this chunk.")
    noise_scale: float = MessageField(description="The noise strength this chunk used, after the per-chunk adaptation to motion.")


class GenerationStarted(ModelMessage):
    """Emitted when a run's first chunk has been rewritten."""


class GenerationComplete(ModelMessage):
    """Emitted when `reset` ends a run that was running or paused."""


CAPABILITIES: dict[str, Any] = {
    "video_in": {"kind": "live control", "accepts": "video track; resized (stretched) to 832x480; resampled to 16 fps by capture time when frames carry one"},
    "set_prompt": {"kind": "live control", "accepts": f"text, up to {PROMPT_TOKEN_BUDGET} umT5 tokens; empty is a value", "applies": "next chunk; the self-attention cache is kept"},
    "set_seed": {"kind": "session setup", "accepts": f"integer 0..{SEED_MAX}, or -1 for random"},
    "set_noise_scale": {"kind": "session setup", "accepts": "0.0..1.0 (the source example uses 0.8)", "means": "starting noise strength; higher departs further from the input. The model adapts it per chunk to motion."},
    "start": {"kind": "lifecycle"}, "pause": {"kind": "lifecycle"}, "resume": {"kind": "lifecycle"}, "reset": {"kind": "lifecycle"},
    "serving_config": {"steps": 2, "timesteps": [700, 500], "resolution": "480x832", "dtype": "bfloat16", "vae": "Wan2.1 VAE", "trained_fps": TRAINED_FPS},
}
