# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

"""Everything a JoyAI-Video-Edit client sees: the tracks, the state, and the messages.

The application half (joy_video_edit.py) imports everything from here.
"""

from typing import Any

import numpy as np
from reactor_runtime import (
    InputField,
    InputState,
    MediaInput,
    MessageField,
    ModelMessage,
    Output,
    UploadedFile,
    Video,
)


# ---------------------------------------------------------------------------
# Output track
# ---------------------------------------------------------------------------

class JoyVideoEditOutput(Output):
    """Edited video frames streamed back to the client, chunk by chunk."""
    main_video: Video


# ---------------------------------------------------------------------------
# Input track (live camera — the video to be edited)
# ---------------------------------------------------------------------------

class JoyVideoEditMedia(MediaInput):
    """Live client video stream (webcam or uploaded video)."""
    camera: Video


# ---------------------------------------------------------------------------
# Client-controllable state (InputState)
# ---------------------------------------------------------------------------

class JoyVideoEditState(InputState):
    # ---- Public: client-settable via set_<field> ----

    prompt: str = InputField(
        default="",
        max_length=2000,
        description=(
            "Edit instruction, in natural language (for example 'Turn the video into watercolor "
            "style'). `start` reads it and it stays fixed for that run."
        ),
        moderate=True,
    )

    reference_image: UploadedFile = InputField(
        default=None,
        description=(
            "Optional reference image whose appearance guides the edit. Set it while not "
            "generating; leave it unset to edit from the instruction alone."
        ),
        moderate=True,
    )

    seed: int = InputField(
        default=42,
        ge=0,
        description=(
            "Seed for the random noise each run starts from. `start` reads it, so a change "
            "while generating applies from the next `start`."
        ),
    )

    # ---- Private: session scratch the application owns (invisible to clients) ----

    # The reference image, decoded to (H, W, 3) uint8 RGB by `set_reference_image`.
    _reference: np.ndarray | None = None

    # Set by `start`; cleared when the run ends (`reset`, the last client leaving).
    _started: bool = False

    # The run `start` opened, the run the model last reported holding, and what the run is
    # conditioned on (a RunConditioning, snapshotted on the run's first step).  The
    # conditioning rides on a step input only while the two ids differ.
    _run_id: int = 0
    _applied_run_id: int | None = None
    _conditioning: Any = None

    # How many camera frames the next step must carry, as the last step result asked.
    _frames_wanted: int = 1

    # Chunks and frames emitted on `main_video` in the current run.
    _chunk_index: int = 0
    _total_frames: int = 0


# ---------------------------------------------------------------------------
# Outbound messages (model → client)
# ---------------------------------------------------------------------------

class GenerationStarted(ModelMessage):
    """Emitted once when `start` begins a run."""
    prompt: str = MessageField(description="The edit instruction this run uses; empty when none is set.")
    has_reference_image: bool = MessageField(description="True if this run uses a reference image.")
    seed: int = MessageField(description="The seed this run uses.")


class ChunkComplete(ModelMessage):
    """Emitted once per edited chunk, just before its frames go out on `main_video`."""
    chunk_index: int = MessageField(
        description="Zero-based index of the chunk within the run; starts again at 0 on each `start`."
    )
    frames_emitted: int = MessageField(
        description=(
            "Frames in this chunk: usually 8. The run's first chunk, and an occasional chunk "
            "after it, carry 1."
        )
    )
    elapsed_ms: float = MessageField(
        description=(
            "Server-side time, in milliseconds, spent editing this chunk. `main_video` plays the "
            "chunk out at `frames_emitted` frames over this time."
        )
    )


class GenerationComplete(ModelMessage):
    """Emitted when a run ends: after `generation_reset` on `reset`, or when the last client disconnects."""
    total_chunks: int = MessageField(description="Chunks emitted on `main_video` during the run.")
    total_frames: int = MessageField(description="Frames emitted on `main_video` during the run.")


class GenerationReset(ModelMessage):
    """Emitted when `reset` stops a run in progress, before `generation_complete`."""
    reason: str = MessageField(description="Why the run stopped. Currently always `client requested`.")


class PromptAccepted(ModelMessage):
    """Emitted when `set_prompt` stores a new edit instruction."""
    prompt: str = MessageField(description="The instruction now stored; the next `start` reads it.")


class ReferenceImageAccepted(ModelMessage):
    """Emitted when `set_reference_image` successfully stores a reference image."""
    width: int = MessageField(description="Width of the uploaded image, in pixels, before it is fitted.")
    height: int = MessageField(description="Height of the uploaded image, in pixels, before it is fitted.")


class CommandError(ModelMessage):
    """Emitted to the sending client when a command is rejected; nothing changes."""
    command: str = MessageField(description="Wire name of the rejected command: `start` or `set_reference_image`.")
    reason: str = MessageField(description="Human-readable reason the command was rejected.")


class SessionState(ModelMessage):
    """Emitted on connect, after `set_prompt`, `set_reference_image`, `start` and `reset`, and when a run ends."""
    started: bool = MessageField(
        description="True while generating: from `start` until `reset` or the last client disconnects."
    )
    prompt: str = MessageField(description="The stored edit instruction; empty when none is set.")
    has_reference_image: bool = MessageField(
        description="True once a reference image is loaded; it stays loaded across `reset`."
    )
    seed: int = MessageField(description="The stored seed; `start` reads it.")
