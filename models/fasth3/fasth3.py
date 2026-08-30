"""FastH3 as a Reactor platform model: an endless, prompt-driven A/V channel.

FastH3 is MiniMax-H3 distilled to four transformer forwards, and on eight
Blackwell GPUs it builds video faster than the video plays. This model turns
that into a continuous stream: it always builds the next clip while the current
one is on the wire, so `main_video` and `main_audio` never run dry.

The unit of work is a whole clip, not a frame, which is why this subclasses
``ReactorModel`` and owns its own ``run()`` loop rather than using
``ReactorPipeline``. Command handlers then run on their own coroutines
concurrent with ``run()``, so `pause` and `stop` answer immediately even while a
clip is being built.

Layout:
  * ``fasth3_types.py``        — everything a client sees (tracks and messages).
  * ``fasth3_clip_plan.py``    — clip geometry (lengths, frame counts, canvases).
  * ``fasth3_session_rules.py`` — which commands each state accepts.
  * ``fasth3.yaml``            — the generation recipe and the weight layout.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from reactor_runtime import (
    ClientInfo,
    InputField,
    ReactorModel,
    connected,
    event,
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

import fasth3_clip_plan as clip_plan
import fasth3_session_rules as session_rules
from fasth3_types import (
    MAX_PROMPT_CHARS,
    CanvasAccepted,
    ChannelFailed,
    ChannelPaused,
    ChannelReset,
    ChannelResumed,
    ChannelStarted,
    ChannelStopped,
    ClipComplete,
    ClipLengthAccepted,
    ClipStarted,
    CommandError,
    FastH3Output,
    PromptAccepted,
    SeedAccepted,
    StateUpdate,
)

logger = get_logger(__name__)

FRAME_RATE = clip_plan.FPS

# The clip-length range, rendered once so the command text and the schema's own
# bounds can never disagree.
_CLIP_RANGE = f"{clip_plan.MIN_SECONDS_PUBLISHED:g} and {clip_plan.MAX_SECONDS_PUBLISHED:g}"

# WebRTC-native rate every clip's waveform is resampled to. The checkpoint's
# audio decoder is 32 kHz; the wire is 48 kHz.
OUTPUT_SAMPLE_RATE = 48_000
NATIVE_SAMPLE_RATE = 32_000

# Frames per emitted slice. The runtime recorder's feed queue cannot absorb
# one-second bursts, and the emitter is a metronome either way, so smaller
# slices cost nothing.
EMIT_FRAMES = 3

# How often the arm loop and the pause hold re-check. Both run on the event
# loop, so this is a scheduling granularity, not a busy-wait.
POLL_SECONDS = 0.05
WORKER_POLL_SECONDS = 0.1

# What the warm-up builds. Never reaches a client: warm-up output is discarded,
# and its only job is to be a syntactically ordinary prompt.
WARMUP_PROMPT = "A slow cinematic shot of sunlight moving across a quiet room."

# The HF snapshot directory inside the weights bundle.
DEFAULT_CHECKPOINT_DIR = "FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"

# Component directories the T2VA pipeline loads. An incomplete bundle must kill
# startup, not surface as a loader traceback on the first clip.
REQUIRED_COMPONENTS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


class _ChannelStopped(Exception):
    """Internal: the generation worker noticed the channel should wind down."""


class _Job:
    """One unit of work for the generation worker, and its outcome.

    The error is carried back rather than only logged: ``load()`` waits on the
    warm-up through this, and a warm-up that failed silently would let the pod
    report ready and then die on its first real clip.
    """

    __slots__ = ("done", "error", "fn")

    def __init__(self, fn) -> None:
        self.fn = fn
        self.done = threading.Event()
        self.error: BaseException | None = None


def _require_weights(root: Path, model_path: Path) -> None:
    """Fail startup loudly when the weights bundle is incomplete."""
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in REQUIRED_COMPONENTS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")
    if problems:
        raise FileNotFoundError(
            f"FastH3 weights bundle under {root} is incomplete:\n  " + "\n  ".join(problems)
        )


class FastH3(ReactorModel):
    """Stream an endless video-and-audio channel from a text prompt."""

    # Pinned: `_emit_paced` is a strict 24 fps metronome and every emit omits
    # `compute_time`, which is exactly the "unmeasured" path this rate tags.
    # Measuring instead re-estimates the rate from observed timing, whose wobble
    # both drops chunks while converging and drifts video against the
    # sample-clocked audio.
    fps = FRAME_RATE
    # Two seconds of transport-side tolerance at 24 fps, so a clip that lands
    # late dents the buffer instead of dropping frames.
    buffer_size = 48

    # ------------------------------------------------------------------ load

    def load(self, config_path: Path | None) -> None:
        """Build the eight-GPU generator and warm every clip shape.

        Runs once at startup, before any session. The runtime marks the pod
        ready only when this returns, so the warm-up below means a deployed pod
        never serves a cold clip.

        Args:
            config_path: Path to ``fasth3.yaml``; its ``inference`` block is the
                generation recipe and its ``runtime`` block holds the weight
                layout and the engine shape.
        """
        document: dict[str, Any] = {}
        if config_path is not None:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.inference_cfg: dict[str, Any] = document.get("inference") or {}
        runtime: dict[str, Any] = document.get("runtime") or {}
        self.runtime_cfg = runtime

        self.default_aspect = str(self.inference_cfg.get("aspect", "16:9"))
        if self.default_aspect not in clip_plan.ASPECT_CHOICES:
            raise ValueError(
                f"inference.aspect must be one of {list(clip_plan.ASPECT_CHOICES)}, "
                f"got {self.default_aspect!r}"
            )
        self.default_clip_frames = clip_plan.frames_for_seconds(
            float(self.inference_cfg.get("clip_seconds", clip_plan.MAX_SECONDS))
        )
        # Time-to-first-frame ramp: the channel opens with these clip lengths
        # before settling on the steady one. Each entry is a distinct shape the
        # warm-up must cover, so keep the list short. An empty list means a
        # uniform cadence from clip zero.
        self.ramp_frames = clip_plan.parse_ramp(
            self.inference_cfg.get("ramp_seconds", [clip_plan.MIN_SECONDS])
        )
        self.default_seed = int(self.inference_cfg.get("seed", 1000))
        # Sigma-grid POINTS, not transformer forwards: the distilled schedule is
        # five points and exactly four forwards.
        self.num_inference_steps = int(self.inference_cfg.get("num_inference_steps", 5))

        # Must happen before the generator is built: the engine spawns worker
        # processes, which inherit os.environ, and these select the attention
        # backend and the sparse kernel.
        self._apply_profile_environment()
        self._validate_profile_dependencies()

        weights = get_weights_path()
        self.model_path = weights / str(runtime.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
        _require_weights(weights, self.model_path)

        self.num_gpus = int(runtime.get("num_gpus", 8))
        logger.info(
            "building fasth3 generator",
            model_path=str(self.model_path),
            num_gpus=self.num_gpus,
            clip_frames=self.default_clip_frames,
            ramp=list(self.ramp_frames),
        )

        from fastvideo import VideoGenerator

        self.generator = VideoGenerator.from_config(self._generator_config())

        # Session-scoped state. A ReactorPipeline would get a fresh `self.state`
        # per session from the runtime; a ReactorModel owns that itself, so the
        # defaults live in one reset function called here (so a command racing
        # ahead of `session_started` reads defaults, never another session's
        # values) and again from the `@session_started` hook.
        self._reset_session_state()

        # One persistent worker thread runs every clip. The GPU work itself
        # lives in the spawned engine processes; this thread exists to serialise
        # submissions and to give teardown a single handle to wait on.
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="fasth3-generation", daemon=True
        )
        self._worker.start()
        self._preload_native_imports()
        self._run_on_worker(self._warmup)
        logger.info("fasth3 loaded")

    @staticmethod
    def _preload_native_imports() -> None:
        """Touch every deferred native import the emit path needs.

        ``_to_wire_audio`` and ``_emit_paced`` import torch, torchaudio and
        numpy lazily so that rendering the schema needs none of them. Lazy means
        the first import happens on the first real clip — after load, after
        warm-up, after the pod reports ready — where a linkage failure is a dead
        session rather than a startup error. The resample below is a real call,
        so it fails here or not at all.
        """
        import numpy  # noqa: F401
        import torch
        import torchaudio.functional as AF

        AF.resample(torch.zeros(2, NATIVE_SAMPLE_RATE // 10), NATIVE_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

    # --------------------------------------------------------------- profile

    def _apply_profile_environment(self) -> None:
        """Set the FastH3 profile environment, exactly as the reference CLI does.

        Mirrors ``examples/inference/basic/basic_fasth3.py:profile_environment``.
        Values are explicit even for disabled features, so a shell's inherited
        experiment settings cannot silently change the profile a pod serves.
        """
        cfg = self.inference_cfg
        vsa_kernel = str(cfg.get("vsa_kernel", "sm100a"))
        fusions = "all" if bool(cfg.get("h3_fusions", True)) else "0"
        environment: dict[str, str | None] = {
            "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
            "FASTVIDEO_VSA_SM100A": "1" if vsa_kernel == "sm100a" else "0",
            "FASTVIDEO_VSA_CUTEDSL": "0",
            # A non-empty path enables the diagnostic probe; it must stay unset.
            "FASTVIDEO_H3_VSA_PROBE": None,
            "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
            "FASTVIDEO_FA4": "1" if bool(cfg.get("fa4", True)) else "0",
            "FASTVIDEO_NVFP4_FA4": "0",
            "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": fusions,
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": (
                "1" if bool(cfg.get("inference_torch_compile", True)) else "0"
            ),
            "FASTVIDEO_VAE_PARALLEL_DECODE": (
                "1" if bool(cfg.get("vae_parallel_decode", True)) else "0"
            ),
            "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
            "FASTVIDEO_ULYSSES_A2A": str(cfg.get("ulysses_a2a", "off")),
            "FASTVIDEO_STAGE_LOGGING": "1",
        }
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        logger.info("fasth3 profile", **{k: (v or "<unset>") for k, v in environment.items()})

    def _validate_profile_dependencies(self) -> None:
        """Fail before the 148 GB load when the selected fast route is absent."""
        import importlib.util

        cfg = self.inference_cfg
        if bool(cfg.get("fa4", True)):
            try:
                present = importlib.util.find_spec("flash_attn.cute") is not None
            except (ImportError, ModuleNotFoundError):
                present = False
            if not present:
                raise RuntimeError(
                    "FastH3's FA4 route needs the pinned flash-attn-4 package. Install it, "
                    "or set inference.fa4: false in fasth3.yaml."
                )
        if str(cfg.get("vsa_kernel", "sm100a")) == "sm100a":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
            except ImportError:
                present = False
            else:
                present = bool(getattr(block_sparse_attn_sm100a, "_HAS_VSA_SM100A", False))
            if not present:
                raise RuntimeError(
                    "FastH3's sm100a route needs fastvideo-kernel built with the Blackwell VSA "
                    "extension. Install a matching wheel, or set inference.vsa_kernel: triton."
                )

    def _generator_config(self):
        """The engine shape, mirroring ``basic_fasth3.py:build_generator_config``."""
        from fastvideo.api import (
            CompileConfig,
            ComponentConfig,
            EngineConfig,
            GeneratorConfig,
            OffloadConfig,
            ParallelismConfig,
            PipelineSelection,
        )

        cfg = self.inference_cfg
        runtime = self.runtime_cfg
        num_gpus = int(runtime.get("num_gpus", 8))
        # The checkpoint's own contract (fastvideo_inference.json) shards the
        # transformer with FSDP. Sharding is what frees the VRAM to keep the
        # text encoder resident, which is the deployment this model wants.
        replicated_dit = bool(runtime.get("replicated_dit", False))
        return GeneratorConfig(
            model_path=str(self.model_path),
            pipeline=PipelineSelection(
                components=ComponentConfig(),
                experimental={
                    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                    "VSA_sparsity": float(cfg.get("vsa_sparsity", 0.9)),
                    "VSA_tile_size": int(cfg.get("vsa_tile_size", 64)),
                    "inference_torch_compile": bool(cfg.get("inference_torch_compile", True)),
                    "vae_parallel_decode": bool(cfg.get("vae_parallel_decode", True)),
                    "vae_parallel_decode_strategy": "gather",
                },
            ),
            engine=EngineConfig(
                num_gpus=num_gpus,
                use_fsdp_inference=num_gpus > 1 and not replicated_dit,
                parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=bool(runtime.get("offload_text_encoder", False)),
                    vae=bool(runtime.get("offload_vae", False)),
                    pin_cpu_memory=bool(runtime.get("pin_cpu_memory", False)),
                ),
                compile=CompileConfig(
                    enabled=False,
                    mode=None,
                    vae_enabled=bool(cfg.get("compile_vae", True)),
                ),
            ),
        )

    # ---------------------------------------------------------- gpu plumbing

    def _worker_loop(self) -> None:
        """Run submitted jobs, one at a time, forever.

        The waiter is always released, even when the job died: a completion
        event that never arrives is indistinguishable from a hang, and this
        thread is the only one that will ever set it.
        """
        logger.info("generation worker ready")
        while True:
            job = self._jobs.get()
            try:
                job.fn()
            except BaseException as error:  # noqa: BLE001 — handed to the waiter
                job.error = error
                logger.exception("generation worker job raised")
            finally:
                job.done.set()

    def _submit(self, fn) -> _Job:
        """Queue work on the generation worker and hand back its handle."""
        job = _Job(fn)
        self._jobs.put(job)
        return job

    def _run_on_worker(self, fn) -> None:
        """Run work on the worker, block until it finishes, and re-raise its failure.

        Used only by ``load()``. Blocking here is the point: the runtime marks
        the pod ready when ``load()`` returns, so a failed warm-up has to stop
        startup rather than being discovered by the first client.
        """
        job = self._submit(fn)
        while not job.done.wait(timeout=WORKER_POLL_SECONDS):
            pass
        if job.error is not None:
            raise job.error

    # ------------------------------------------------------------- warm-up

    def _warmup(self) -> None:
        """Build one throwaway clip per shape, before the pod reports ready.

        Every distinct frame count and canvas is a separate one-time cost —
        regional compile, sparse-kernel autotune, allocator growth — and paying
        it here means the first real clip streams at warm speed. Results are
        discarded: `return_frames=False, save_video=False` skips the whole
        post-decode path, so a warm-up costs generation time and nothing else.
        """
        aspects = self.inference_cfg.get("warmup_aspects") or [self.default_aspect]
        shapes = list(dict.fromkeys([*self.ramp_frames, self.default_clip_frames]))
        cold = [a for a in clip_plan.ASPECT_CHOICES if a not in aspects]
        if cold:
            logger.info(
                "aspects left cold; their first clip pays a one-off compile stall", aspects=cold
            )
        for aspect in aspects:
            height, width = clip_plan.canvas_for_choice(str(aspect))
            for frames in shapes:
                started = time.monotonic()
                self.generator.generate(
                    self._request(
                        frames=frames,
                        prompt=WARMUP_PROMPT,
                        seed=self.default_seed,
                        height=height,
                        width=width,
                        keep_output=False,
                    )
                )
                logger.info(
                    "warmed clip shape",
                    aspect=aspect,
                    frames=frames,
                    height=height,
                    width=width,
                    seconds=round(time.monotonic() - started, 2),
                )

    # -------------------------------------------------------- session state

    def _reset_session_state(self) -> None:
        """Return every session-scoped field to its default.

        The replacement for a runtime-built ``InputState``: the same fields with
        the same defaults, as plain attributes. Called once at ``load()`` and at
        every ``@session_started``, which is what keeps one session from ever
        observing another's conditions.
        """
        # Client-settable conditions. The prompt starts empty on purpose: the
        # channel is whatever the client asks for, so `start` waits for one
        # rather than falling back to a stock scene nobody chose.
        self._prompt: str = ""
        self._clip_frames: int = self.default_clip_frames
        self._seed: int = self.default_seed
        self._aspect: str = self.default_aspect

        # Channel lifecycle. `_started` arms the run loop; `_running` is true
        # while a channel is live; `_do_reset` asks it to wind down; `_stop_only`
        # marks that wind-down as a `stop` (conditions kept) rather than a reset.
        self._started: bool = False
        self._running: bool = False
        self._do_reset: bool = False
        self._stop_only: bool = False
        self._paused: bool = False

        # Progress, mirrored so a `state_update` is a complete snapshot.
        self._clip_index: int = -1
        self._clips_sent: int = 0
        self._frames_sent: int = 0
        self._seconds_sent: float = 0.0
        self._clip_start_seconds: float = 0.0
        self._current_clip_seconds: float = 0.0

    def _canvas(self) -> tuple[int, int]:
        """The `(height, width)` this session generates at."""
        return clip_plan.canvas_for_choice(self._aspect)

    def _frames_for_clip(self, index: int) -> int:
        """Frame count for clip ``index``, ramp included."""
        return clip_plan.clip_frames(index, self._clip_frames, self.ramp_frames)

    def _prompt_effective(self) -> tuple[int, float]:
        """Which clip the current prompt lands on, and how far away that is.

        The channel always has one clip in flight, so a prompt set now applies to
        the next clip *submitted* — never the one already being built.
        """
        if not self._running:
            return 0, 0.0
        if self._clip_index < 0:
            # Clip 0 is being built with the prompt as it stood at `start`; a
            # prompt set now is the first one that can land, on clip 1.
            return 1, round(clip_plan.seconds_for_frames(self._frames_for_clip(0)), 2)
        played = self._seconds_sent - self._clip_start_seconds
        remaining = max(0.0, self._current_clip_seconds - played)
        queued = clip_plan.seconds_for_frames(self._frames_for_clip(self._clip_index + 1))
        return self._clip_index + 2, round(remaining + queued, 2)

    def _snapshot(self) -> StateUpdate:
        """Everything a client can observe, in one message.

        The single source of the snapshot: `state_update` broadcasts it, a
        joining client is greeted with it, and `get_state` answers with it. Built
        once here so those three can never disagree.
        """
        height, width = self._canvas()
        effective_index, effective_seconds = self._prompt_effective()
        return StateUpdate(
            prompt=self._prompt or None,
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            clip_seconds_min=clip_plan.MIN_SECONDS_PUBLISHED,
            clip_seconds_max=clip_plan.MAX_SECONDS_PUBLISHED,
            seed=self._seed,
            aspect=self._aspect,
            width=width,
            height=height,
            ready=bool(self._prompt),
            running=self._running,
            paused=self._paused,
            clip_index=self._clip_index,
            clips_sent=self._clips_sent,
            seconds_sent=round(self._seconds_sent, 2),
            prompt_effective_clip_index=effective_index,
            prompt_effective_in_seconds=effective_seconds,
            valid_commands=session_rules.valid_commands(
                running=self._running, paused=self._paused, ready=bool(self._prompt)
            ),
        )

    async def _send_state_update(self) -> None:
        """Broadcast the snapshot to every connected client."""
        await self.send(self._snapshot())

    async def _refuse(self, command: str, reason: str) -> None:
        """Reject a command: tell every client, and leave its reply bodyless.

        A handler returns only the message its annotation names, and reports
        a failure by broadcasting `command_error` and returning without a
        value. The runtime answers that with a correlated bodyless
        acknowledgement, so an awaiting client resolves rather than hanging —
        and unlike a raised runtime ``CommandError``, whose failure frame is
        withheld from v0 clients, the broadcast reaches every SDK generation.

        Logged as well, so refusals are visible server-side and not only in the
        client's message.
        """
        logger.info("command refused", command=command, reason=reason)
        await self.send(CommandError(command=command, reason=reason))

    # ------------------------------------------------------------ lifecycle

    @session_started
    async def on_session_started(self) -> None:
        """Clear every condition so a new session never inherits an old one."""
        self._reset_session_state()

    @session_ended
    async def on_session_ended(self) -> None:
        """Wind the channel down; the only hook guaranteed to fire on every path."""
        self._started = False
        self._do_reset = True

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        """Greet the joining client with the full state, so it can render at once.

        Addressed rather than broadcast: the clients already watching have this
        state, and a late joiner needs it without replaying every command.
        """
        await client.send(self._snapshot())

    # ------------------------------------------------------------- commands

    @event(
        name="set_prompt",
        description=(
            "Set what the channel shows. Required before `start`; an empty text "
            "clears it again. Valid at any time: while idle it applies to the "
            "next `start`, and while streaming it applies to a later clip, "
            "because the model always builds one clip ahead. Emits "
            "`prompt_accepted` and `state_update`, both of which report the clip "
            "this prompt lands on and how much already-built video plays first."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=MAX_PROMPT_CHARS,
            description=(
                "What the channel should show, up to 800 characters. Read when "
                "the model starts building a clip, so a change reaches the "
                "viewer a clip or two later rather than immediately. Blank text "
                "clears it, and `start` is then rejected until one is set again."
            ),
        ),
    ) -> PromptAccepted:
        """Set the prompt used from the next clip the model starts."""
        self._prompt = prompt.strip()
        effective_index, effective_seconds = self._prompt_effective()
        await self._send_state_update()
        return PromptAccepted(
            prompt=self._prompt or None,
            effective_clip_index=effective_index,
            effective_in_seconds=effective_seconds,
        )

    @event(
        name="set_clip_seconds",
        description=(
            "Set how long each clip is. The value is snapped to the nearest "
            "length the model can produce, so read the effective one back from "
            "`clip_length_accepted`. Valid at any time; while streaming it "
            "applies from the next clip the model starts. Longer clips cut less "
            "often, shorter clips let a new prompt reach the viewer sooner. "
            "Emits `clip_length_accepted` and `state_update`."
        ),
    )
    async def set_clip_seconds(
        self,
        seconds: float = InputField(
            default=clip_plan.MAX_SECONDS_PUBLISHED,
            ge=clip_plan.MIN_SECONDS_PUBLISHED,
            le=clip_plan.MAX_SECONDS_PUBLISHED,
            description=(
                f"Clip length in seconds, between {_CLIP_RANGE}. Snapped to the "
                "nearest length the model can produce, so the value that takes "
                "effect can differ slightly; `state_update.clip_seconds` always "
                "carries the one in force."
            ),
        ),
    ) -> ClipLengthAccepted:
        """Set the steady-state clip length."""
        self._clip_frames = clip_plan.frames_for_seconds(float(seconds))
        await self._send_state_update()
        return ClipLengthAccepted(
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            frames=self._clip_frames,
        )

    @event(
        name="set_seed",
        description=(
            "Set the seed the channel starts from; each clip advances it by "
            "one, so a channel is reproducible end to end while its clips still "
            "differ. Valid at any time; applies from the next clip the model "
            "starts. Emits `seed_accepted` and `state_update`."
        ),
    )
    async def set_seed(
        self,
        seed: int = InputField(
            default=1000,
            ge=0,
            description=(
                "Seed for the next channel. The clip at index `n` uses this "
                "value plus `n`. Reproduction is close rather than exact: the "
                "deployment runs fused kernels that can reorder arithmetic."
            ),
        ),
    ) -> SeedAccepted:
        """Set the seed the next channel run starts from."""
        self._seed = int(seed)
        await self._send_state_update()
        return SeedAccepted(seed=self._seed)

    @event(
        name="set_canvas",
        description=(
            "Choose the aspect ratio of `main_video`. The video track keeps one "
            "size for the whole channel, so this is only valid while nothing is "
            "streaming — send it before `start` or after `stop`. Emits "
            "`canvas_accepted`, carrying the exact pixel size, and "
            "`state_update`, or `command_error` if a channel is streaming or the "
            "ratio is not one this model offers."
        ),
    )
    async def set_canvas(
        self,
        aspect: str = InputField(
            default="16:9",
            choices=list(clip_plan.ASPECT_CHOICES),
            description=(
                "Aspect ratio of `main_video`, fixed for the whole channel. "
                "`canvas_accepted` and `state_update` report the width and "
                "height in pixels it resolves to."
            ),
        ),
    ) -> CanvasAccepted:
        """Set the session's canvas; refused once a channel is live."""
        if self._running:
            await self._refuse(
                "set_canvas",
                "The video size is fixed for the life of a channel; send `stop` first.",
            )
            return None
        try:
            height, width = clip_plan.canvas_for_choice(aspect)
        except ValueError as error:
            await self._refuse("set_canvas", str(error))
            return None
        self._aspect = aspect
        await self._send_state_update()
        return CanvasAccepted(aspect=aspect, width=width, height=height)

    @event(
        name="start",
        description=(
            "Start the channel. Valid only while nothing is streaming, and only "
            "once a prompt is set; everything else has a default. Emits "
            "`channel_started` and `state_update` at once, then a `clip_started` "
            "as each clip reaches the output tracks. The first clip has to be "
            "built before anything can stream, so expect several seconds of no "
            "video, or `command_error` if a channel is already streaming or no "
            "prompt is set."
        ),
    )
    async def start(self) -> None:
        """Arm the channel loop; it begins on its next poll."""
        if self._running:
            await self._refuse(
                "start", "A channel is already streaming; send `stop` to end it first."
            )
            return
        if not self._prompt:
            await self._refuse("start", "No prompt is set; send `set_prompt` first.")
            return
        self._started = True
        self._do_reset = False
        await self._send_state_update()

    @event(
        name="pause",
        description=(
            "Freeze the output stream on its current frame. Valid only while a "
            "channel is streaming and not already paused. The model keeps "
            "building ahead, so `resume` continues instantly. Emits "
            "`channel_paused` and `state_update`, or `command_error` if no "
            "channel is streaming or it is already paused."
        ),
    )
    async def pause(self) -> ChannelPaused:
        """Hold the stream; the channel itself is unaffected."""
        if not self._running:
            await self._refuse("pause", "No channel is streaming.")
            return None
        if self._paused:
            await self._refuse("pause", "The stream is already paused.")
            return None
        self._paused = True
        await self._send_state_update()
        return ChannelPaused(seconds_sent=round(self._seconds_sent, 2))

    @event(
        name="resume",
        description=(
            "Continue a paused stream exactly where it froze, with no warm-up "
            "and without skipping ahead to catch up. Valid only while paused. "
            "Emits `channel_resumed` and `state_update`, or `command_error` if "
            "the stream is not paused."
        ),
    )
    async def resume(self) -> ChannelResumed:
        """Release a paused stream."""
        if not self._paused:
            await self._refuse("resume", "The stream is not paused.")
            return None
        self._paused = False
        await self._send_state_update()
        return ChannelResumed(seconds_sent=round(self._seconds_sent, 2))

    @event(
        name="stop",
        description=(
            "End the channel, keeping every condition, so `start` begins a "
            "fresh one with the same setup. Valid only while a channel is "
            "streaming. The stream stops within about a second, but the clip "
            "already being built cannot be cancelled, so `state_update.running` "
            "can stay true for a few seconds longer. Emits `channel_stopped` "
            "and `state_update`, or `command_error` if no channel is streaming."
        ),
    )
    async def stop(self) -> None:
        """Wind the channel down, keeping the conditions."""
        if not self._running:
            await self._refuse("stop", "No channel is streaming.")
            return
        self._started = False
        self._stop_only = True
        self._do_reset = True
        self._paused = False
        await self._send_state_update()

    @event(
        name="reset",
        description=(
            "Return every condition to its default and clear whatever is queued "
            "on the output tracks. Stops the channel first if one is streaming. "
            "Valid at any time. Emits `channel_reset` and `state_update`."
        ),
    )
    async def reset(self) -> ChannelReset:
        """Clear the session back to its defaults."""
        was_running = self._running
        self._started = False
        self._stop_only = False
        # Only ask a live channel to wind down; see `_wait_until_armed`.
        self._do_reset = was_running
        self._prompt = ""
        self._clip_frames = self.default_clip_frames
        self._seed = self.default_seed
        self._aspect = self.default_aspect
        self._paused = False
        self.output.flush()
        await self._send_state_update()
        return ChannelReset(was_running=was_running)

    @event(
        name="get_state",
        description=(
            "Return a snapshot of everything the session exposes: the "
            "conditions in force, the lifecycle flags, progress through the "
            "channel, and the commands that are valid right now. The same "
            "payload the model broadcasts as `state_update`, so a client can "
            "render its whole interface from this one message. Valid at any "
            "time."
        ),
    )
    async def get_state(self) -> StateUpdate:
        """Answer with the same snapshot `state_update` broadcasts."""
        return self._snapshot()

    # ------------------------------------------------------------- run loop

    async def run(self) -> None:
        """The model's control loop: wait for an audience, arm, stream, repeat.

        Nothing here may raise: an exception out of ``run()`` is an
        unrecoverable crash of the whole model loop, not the end of one session,
        so ``_run_channel`` owns its own failure reporting.
        """
        while True:
            await self.connected.wait()
            if await self._wait_until_armed():
                await self._run_channel()

    async def _wait_until_armed(self) -> bool:
        """Wait for `start`; False if the audience left first."""
        while True:
            # Deliberately not gated on `_do_reset`: `reset` while idle sets
            # it and nothing would ever clear it, which would wedge this loop
            # forever. `_run_channel` clears it as it starts, and `stop`/`reset`
            # drop `_started`, so a stale flag cannot start a doomed channel.
            if self._started and self._prompt:
                return True
            if not self.connected.is_set():
                return False
            await asyncio.sleep(POLL_SECONDS)

    def _should_abort(self) -> bool:
        """Whether the channel must wind down now.

        `_do_reset` is `stop`/`reset`/session end; a lost audience is the other
        reason. Every abort check — the worker, the emitter — reads this rather
        than the flag alone.
        """
        return self._do_reset or not self.connected.is_set()

    async def _run_channel(self) -> None:
        """Stream clips back to back, always building one ahead."""
        height, width = self._canvas()
        self._running = True
        self._do_reset = False
        self._stop_only = False
        self._paused = False
        self._clip_index = -1
        self._clips_sent = 0
        self._frames_sent = 0
        self._seconds_sent = 0.0
        self._clip_start_seconds = 0.0
        self._current_clip_seconds = 0.0
        channel_started_at = time.monotonic()

        # Depth 1 is the lookahead: the worker can be at most one finished clip
        # ahead of the transport, which is exactly the backpressure that keeps
        # a prompt change from being buried behind a deep queue.
        results: queue.Queue = queue.Queue(maxsize=1)
        pending: _Job | None = None
        # Emission pacing carried across clips, so a clip seam costs no time.
        pacer = {"clock_start": None, "frames_paced": 0}

        def submit(index: int) -> _Job:
            """Queue clip ``index``, capturing the conditions as they stand now."""
            frames = self._frames_for_clip(index)
            prompt = self._prompt
            seed = self._seed + index

            def job() -> None:
                try:
                    if self._should_abort():
                        raise _ChannelStopped
                    built = self._generate_clip(index, frames, prompt, seed, height, width)
                    if self._should_abort():
                        raise _ChannelStopped
                    results.put(("clip", index, prompt, built))
                except _ChannelStopped:
                    results.put(("stopped", index, prompt, None))
                except BaseException as error:  # noqa: BLE001 — reported to the client
                    logger.exception("clip generation failed", clip=index)
                    results.put(("error", index, prompt, error))

            return self._submit(job)

        try:
            await self.send(
                ChannelStarted(
                    width=width,
                    height=height,
                    clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
                    first_clip_seconds=round(
                        clip_plan.seconds_for_frames(self._frames_for_clip(0)), 3
                    ),
                )
            )
            await self._send_state_update()

            pending = submit(0)
            while True:
                kind, index, prompt, payload = await asyncio.to_thread(results.get)
                if kind == "error":
                    raise payload
                if kind == "stopped":
                    break

                frames_list, samples = payload
                if index == 0:
                    logger.info(
                        "first clip ready",
                        ttff_s=round(time.monotonic() - channel_started_at, 2),
                    )
                # Submit the next clip BEFORE emitting this one, so it is built
                # while this one plays. This is what makes the channel endless.
                pending = submit(index + 1)

                self._clip_index = index
                self._current_clip_seconds = clip_plan.seconds_for_frames(len(frames_list))
                self._clip_start_seconds = self._seconds_sent
                await self.send(
                    ClipStarted(
                        clip_index=index,
                        clip_seconds=round(self._current_clip_seconds, 3),
                        prompt=prompt,
                    )
                )
                await self._send_state_update()

                await self._emit_paced(frames_list, samples, pacer)
                if self._should_abort():
                    break

                self._clips_sent = index + 1
                await self.send(
                    ClipComplete(clip_index=index, seconds_sent=round(self._seconds_sent, 2))
                )
                await self._send_state_update()

            if self._stop_only:
                await self.send(
                    ChannelStopped(
                        seconds_sent=round(self._seconds_sent, 2), clips_sent=self._clips_sent
                    )
                )
                logger.info(
                    "channel stopped", clips=self._clips_sent, seconds=round(self._seconds_sent, 2)
                )
        except (Exception, SystemExit) as error:
            logger.exception("channel failed")
            # Disarm before reporting. `run()` re-enters the arm loop the moment
            # this returns, and `_started` is what it gates on — leaving it set
            # would restart the channel straight into the same failure, over and
            # over, at eight GPUs a go. The client is told to `start` again.
            self._started = False
            try:
                await self.send(
                    ChannelFailed(reason=str(error), seconds_sent=round(self._seconds_sent, 2))
                )
            except Exception:  # noqa: BLE001 — reporting must not crash run()
                logger.exception("failed to report the channel failure")
        finally:
            self._running = False
            self._paused = False
            self._stop_only = False
            self._drain_until_finished(results, pending)
            self._do_reset = False
            try:
                await self._send_state_update()
            except Exception:  # noqa: BLE001 — teardown must not crash run()
                logger.exception("failed to send the closing state update")

    def _drain_until_finished(self, results: queue.Queue, pending: _Job | None) -> None:
        """Ask the in-flight clip to stop and drain its handoff until it returns.

        A clip cannot be cancelled once it is being built, so this waits it out —
        blocking, not awaiting, because it also runs under cancellation. Draining
        while it winds down is what keeps a blocked ``put`` from deadlocking, and
        waiting is not optional: a job left blocked on a full queue would stall
        every later channel behind it.
        """
        if pending is None or pending.done.is_set():
            return
        self._do_reset = True
        deadline = 300
        while not pending.done.is_set() and deadline > 0:
            try:
                while True:
                    results.get_nowait()
            except queue.Empty:
                pass
            if not self._worker.is_alive():
                logger.error("generation worker is gone; no further channel can be served")
                return
            pending.done.wait(timeout=1)
            deadline -= 1
        if not pending.done.is_set():
            logger.error("clip did not wind down within 300s; later channels will queue behind it")

    # -------------------------------------------------------------- emitter

    async def _emit_paced(self, frames_list, samples, pacer: dict) -> None:
        """Emit one clip as paced slices on a 24 fps metronome.

        - Paced by FRAMES, not slices: a clip's tail slice is short, and charging
          it a full slot would open a hole in the cadence.
        - The clock carries across clips through ``pacer``, so a seam costs no
          time as long as the next clip was ready.
        - Never burst to catch up: if a clip landed late, re-anchor instead. A
          catch-up burst only overflows the transport queue. `seam late` is the
          log line that says the channel is not keeping up.
        - Emits omit ``compute_time``, so every slice is tagged at the pinned
          24 fps — the rate the audio is already sample-clocked against.
        - `pause` holds by awaiting a sleep; handlers dispatch on their own
          coroutine, so this starves nothing.
        """
        import numpy as np

        samples_per_frame = OUTPUT_SAMPLE_RATE / FRAME_RATE
        total = len(frames_list)
        for lo in range(0, total, EMIT_FRAMES):
            while self._paused and not self._should_abort():
                await asyncio.sleep(POLL_SECONDS)
            if self._should_abort():
                return
            hi = min(lo + EMIT_FRAMES, total)
            alo = round(lo * samples_per_frame)
            ahi = round(hi * samples_per_frame)

            now = asyncio.get_running_loop().time()
            if pacer["clock_start"] is None:
                pacer["clock_start"] = now
            content_pos = pacer["frames_paced"] / FRAME_RATE
            late = now - (pacer["clock_start"] + content_pos)
            if lo == 0 and late > 0.05:
                logger.info("seam late", clip=self._clip_index, late_s=round(late, 2))
            pacer["clock_start"] = max(pacer["clock_start"], now - content_pos)
            delay = pacer["clock_start"] + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            pacer["frames_paced"] += hi - lo
            self._frames_sent += hi - lo
            self._seconds_sent = self._frames_sent / FRAME_RATE
            video = np.ascontiguousarray(np.stack(frames_list[lo:hi]))
            await self.emit(
                FastH3Output(main_video=video, main_audio=samples[:, alo:ahi])
            )

    # ------------------------------------------------------------ generation

    def _request(
        self,
        *,
        frames: int,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        keep_output: bool,
    ):
        """Build one generation request.

        Mirrors ``basic_fasth3.py:build_request``. ``keep_output=False`` is the
        warm-up shape: it skips the whole post-decode path, so a warm-up costs
        generation time and nothing else.
        """
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        return GenerationRequest(
            prompt=prompt,
            # MiniMax-H3 is guidance-distilled, so there is no negative branch
            # to steer and no CFG pass to pay for.
            negative_prompt="",
            sampling=SamplingConfig(
                height=height,
                width=width,
                num_frames=frames,
                fps=FRAME_RATE,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=1.0,
                batch_cfg=False,
                seed=seed,
            ),
            output=OutputConfig(save_video=False, return_frames=keep_output),
        )

    def _generate_clip(
        self, index: int, frames: int, prompt: str, seed: int, height: int, width: int
    ):
        """Build one clip and convert it to what the output tracks want.

        Returns ``(frames_list, samples)``: a list of RGB uint8 ``[h, w, 3]``
        arrays and int16 ``[1, samples]`` at 48 kHz, trimmed to exactly
        ``len(frames_list) / 24`` seconds so the two tracks stay in lockstep.
        """
        started = time.monotonic()
        result = self.generator.generate(
            self._request(
                frames=frames,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                keep_output=True,
            )
        )
        built = time.monotonic() - started

        frames_list = result.frames
        if not frames_list:
            raise RuntimeError("the generator returned no frames")
        samples = self._to_wire_audio(result.audio, result.audio_sample_rate, len(frames_list))
        logger.info(
            "clip built",
            clip=index,
            frames=len(frames_list),
            content_s=round(len(frames_list) / FRAME_RATE, 2),
            build_s=round(built, 2),
            stages=self._stage_times(result),
        )
        return frames_list, samples

    @staticmethod
    def _stage_times(result) -> dict:
        """Per-stage seconds from the generator, for the clip log line.

        This is where a regression shows up first: post-decode frame processing
        scales with resolution x frames and competes with the channel's slack.
        """
        try:
            stages = getattr(getattr(result, "logging_info", None), "stages", None)
            if not stages:
                return {}
            return {
                name: round(float(metrics["execution_time"]), 3)
                for name, metrics in stages.items()
                if metrics.get("execution_time") is not None
            }
        except Exception:  # noqa: BLE001 — a log line must never fail a clip
            logger.exception("could not read the generator stage timings")
            return {}

    def _to_wire_audio(self, audio, sample_rate, frames: int):
        """Resample, downmix and quantize one clip's waveform for the wire.

        Mono at the source is deliberate: the transport mean-downmixes before
        the wire anyway, and the runtime recorder flattens two channels by
        concatenation, so a stereo emit only corrupts recordings. Averaging here,
        in float and before the int16 scale, is the same downmix one step
        earlier.
        """
        import torch
        import torchaudio.functional as AF

        if audio is None:
            raise RuntimeError("the generator returned no audio")
        waveform = audio if torch.is_tensor(audio) else torch.as_tensor(audio)
        waveform = waveform.detach().float().cpu()
        # The decoder hands back [samples, channels]; the wire wants channel-major.
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)
        waveform = waveform.contiguous()

        rate = int(sample_rate or NATIVE_SAMPLE_RATE)
        if rate != OUTPUT_SAMPLE_RATE:
            waveform = AF.resample(waveform, rate, OUTPUT_SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        want = round(frames / FRAME_RATE * OUTPUT_SAMPLE_RATE)
        if waveform.shape[-1] > want:
            waveform = waveform[:, :want]
        elif waveform.shape[-1] < want:
            pad = torch.zeros(
                (waveform.shape[0], want - waveform.shape[-1]), dtype=waveform.dtype
            )
            waveform = torch.cat([waveform, pad], dim=-1)
        return (waveform.clamp(-1, 1) * 32767).to(torch.int16).numpy()
