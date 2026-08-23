# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""RLDX-1 ported to the Reactor Runtime (ReactorPipeline pattern).

Wraps the upstream ``RLDXPolicy`` (RLWRLD/RLDX-1) behind the runtime. The client
(cpp_sdk) publishes the camera views as input tracks and sends the robot proprio
state and task; this pipeline assembles the observation exactly as
``RLDXSimPolicyWrapper.get_action`` expects, runs the policy, and streams the
predicted action chunk back as an :class:`ActionPrediction` message. No video is
emitted — this is a video-in -> action-out model.

The input spec (views, window, state/action dims) comes from the checkpoint's
modality config — RLWRLD's source of truth — read at load and announced to the
client in a ``model_schema`` message at session start, so a new
checkpoint reconfigures both sides without code changes here.

Robot state arrives as **video frame metadata** — every view's frame is tagged
with the proprio JSON — with the ``state_json`` field as the fallback for clients
that cannot tag frames. ``robot_state.py`` owns that seam; see
:meth:`RLDXPipeline._resolve_state`.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from reactor_runtime import (
    Idle,
    ReactorPipeline,
    ReadMode,
    connected,
    event,
    session_started,
)

from model_schema import build_schema
from model_types import (
    ActionPrediction,
    CommandError,
    ModelSchema,
    RLDXInput,
    RLDXState,
)
from robot_state import STATE_TAG_KEYS, FrameStateTags, parse_state, zero_state

VIEWS = ("left_view", "right_view", "wrist_view")

# Fallback temporal window length, used only if the checkpoint's modality config
# is somehow unavailable at load time. The authoritative window comes from
# ``modality_configs["video"].delta_indices`` (see ``RLDXPipeline.load``); this
# constant just sizes a last-resort consecutive-frame window.
TV = 4

# Frames held per view between commits, waiting to be aligned across views by
# capture stamp (see ``_align_by_capture``). The alignment never reaches further
# back than the laggiest view, so this only has to cover a control step's worth
# of publishing on the fastest one — not the whole temporal window.
_RECENT_FRAMES = 8

# Default policy for missing / unparseable robot state (see ``_resolve_state``).
_DEFAULT_STATE_FALLBACK = "hold_last"

# Where this pipeline reads robot state from, announced in the handshake so a
# client configures its carrier from the schema instead of from a release note.
# Preference order, not exclusivity: ``set_state_json`` still works (see
# ``_resolve_state``), but a client that can tag frames should.
_STATE_SOURCE = "frame_metadata"

# state vector key -> dimension (RoboCasa GENERAL_EMBODIMENT)
_STATE_DIMS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}


@dataclass(frozen=True)
class _FrameCandidate:
    """One ready frame and the state metadata that arrived with it."""

    capture_time_us: int | None
    data: Any
    metadata: bytes | None = None


@dataclass(frozen=True)
class _AlignedCommit:
    """Frames and state candidates selected for one control-step commit."""

    frames: dict[str, Any]
    state_candidates: tuple[_FrameCandidate, ...]
    view_skew_us: int | None


def read_config(config_path: Path | None) -> dict[str, Any]:
    """Parse the ``runtime.config`` file the runtime hands over as a path.

    The runtime passes ``load()`` the *path* to the file ``reactor.yaml`` names
    under ``runtime.config`` and never reads its contents, so config.yml is
    parsed here; the resulting dict is exactly what the 2.x runtime used to hand
    ``load()`` pre-parsed.
    """
    if config_path is None:
        return {}
    return yaml.safe_load(Path(config_path).read_text()) or {}


def _window_indices(deltas: list[int], frames_per_step: int, hist_len: int) -> list[int]:
    """Map action-step delta offsets to indices into a per-view frame history.

    ``deltas`` are the checkpoint's chronological video ``delta_indices`` — each a
    non-positive action-step offset with the most-recent frame anchored at 0
    (e.g. ``[-6, -4, -2, 0]``). ``frames_per_step`` is how many buffered frames
    make up one action-step (the client's publish cadence). ``hist_len`` is the
    number of frames currently buffered for a view (oldest first, newest last).

    Returns one index into that history per delta, most-recent last. Indices are
    clamped to ``[0, hist_len - 1]`` so that during warm-up (fewer frames than the
    window spans) the oldest available frame is repeated — preserving the
    historical left-pad behaviour rather than failing.
    """
    last = hist_len - 1
    idxs: list[int] = []
    for d in deltas:
        i = last + d * frames_per_step
        if i < 0:
            i = 0
        elif i > last:
            i = last
        idxs.append(i)
    return idxs


def _align_by_capture(
    candidates: Mapping[str, Sequence[_FrameCandidate]],
) -> _AlignedCommit:
    """Choose one frame per view near one instant, with its matching state.

    ``candidates`` holds each view's recent frames, oldest first. The returned
    state candidates are taken only from the selected frames, never from newer
    frames that arrived but were held back by alignment.

    The reference instant is the *oldest* of the views' newest stamps — the most
    recent moment every view has actually covered. A view running ahead is held
    back to it, because a window that pairs a fresh frame from one camera with a
    stale one from another shows the policy a scene that never existed; being one
    frame behind on every view does not.

    A view whose newest frame carries no stamp drops the whole commit back to
    newest-per-view with ``skew_us`` ``None``: with nothing to compare against,
    aligning the stamped views around it would be a guess.
    """
    newest = {v: entries[-1] for v, entries in candidates.items()}
    if any(candidate.capture_time_us is None for candidate in newest.values()):
        return _AlignedCommit(
            frames={v: candidate.data for v, candidate in newest.items()},
            # With no common transport clock, preserve the legacy rule among
            # only the frames actually committed: embedded stamp, then wire
            # stamp, then deterministic view/arrival order.
            state_candidates=tuple(newest.values()),
            view_skew_us=None,
        )

    def capture_stamp(candidate: _FrameCandidate) -> int:
        assert candidate.capture_time_us is not None
        return candidate.capture_time_us

    ref = min(capture_stamp(candidate) for candidate in newest.values())
    chosen: dict[str, _FrameCandidate] = {}
    for view, entries in candidates.items():
        stamped = [entry for entry in entries if entry.capture_time_us is not None]
        chosen[view] = min(
            stamped,
            key=lambda entry: abs(capture_stamp(entry) - ref),
        )

    # The selected frame nearest the common reference owns this commit's state.
    # Normally all three selected frames carry byte-identical metadata; this is
    # the explicit deterministic rule for a client whose view tags disagree.
    state_source = min(
        chosen.values(),
        key=lambda candidate: abs(capture_stamp(candidate) - ref),
    )
    stamps = [capture_stamp(candidate) for candidate in chosen.values()]
    return _AlignedCommit(
        frames={view: candidate.data for view, candidate in chosen.items()},
        state_candidates=(state_source,),
        view_skew_us=max(stamps) - min(stamps),
    )


class RLDXPipeline(ReactorPipeline):
    input: RLDXInput
    state: RLDXState
    buffer_size = 8

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path)

        # Heavy imports deferred so `reactor schema` can run on CPU.
        from reactor_runtime import get_weights_path

        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

        self._H = int(config.get("height", 256))
        self._W = int(config.get("width", 256))

        # Resolve the checkpoint under the release weights root
        # (REACTOR_WEIGHTS_PATH in production, the local cache in dev). Empty ->
        # the root itself; a relative name resolves under it; an absolute path
        # bypasses it. Weights load offline — nothing is fetched at run time.
        root = str(get_weights_path())
        ckpt = config.get("checkpoint_dir", "")
        model_path = ckpt if os.path.isabs(ckpt) else os.path.join(root, ckpt)

        embodiment = getattr(
            EmbodimentTag, config.get("embodiment_tag", "GENERAL_EMBODIMENT")
        )
        device = f"cuda:{config.get('device_id', 0)}"

        policy = RLDXPolicy(
            embodiment_tag=embodiment,
            model_path=model_path,
            device=device,
            strict=False,
        )
        # The sim wrapper takes the flat video.*/state.*/annotation.* obs format
        # and returns flat action.* keys — exactly the quickstart recipe.
        self._policy = RLDXSimPolicyWrapper(policy, strict=False)

        # Temporal window contract: the checkpoint's video ``delta_indices`` are
        # chronological action-step offsets with the most-recent frame at 0 —
        # e.g. [-6, -4, -2, 0] for a video_length=4 / video_stride=2 checkpoint,
        # or [0] when the memory/video window is disabled. Reading them here
        # (instead of hardcoding a consecutive TV-frame window) makes the buffer
        # sample frames at the stride the model was trained on, and keeps a
        # single-frame checkpoint working too.
        try:
            video_cfg = self._policy.get_modality_config()["video"]
            self._video_deltas = list(video_cfg.delta_indices)
            self._views = tuple(video_cfg.modality_keys)
        except Exception:
            self._video_deltas = list(range(-(TV - 1), 1))  # [-3,-2,-1,0]
            self._views = VIEWS

        # Observation cadence — matches RLDX-1's validated eval loop (RLWRLD
        # MultiStepWrapper): buffer one frame per control step, and re-plan once
        # per executed chunk. Two independent clocks, both derived from the
        # RoboCasa control rate (control_freq=20):
        #   * commit the freshest frame every 1/control_hz s  -> the per-control-
        #     step window buffer the model was trained/eval'd on. delta_indices
        #     are in control steps, so this makes the strided window come out at
        #     the right real-time spacing regardless of the client's publish fps.
        #   * re-plan (get_action) only once per exec_horizon control steps: the
        #     model is memory-stateful (memory_stride == execution_horizon) and
        #     assumes each chunk is executed before the next re-plan; free-running
        #     advances memory faster than reality and degrades rollouts.
        self._control_hz = float(config.get("control_hz", 20))
        self._pace = bool(config.get("pace_inference", True))

        # Execution horizon: control steps the client executes per chunk before
        # the server re-plans. Derive it from the checkpoint (its
        # ``action_horizon``) the same way the window reads ``delta_indices`` —
        # so swapping to a checkpoint with a different horizon keeps the pacing
        # in lockstep instead of silently drifting while the strided window
        # auto-adjusts. The config value is only a fallback when the model config
        # can't be read.
        try:
            self._exec_horizon = int(self._policy.policy.model.action_horizon)
        except Exception:
            self._exec_horizon = int(config.get("exec_horizon", 16))

        # Policy for missing / unparseable robot state:
        #   "hold_last" (default) - reuse the last valid state; skip inference
        #                           until the first valid state arrives.
        #   "zero"                - zero-fill (legacy behaviour) but signal it.
        #   "error"               - skip inference and signal on every bad frame.
        self._state_fallback = str(
            config.get("state_fallback", _DEFAULT_STATE_FALLBACK)
        ).lower()
        self._last_state: dict[str, np.ndarray] | None = None
        self._state_degraded = False
        self._schema_pending = False
        # Proprio tag selected from the latest aligned frame commit
        # (robot_state.py owns parsing and fallback ordering).
        self._frame_tags = FrameStateTags()
        # Cross-view capture spread of the last committed frames, echoed with
        # the chunk those frames fed.
        self._last_skew_us: int | None = None

        # State/action dims from the checkpoint's normalization params — the
        # same source the policy's own wire-boundary validator checks against.
        try:
            validator = self._policy.policy.validator
            self._state_dims = {
                k: int(d) for k, d in validator.expected_state_dims.items()
            }
            action_dims = {
                k: int(d) for k, d in validator.expected_action_dims.items()
            }
        except Exception:
            self._state_dims = dict(_STATE_DIMS)
            action_dims = {}
        try:
            rtc_mode = str(self._policy.policy.rtc_inference_mode)
        except Exception:
            rtc_mode = "unknown"

        # Session-start handshake payload: the checkpoint-derived
        # values above, exactly as this process serves them. Raises at load if
        # the checkpoint's camera views don't match the declared input tracks —
        # a checkpoint this port cannot serve must never reach a session.
        self._schema = build_schema(
            views=self._views,
            # The inbound tracks this port declares. An ``Input`` subclass is
            # not a dataclass under the standalone runtime — the base resolves
            # its annotated tracks into ``__tracks__`` when the class is
            # declared, and binds a live buffer per track at connect.
            declared_views=tuple(RLDXInput.__tracks__),
            video_delta_indices=self._video_deltas,
            state_dims=self._state_dims,
            action_dims=action_dims,
            # RLDX-1 re-plans in lockstep with chunk execution, so the chunk
            # length and the execution horizon are the same checkpoint value.
            action_horizon=self._exec_horizon,
            exec_horizon=self._exec_horizon,
            control_hz=self._control_hz,
            resolution=(self._H, self._W),
            rtc_mode=rtc_mode,
            embodiment=str(getattr(embodiment, "value", embodiment)),
            state_fallback=self._state_fallback,
            state_source=_STATE_SOURCE,
            state_tag_keys=list(STATE_TAG_KEYS),
        )

    @session_started
    def on_session_started(self) -> None:
        self._reset_memory()

    @connected
    async def on_connect(self) -> None:
        # Session-start handshake. Best-effort: @connected can fire
        # before the data channel finishes opening, and the transport silently
        # drops messages sent before then — so the inference loop re-announces
        # once media is flowing (which proves the channel is up), and
        # `get_schema` serves it on demand.
        self._schema_pending = True
        await self.send(ModelSchema(**self._schema))

    @event(
        name="get_schema",
        description="Re-send the loaded checkpoint's input/output contract. Valid any time. Emits `model_schema`.",
    )
    async def get_schema(self) -> None:
        await self.send(ModelSchema(**self._schema))

    @event(name="reset", description="Reset episode memory and frame buffers")
    async def reset(self) -> None:
        self.state._reset = True

    def _reset_memory(self) -> None:
        try:
            self._policy.reset()
        except Exception:
            pass
        self._last_state = None
        self._state_degraded = False
        self._last_skew_us = None
        self._frame_tags.clear()

    def _resolve_state(self) -> tuple[dict[str, np.ndarray] | None, str | None]:
        """Resolve the robot state for this tick, applying the fallback policy.

        Returns ``(state, degraded_reason)``:
          * ``state`` is the ``{"state.<key>": (1,1,D)}`` dict to feed the model,
            or ``None`` when inference should be skipped this tick.
          * ``degraded_reason`` is ``None`` when the state is fresh and valid,
            otherwise a short explanation of the fallback that engaged.

        Never silently zero-fills — that only happens under an explicit
        ``state_fallback: "zero"`` and still reports a reason.

        Two carriers, in order of preference:
          1. the **frame tag selected with the latest aligned commit** — state
             that arrived attached to a frame the policy will actually see;
          2. the ``state_json`` field, for a client whose SDK cannot tag frames.
        A tagging client never populates the field, and a field client never
        tags, so in practice one of the two is empty; the order only decides who
        wins for a client that does both.
        """
        parsed = self._frame_tags.parse(self._state_dims)
        if parsed is None:
            parsed = parse_state(self.state.state_json, self._state_dims)
        if parsed is not None:
            self._last_state = parsed
            return parsed, None

        reason = "robot state missing or unparseable (no usable frame tag, no state_json)"
        if self._state_fallback == "zero":
            return (
                zero_state(self._state_dims),
                f"{reason}; zero-filling (state_fallback=zero)",
            )
        if self._state_fallback == "hold_last" and self._last_state is not None:
            return self._last_state, f"{reason}; holding last-known state"
        # "error", or "hold_last" before any valid state has arrived.
        return None, (
            f"{reason}; skipping inference (state_fallback={self._state_fallback})"
        )

    async def inference(self):
        import cv2  # local: only needed at run time

        deltas = self._video_deltas
        step_s = (1.0 / self._control_hz) if self._control_hz > 0 else 0.0
        replan_s = (self._exec_horizon * step_s) if self._pace else 0.0
        # Per-control-step window buffer: deep enough to reach the oldest offset.
        maxlen = -min(deltas) + 1

        views = self._views
        bufs = {v: deque(maxlen=maxlen) for v in views}
        # Per-view (capture stamp, frame) recency, aligned across views at commit.
        recent = {v: deque(maxlen=_RECENT_FRAMES) for v in views}
        step = 0
        last_commit: float | None = None  # time.monotonic() of the last commit
        last_replan: float | None = None  # time.monotonic() of the last get_action

        while True:
            # Episode reset (clears RLDX memory + frame/state buffers).
            if self.state._reset:
                self.state._reset = False
                self._reset_memory()
                bufs = {v: deque(maxlen=maxlen) for v in views}
                recent = {v: deque(maxlen=_RECENT_FRAMES) for v in views}
                step = 0
                last_commit = None
                last_replan = None

            # Collect the freshest frame per view (non-blocking), keeping its
            # metadata attached until alignment selects the commit. Resize on
            # read so the commit tick only picks between ready frames.
            for v in views:
                frames = getattr(self.input, v).try_read(1, mode=ReadMode.LATEST)
                if frames:
                    f = frames[0].data  # (H, W, 3) uint8 RGB
                    if f.shape[0] != self._H or f.shape[1] != self._W:
                        f = cv2.resize(f, (self._W, self._H))
                    recent[v].append(
                        _FrameCandidate(
                            capture_time_us=frames[0].capture_time_us,
                            data=f,
                            metadata=frames[0].metadata,
                        )
                    )

            # Handshake delivery guard: the @connected send can race
            # the data channel opening and be dropped. Frames flowing prove the
            # channel is up, so re-announce once after each connect.
            if self._schema_pending and any(recent[v] for v in views):
                self._schema_pending = False
                await self.send(ModelSchema(**self._schema))

            # Need at least one frame in every view before we can step.
            if any(not recent[v] for v in views):
                yield Idle
                continue

            now = time.monotonic()

            # Commit one frame per view once per control step, downsampling the
            # live stream to control_hz so the buffer holds one frame per control
            # step (what delta_indices are expressed in) — independent of the
            # client's publish fps. Which frame is the aligner's call: the three
            # views are independent tracks and drift against each other, so
            # "freshest per view" is not one instant.
            if last_commit is None or (now - last_commit) >= step_s:
                aligned = _align_by_capture(recent)
                for v in views:
                    bufs[v].append(aligned.frames[v])
                # Replace the prior commit's tag even when this commit is
                # untagged. The configured fallback, not an unrelated newer or
                # older frame, decides what inference does next.
                self._frame_tags.clear()
                for candidate in aligned.state_candidates:
                    self._frame_tags.offer(
                        candidate.metadata,
                        capture_time_us=candidate.capture_time_us,
                    )
                self._last_skew_us = aligned.view_skew_us
                last_commit = now

            # Pace re-planning to the execution cadence: don't re-plan until the
            # client has had one exec_horizon of wall time to run the last chunk.
            if replan_s and last_replan is not None and (now - last_replan) < replan_s:
                yield Idle
                continue

            # Resolve robot state; never silently feeds zeros. Signal
            # once per transition into a degraded state, not every tick.
            robot_state, degraded = self._resolve_state()
            source_capture_us, source_seq = self._frame_tags.stamp
            if degraded is not None:
                if not self._state_degraded:
                    self._state_degraded = True
                    await self.send(CommandError(command="state", reason=degraded))
            else:
                self._state_degraded = False
            if robot_state is None:
                yield Idle
                continue

            # Build the strided temporal window per view from the per-control-step
            # buffer at the checkpoint's offsets. Output is
            # (1, T, H, W, 3) with the most-recent frame last — as the policy's
            # observation validator asserts (T == len(delta_indices)).
            def window(v: str) -> np.ndarray:
                hist = list(bufs[v])
                idxs = _window_indices(deltas, 1, len(hist))
                return np.stack([hist[i] for i in idxs])[None]

            obs = {
                **{f"video.{v}": window(v) for v in views},
                **robot_state,
                "annotation.human.action.task_description": (
                    self.state.task_description or "pick up the mug",
                ),
            }

            actions, _info = self._policy.get_action(obs)

            def chunk(key: str) -> list:
                return np.asarray(actions[f"action.{key}"][0]).astype(float).tolist()

            await self.send(ActionPrediction(
                end_effector_position=chunk("end_effector_position"),
                end_effector_rotation=chunk("end_effector_rotation"),
                gripper_close=chunk("gripper_close"),
                base_motion=chunk("base_motion"),
                control_mode=chunk("control_mode"),
                step=step,
                source_capture_us=source_capture_us,
                source_seq=source_seq,
                view_skew_us=self._last_skew_us,
            ))
            last_replan = now
            step += 1
            yield Idle
