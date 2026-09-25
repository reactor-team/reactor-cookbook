# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""RLDX-1 served through Reactor Runtime's native action-prediction step loop.

Wraps the upstream ``RLDXPolicy`` (RLWRLD/RLDX-1) behind the runtime. The client
(cpp_sdk) publishes the camera views as input tracks and sends the robot proprio
state + task; this pipeline assembles the observation exactly as
``RLDXSimPolicyWrapper.get_action`` expects, runs the policy, and streams the
predicted action chunk back as an :class:`ActionPrediction` message. No video is
emitted — this is a video-in -> action-out model.

The input spec (views, window, state/action dims) comes from the checkpoint's
modality config — RLWRLD's source of truth — read at load and announced to the
client in a ``model_schema`` message at session start (REA-4318), so a new
checkpoint reconfigures both sides without code changes here.

Robot state arrives as **video frame metadata** — every view's frame is tagged
with the proprio JSON — with the ``state_json`` field as the fallback for clients
that cannot tag frames. ``rldx1_state.py`` owns that seam; see
:meth:`RLDXPipeline._resolve_state`.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from reactor_runtime import (
    ApplicationError,
    ReactorApp,
    ReadMode,
    StepOutcome,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from rldx1_model import RLDXModel, RLDXModelInput, RLDXModelResult
from rldx1_rtc import (
    RTCRequest,
    RTCRequestMailbox,
    build_rtc_request,
)
from rldx1_schema import build_schema
from rldx1_state import STATE_TAG_KEYS, FrameStateTags, parse_state, zero_state
from rldx1_types import (
    ActionPrediction,
    CommandError,
    ModelSchema,
    RLDXInput,
    RLDXState,
)

VIEWS = ("left_view", "right_view", "wrist_view")

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


def _window_indices(
    deltas: list[int], frames_per_step: int, hist_len: int
) -> list[int]:
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

    State candidates come only from the frames selected for the commit. A newer
    frame held back by cross-view alignment must not contribute its state or
    source timestamp to the action chunk.

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

    tagged = [candidate for candidate in chosen.values() if candidate.metadata]
    state_source = min(
        tagged or list(chosen.values()),
        key=lambda candidate: abs(capture_stamp(candidate) - ref),
    )
    stamps = [capture_stamp(candidate) for candidate in chosen.values()]
    return _AlignedCommit(
        frames={view: candidate.data for view, candidate in chosen.items()},
        state_candidates=(state_source,),
        view_skew_us=max(stamps) - min(stamps),
    )


@dataclass(frozen=True)
class _Publication:
    """Application-owned attribution for the pending policy step."""

    request: RTCRequest | None
    source_capture_us: int | None
    source_seq: int | None
    view_skew_us: int | None
    started_at: float | None
    messages: tuple[ModelSchema | CommandError, ...]


class RLDXPipeline(ReactorApp):
    input: RLDXInput
    state: RLDXState
    buffer_size = 8

    def __init__(self) -> None:
        super().__init__()
        self._engine = RLDXModel()
        self._episode_id = 0
        self._applied_episode_id: int | None = None
        self._publication: _Publication | None = None
        self._last_completed_predictions = 0
        self._prediction_offset = 0

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path)

        from reactor_runtime import get_weights_path

        setup = self._engine.load(config, get_weights_path())
        self._H = int(config.get("height", 256))
        self._W = int(config.get("width", 256))
        self._video_deltas = list(setup.video_deltas)
        self._views = setup.views
        # Observation cadence — buffer one frame per control step. In ordinary
        # streaming mode the server re-plans once per full action chunk. In RTC
        # mode the client triggers each re-plan and owns its execution cursor.
        # Both modes derive frame cadence from RoboCasa's control rate:
        #   * commit the freshest frame every 1/control_hz s  -> the per-control-
        #     step window buffer the model was trained/eval'd on. delta_indices
        #     are in control steps, so this makes the strided window come out at
        #     the right real-time spacing regardless of the client's publish fps.
        self._control_hz = float(config.get("control_hz", 20))
        self._pace = bool(config.get("pace_inference", True))

        # Policy for missing / unparseable robot state (REA-4319):
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
        # Proprio tag selected from the latest aligned frame commit.
        self._frame_tags = FrameStateTags()
        # Cross-view capture spread of the last committed frames, echoed with
        # the chunk those frames fed.
        self._last_skew_us: int | None = None

        self._state_dims = setup.state_dims
        action_dims = setup.action_dims
        self._action_order = tuple(action_dims)
        self._action_dim = sum(action_dims.values())
        self._rtc_timing = setup.timing
        self._action_horizon = setup.timing.action_horizon
        self._exec_horizon = setup.timing.exec_horizon
        self._rtc_requests = RTCRequestMailbox()
        self._last_plan_id: int | None = None

        # Session-start handshake payload (REA-4318): the checkpoint-derived
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
            action_order=self._action_order,
            action_horizon=self._action_horizon,
            exec_horizon=self._exec_horizon,
            rtc_delay=self._rtc_timing.delay,
            inference_trigger=(
                "client_request" if self._rtc_timing.enabled else "streaming"
            ),
            control_hz=self._control_hz,
            resolution=(self._H, self._W),
            rtc_mode=setup.rtc_mode,
            embodiment=setup.embodiment,
            state_fallback=self._state_fallback,
            state_source=_STATE_SOURCE,
            state_tag_keys=list(STATE_TAG_KEYS),
        )

    @session_started
    def on_session_started(self) -> None:
        self._reset_memory()
        self._clear_observation_window()

    def _clear_observation_window(self) -> None:
        """Release per-connection frame windows while retaining policy memory."""
        maxlen = -min(self._video_deltas) + 1
        self._bufs = {v: deque(maxlen=maxlen) for v in self._views}
        self._recent = {v: deque(maxlen=_RECENT_FRAMES) for v in self._views}
        # The public step labels each connection's window; native memory continues.
        self._prediction_offset = self._last_completed_predictions
        self._last_commit = None
        self._last_replan = None

    @disconnected
    def on_disconnect(self) -> None:
        """Discard windows when the last observer leaves the session."""
        if not self.connected.is_set():
            self._clear_observation_window()

    @session_ended
    def on_session_ended(self) -> None:
        """Release buffered images and native policy memory at session end."""
        self._clear_observation_window()
        self._engine.reset()
        self._applied_episode_id = None
        self._publication = None

    @connected
    async def on_connect(self) -> None:
        # Session-start handshake (REA-4318). Best-effort: @connected can fire
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

    @event(
        name="request_action",
        description=(
            "Trigger one RTC inference. The first request uses base_plan_id=-1 "
            "and an empty prefix; later requests name the active plan and send "
            "the physical-unit actions that remain scheduled during inference."
        ),
    )
    async def request_action(
        self,
        request_id: int,
        base_plan_id: int,
        install_step: int,
        rtc_prefix_len: int,
        action_prefix: list[list[float]],
    ) -> None:
        if not self._rtc_timing.enabled:
            await self.send(
                CommandError(
                    command="request_action",
                    reason="RTC is disabled; set rtc_inference_mode and rtc_inference_delay",
                )
            )
            return
        try:
            request = build_rtc_request(
                request_id=request_id,
                base_plan_id=base_plan_id,
                install_step=install_step,
                rtc_prefix_len=rtc_prefix_len,
                action_prefix=action_prefix,
                expected_base_plan_id=self._last_plan_id,
                configured_delay=self._rtc_timing.delay,
                action_dim=self._action_dim,
            )
            self._rtc_requests.offer(request)
        except ValueError as exc:
            await self.send(CommandError(command="request_action", reason=str(exc)))

    @event(name="reset", description="Reset episode memory and frame buffers")
    async def reset(self) -> None:
        self._reset_memory()
        self._clear_observation_window()

    def _reset_memory(self) -> None:
        self._episode_id += 1
        self._last_completed_predictions = 0
        self._prediction_offset = 0
        self._last_state = None
        self._state_degraded = False
        self._last_skew_us = None
        self._frame_tags.clear()
        self._rtc_requests.clear()
        self._last_plan_id = None

    def _resolve_state(self) -> tuple[dict[str, np.ndarray] | None, str | None]:
        """Resolve the robot state for this tick, applying the fallback policy.

        Returns ``(state, degraded_reason)``:
          * ``state`` is the ``{"state.<key>": (1,1,D)}`` dict to feed the model,
            or ``None`` when inference should be skipped this tick.
          * ``degraded_reason`` is ``None`` when the state is fresh and valid,
            otherwise a short explanation of the fallback that engaged.

        Never silently zero-fills — that only happens under an explicit
        ``state_fallback: "zero"`` and still reports a reason (REA-4319).

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

        reason = (
            "robot state missing or unparseable (no usable frame tag, no state_json)"
        )
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

    async def process_input(self) -> RLDXModelInput:
        """Align incoming views and snapshot one client-triggered action request."""
        deltas = self._video_deltas
        step_s = (1.0 / self._control_hz) if self._control_hz > 0 else 0.0
        replan_s = (
            self._exec_horizon * step_s
            if self._pace and not self._rtc_timing.enabled
            else 0.0
        )
        views = self._views
        messages = []

        def snapshot(
            obs=None,
            options=None,
            rtc_request=None,
            source_capture_us=None,
            source_seq=None,
            now=None,
        ):
            new_episode = self._episode_id != self._applied_episode_id
            if obs is None and not messages and not new_episode:
                raise ApplicationError(
                    "Waiting for aligned observations and an eligible action request."
                )
            self._publication = _Publication(
                request=rtc_request,
                source_capture_us=source_capture_us,
                source_seq=source_seq,
                view_skew_us=self._last_skew_us,
                started_at=now,
                messages=tuple(messages),
            )
            return RLDXModelInput(self._episode_id, obs, options)

        # Collect the freshest frame per view (non-blocking), keeping its
        # metadata attached until alignment selects the commit. Resize on
        # read so the commit tick only picks between ready frames.
        for v in views:
            frames = getattr(self.input, v).try_read(1, mode=ReadMode.LATEST)
            if frames:
                f = frames[0].data  # (H, W, 3) uint8 RGB
                if f.shape[0] != self._H or f.shape[1] != self._W:
                    import cv2

                    f = cv2.resize(f, (self._W, self._H))
                self._recent[v].append(
                    _FrameCandidate(
                        capture_time_us=frames[0].capture_time_us,
                        data=f,
                        metadata=frames[0].metadata,
                    )
                )

        # Handshake delivery guard (REA-4318): the @connected send can race
        # the data channel opening and be dropped. Frames flowing prove the
        # channel is up, so re-announce once after each connect.
        if self._schema_pending and any(self._recent[v] for v in views):
            self._schema_pending = False
            messages.append(ModelSchema(**self._schema))

        # Need at least one frame in every view before we can step.
        if any(not self._recent[v] for v in views):
            return snapshot()

        now = time.monotonic()

        # Commit one frame per view once per control step, downsampling the
        # live stream to control_hz so the buffer holds one frame per control
        # step (what delta_indices are expressed in) — independent of the
        # client's publish fps. Which frame is the aligner's call: the three
        # views are independent tracks and drift against each other, so
        # "freshest per view" is not one instant.
        if self._last_commit is None or (now - self._last_commit) >= step_s:
            aligned = _align_by_capture(self._recent)
            for v in views:
                self._bufs[v].append(aligned.frames[v])
            self._frame_tags.clear()
            for candidate in aligned.state_candidates:
                self._frame_tags.offer(
                    candidate.metadata,
                    capture_time_us=candidate.capture_time_us,
                )
            self._last_skew_us = aligned.view_skew_us
            self._last_commit = now

        # RTC is client-triggered: timestamps identify the observation, but
        # the client-owned execution cursor decides when a replacement plan
        # may be installed. Never infer an RTC plan from server wall time.
        rtc_request = self._rtc_requests.pending
        if self._rtc_timing.enabled and rtc_request is None:
            return snapshot()

        # Pace re-planning to the execution cadence: don't re-plan until the
        # client has had one exec_horizon of wall time to run the last chunk.
        if (
            replan_s
            and self._last_replan is not None
            and (now - self._last_replan) < replan_s
        ):
            return snapshot()

        # Resolve robot state; never silently feeds zeros (REA-4319). Signal
        # once per transition into a degraded state, not every tick.
        robot_state, degraded = self._resolve_state()
        source_capture_us, source_seq = self._frame_tags.stamp
        if degraded is not None:
            if not self._state_degraded:
                self._state_degraded = True
                messages.append(CommandError(command="state", reason=degraded))
        else:
            self._state_degraded = False
        if robot_state is None:
            return snapshot()

        # Build the strided temporal window per view from the per-control-step
        # buffer at the checkpoint's offsets (REA-4317). Output is
        # (1, T, H, W, 3) with the most-recent frame last — as the policy's
        # observation validator asserts (T == len(delta_indices)).
        def window(v: str) -> np.ndarray:
            hist = list(self._bufs[v])
            idxs = _window_indices(deltas, 1, len(hist))
            return np.stack([hist[i] for i in idxs])[None]

        obs = {
            **{f"video.{v}": window(v) for v in views},
            **robot_state,
            "annotation.human.action.task_description": (
                self.state.task_description or "pick up the mug",
            ),
        }

        options = None
        if rtc_request is not None:
            # Consume only after frames and valid state are ready. The
            # policy accepts physical-unit actions and normalizes them at
            # its boundary before RTC prefix injection.
            rtc_request = self._rtc_requests.take()
            assert rtc_request is not None
            if rtc_request.rtc_prefix_len:
                options = {
                    "action_prefix": rtc_request.prefix_array(),
                    "rtc_prefix_len": rtc_request.rtc_prefix_len,
                }

        return snapshot(obs, options, rtc_request, source_capture_us, source_seq, now)

    def generate(self, input: RLDXModelInput) -> RLDXModelResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> None:
        """Publish the action chunk and advance the acknowledged plan chain."""
        if outcome.error is not None:
            self._publication = None
            # A native policy failure cannot be repaired by silently resetting memory.
            raise outcome.error
        result: RLDXModelResult = outcome.result
        publication = self._publication
        if publication is None:
            raise RuntimeError("No pending RLDX publication")
        self._publication = None
        self._applied_episode_id = result.episode_id
        self._last_completed_predictions = result.completed_predictions
        for message in publication.messages:
            await self.send(message)
        if result.actions is None:
            return

        def chunk(key: str) -> list:
            return np.asarray(result.actions[f"action.{key}"][0]).astype(float).tolist()

        request = publication.request
        await self.send(
            ActionPrediction(
                end_effector_position=chunk("end_effector_position"),
                end_effector_rotation=chunk("end_effector_rotation"),
                gripper_close=chunk("gripper_close"),
                base_motion=chunk("base_motion"),
                control_mode=chunk("control_mode"),
                step=result.completed_predictions - self._prediction_offset - 1,
                source_capture_us=publication.source_capture_us,
                source_seq=publication.source_seq,
                view_skew_us=publication.view_skew_us,
                request_id=request.request_id if request else None,
                plan_id=request.request_id if request else None,
                base_plan_id=request.base_plan_id if request else None,
                install_step=request.install_step if request else None,
                rtc_prefix_len=request.rtc_prefix_len if request else None,
            )
        )
        if request is not None:
            self._last_plan_id = request.request_id
        self._last_replan = publication.started_at
