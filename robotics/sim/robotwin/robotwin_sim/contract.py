# ──────────────────────────────────────────────────────────────────────────
# Wire contracts: BOTH of them, because this example is a gateway.
#
# Sim side (pickle-over-zmq, what the RoboTwin 2.0 authors' eval client
# speaks): the client connects to a broker frontend port, sends one pickled
# observation dict per control step and blocks for a pickled reply. That is
# the authors' own protocol, unchanged: this gateway binds their port and
# answers it, so their client, seeds, expert checks, instruction sampling
# and success predicates all run exactly as published.
#
# Reactor side (reactor-sdk): three named video tracks + two commands
# (set_task_description / set_state_json) in, action_prediction messages
# out. The model is LOCK-STEP: one request outstanding at a time, and the
# reply echoes the request's chunk_id in its `step` field.
#
# The one lossy hop is deliberate and worth naming: the authors' transport
# carried raw arrays, WebRTC carries H.264 video. Both arms of any A/B
# comparison share it, but a comparison against the authors' published
# numbers does not.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# ── Reactor side ────────────────────────────────────────────────────────────

#: Track names in the checkpoint's training-time camera order. The video
#: array's first axis is in this same order, so index 0 is the head view.
#: Publishing a wrist frame on the head track produces a wrong prediction with
#: no error raised.
VIEWS: tuple[str, str, str] = ("head_view", "left_wrist_view", "right_wrist_view")

CMD_SET_TASK = "set_task_description"
CMD_SET_STATE = "set_state_json"
CMD_RESET = "reset"
FIELD_TASK = "task_description"
FIELD_STATE = "state_json"
MESSAGE_ACTION_PREDICTION = "action_prediction"

#: (control steps, action dim). 14 = bimanual delta joint actions.
ACTION_SHAPE: tuple[int, int] = (32, 14)
#: (steps, dim): predicted future robot states. Diagnostic; nothing executes
#: these, but the authors' client expects the key, so it is relayed as-is.
PROPRIO_PRED_SHAPE: tuple[int, int] = (9, 16)
#: Proprioception width the model accepts.
PROPRIO_DIM = 16
#: The frame geometry the checkpoint was trained on.
FRAME_HW: tuple[int, int] = (240, 320)

# ── Sim side: the authors' pickle keys ──────────────────────────────────────

REQ_VIDEO = "video"          # float32 [V, H, W, 3] in [-1, 1]
REQ_PROPRIOS = "proprios"    # float64 [16]
REQ_PROMPT = "prompt"        # [str] (a one-element list) or str
REQ_ENV_RANK = "env_rank"
REQ_ROLLOUT_ID = "rollout_id"
REQ_STEP_ID = "step_id"
REQ_CFG = "cfg"
REPLY_ACTIONS = "actions"    # [32, 14]
REPLY_PROPRIOS = "proprios"  # [9, 16]


@dataclass
class SimRequest:
    """One decoded observation from the authors' client."""

    frames: dict[str, np.ndarray]
    """``{track_name: (H, W, 3) uint8}`` for all three of :data:`VIEWS`."""

    proprio: np.ndarray
    """16 finite floats: the current bimanual robot state."""

    task: str
    """The episode's language instruction."""

    seed: tuple[int, int, int]
    """``(env_rank, rollout_id, step_id)``. The model derives its sampling
    noise from this triple, so relaying it verbatim is what makes a rollout
    reproducible; drop it and every episode silently becomes a new sample."""

    cfg: float
    """Guidance scale, as the client set it."""


def frames_from_video(video: np.ndarray) -> dict[str, np.ndarray]:
    """Recover the uint8 frames the simulator rendered from the client's video.

    The client normalises its rendered uint8 frames to ``[-1, 1]`` before
    sending them, and that mapping is bijective from uint8, so inverting it
    yields the *exact* pixels SAPIEN produced, not an approximation. The
    ``round()`` is what undoes the float32 division error; without it a
    boundary value lands one LSB low.

    Args:
        video: ``[V, H, W, 3]`` float in ``[-1, 1]``, views in :data:`VIEWS`
            order.
    """
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != len(VIEWS) or arr.shape[3] != 3:
        raise ValueError(
            f"video has shape {arr.shape}, want ({len(VIEWS)}, H, W, 3) with "
            f"views in {VIEWS} order"
        )
    frames8 = np.clip(np.round((arr + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return {
        view: np.ascontiguousarray(frames8[i]) for i, view in enumerate(VIEWS)
    }


def decode_request(data: dict) -> SimRequest:
    """Turn one unpickled client request into a :class:`SimRequest`."""
    if REQ_VIDEO not in data or REQ_PROPRIOS not in data:
        raise ValueError(
            f"request is missing {REQ_VIDEO!r}/{REQ_PROPRIOS!r}. Is this the "
            "RoboTwin client from the authors' evaluation stack?"
        )
    prompt = data.get(REQ_PROMPT, "")
    if isinstance(prompt, (list, tuple)):
        prompt = prompt[0] if prompt else ""

    proprio = np.asarray(data[REQ_PROPRIOS], dtype=np.float64).reshape(-1)
    if proprio.shape != (PROPRIO_DIM,):
        raise ValueError(
            f"proprios must be {PROPRIO_DIM} floats, got {proprio.shape}"
        )
    if not np.isfinite(proprio).all():
        # The model drops a malformed request rather than zero-filling it, so
        # fail here instead of waiting out the reply timeout.
        raise ValueError("proprios contains non-finite values")

    return SimRequest(
        frames=frames_from_video(data[REQ_VIDEO]),
        proprio=proprio,
        task=str(prompt),
        seed=(
            int(data.get(REQ_ENV_RANK, 0)),
            int(data.get(REQ_ROLLOUT_ID, 0)),
            int(data.get(REQ_STEP_ID, 0)),
        ),
        cfg=float(data.get(REQ_CFG, 0.0)),
    )


def encode_state_json(
    request: SimRequest, chunk_id: int, retry: int = 0
) -> str:
    """Build the ``state_json`` payload for one prediction.

    ``retry`` must change the string, not the seeds. The model answers once
    per *distinct* ``state_json`` value: an identical re-send is
    indistinguishable from the continuous re-delivery of unchanged state and
    is deduplicated, producing no reply at all. Bumping ``retry`` while
    holding ``chunk_id`` and the seed triple makes the re-send a new request
    whose answer is identical to the lost one, because the sampling noise
    is a pure function of the seeds.
    """
    env_rank, rollout_id, step_id = request.seed
    payload: dict[str, object] = {
        "proprio": request.proprio.tolist(),
        "chunk_id": int(chunk_id),
        "env_rank": env_rank,
        "rollout_id": rollout_id,
        "step_id": step_id,
        "cfg": request.cfg,
    }
    if retry:
        payload["retry"] = int(retry)
    return json.dumps(payload)


def decode_prediction(data: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """``action_prediction`` payload -> ``(step, actions, proprios)``.

    ``step`` is the model's echo of the request's ``chunk_id``; the caller
    compares it before using the chunk, because a reply that crosses an
    episode reset is the classic harness bug.
    """
    if REPLY_ACTIONS not in data:
        raise ValueError(f"action_prediction has no {REPLY_ACTIONS!r} field")
    actions = np.asarray(data[REPLY_ACTIONS], dtype=np.float64)
    proprios = np.asarray(data.get(REPLY_PROPRIOS, []), dtype=np.float64)
    if actions.shape != ACTION_SHAPE or not np.isfinite(actions).all():
        raise ValueError(
            f"actions have shape {actions.shape}, want {ACTION_SHAPE} and finite"
        )
    return int(data.get("step", -1)), actions, proprios


def encode_reply(actions: np.ndarray, proprios: np.ndarray) -> dict:
    """The reply dict the authors' client unpickles.

    float64 both, matching what their own policy server returned. The client
    integrates these into poses, and a dtype change there is a silent
    precision change in the executed trajectory.
    """
    return {
        REPLY_ACTIONS: np.asarray(actions, dtype=np.float64),
        REPLY_PROPRIOS: np.asarray(proprios, dtype=np.float64),
    }
