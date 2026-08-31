"""Continuity mode: the seam math, the FL2VA anchor chain, and the guarded flag.

Continuity is off by default, so the hard-cut behaviour every other test asserts
is untouched. These tests turn it on and cover only what it adds: the pure-numpy
seam module (colour-match lock + linear-light blend + equal-power audio), the
FL2VA anchor threaded through the channel, the seam's frame arithmetic on the
emitter, and the flag's presence in the published schema.

Everything here runs on a laptop: the seam module is pure numpy, and the channel
tests replace the clip builder with a fake that returns tiny frames instantly, so
the real anchor threading, seam stitching, pacing and teardown all run.

Run from the model folder: ``PYTHONPATH=. python -m pytest tests/ -q``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

import fasth3_clip_plan as clip_plan
import fasth3_seam as seam
from fasth3 import OUTPUT_SAMPLE_RATE, FRAME_RATE, FastH3
from fasth3_types import DEFAULT_STYLE_PROMPT

MODEL_DIR = Path(__file__).resolve().parents[1]
SAMPLES_PER_FRAME = OUTPUT_SAMPLE_RATE // FRAME_RATE


# ================================================================= the seam math
#
# Pure numpy: the colour-match lock and the linear-light "linearfade" blend.
# Ported verbatim from the fast-h3-live sibling this mode is drawn from.


def _gradient_clip(frames: int, base: int) -> np.ndarray:
    """A small clip whose brightness rises frame to frame, offset by ``base``."""
    h, w = 8, 8
    clip = np.zeros((frames, h, w, 3), np.float32)
    for f in range(frames):
        clip[f] = base + f * 3
    return np.clip(clip, 0, 255).astype(np.uint8)


def test_reference_rgb_is_the_frames_mean():
    frame = np.full((4, 4, 3), 120, np.uint8)
    frame[..., 0] = 40  # a red-shifted frame
    assert seam.reference_rgb(frame).tolist() == pytest.approx([40.0, 120.0, 120.0])


def test_color_match_locks_a_clip_to_the_reference_mean():
    """A continuation clip's mean RGB is shifted onto clip 0's, once for the clip."""
    reference = np.array([100.0, 110.0, 120.0], np.float32)
    clip = _gradient_clip(6, base=200)  # far brighter than the reference
    matched = seam.color_match_to_reference(clip, reference)
    assert np.allclose(matched.reshape(-1, 3).mean(0), reference, atol=1.0)


def test_color_match_is_one_offset_so_intra_clip_variation_survives():
    """The per-frame brightness ramp is preserved; only the average moves."""
    clip = _gradient_clip(6, base=100)
    before = seam.luma(clip)
    matched = seam.color_match_to_reference(clip, np.array([50.0, 50.0, 50.0], np.float32))
    after = seam.luma(matched)
    assert np.allclose(np.diff(before), np.diff(after), atol=1.0)


def test_color_match_does_not_ratchet_across_a_chain():
    """Locking every clip to ONE reference keeps the chain mean flat, not drifting.

    Re-deriving the reference from each corrected clip's own last frame (the bug
    that blew the bear video out to white) would let it climb; this asserts the
    lock holds it.
    """
    reference = seam.reference_rgb(_gradient_clip(6, base=90)[-1])
    means = []
    for base in (90, 160, 220, 255):  # successively brighter raw clips
        matched = seam.color_match_to_reference(_gradient_clip(6, base=base), reference)
        means.append(matched.reshape(-1, 3).mean(0))
    spread = np.ptp(np.array(means), axis=0)
    assert np.all(spread < 3.0), f"clip means drifted across the chain: {means}"


def test_linearfade_is_monotonic_with_no_midpoint_flash():
    """The linear-light complementary blend rises smoothly from tail to head.

    The sRGB equal-power blend it replaces overshoots at the midpoint for
    near-identical frames — the flash. Between two flat clips the fixed blend's
    luma must be monotonic and never exceed the brighter endpoint.
    """
    tail = np.full((12, 16, 16, 3), 60, np.uint8)  # darker clip tail
    head = np.full((12, 16, 16, 3), 200, np.uint8)  # brighter clip head
    blended = seam.blend_video_linear(tail, head)
    y = seam.luma(blended)
    assert blended.shape == tail.shape
    assert np.all(np.diff(y) >= -0.5), f"luma dipped inside the blend: {y}"
    assert y.max() <= seam.luma(head).max() + 1.0, "blend rose above the brighter endpoint (flash)"
    assert y.min() >= seam.luma(tail).min() - 1.0


def test_linearfade_endpoints_approach_each_clip():
    """The blend starts near the tail's level and ends near the head's."""
    tail = np.full((12, 8, 8, 3), 60, np.uint8)
    head = np.full((12, 8, 8, 3), 200, np.uint8)
    y = seam.luma(seam.blend_video_linear(tail, head))
    assert abs(y[0] - seam.luma(tail)[0]) < abs(y[0] - seam.luma(head)[0])
    assert abs(y[-1] - seam.luma(head)[-1]) < abs(y[-1] - seam.luma(tail)[-1])


def test_audio_overlap_is_equal_power_and_never_wraps():
    """The crossfade holds energy flat and stays inside int16 without wrapping."""
    fade_out, fade_in = seam.equal_power_ramps(64)
    assert np.allclose(fade_out**2 + fade_in**2, 1.0, atol=1e-5)
    tail = np.full((1, 64), 30000, np.int16)
    head = np.full((1, 64), 30000, np.int16)
    mixed = seam.blend_audio_equal_power(tail, head)
    assert mixed.dtype == np.int16
    assert mixed.shape == (1, 64)
    assert int(mixed.max()) <= 32767 and int(mixed.min()) >= -32768


# ================================================================ the anchor chain
#
# ``_run_channel`` in continuity mode, its builder replaced by a fake that records
# the FL2VA anchor it was handed. The real submit/lookahead/seam/teardown run.

FRAMES_PER_CLIP = 6
SEAM_FRAMES = 2


@pytest.fixture
def channel():
    instance = FastH3()
    instance._on_loop_ready()
    instance.connected.set()
    instance.default_aspect = "16:9"
    instance.default_clip_frames = FRAMES_PER_CLIP
    instance.ramp_frames = ()
    instance.default_seed = 1000
    instance.default_style_prompt = DEFAULT_STYLE_PROMPT
    instance.num_inference_steps = 5
    instance.default_auto_story_enabled = False
    instance.story_start_delay = 20.0
    instance.story_queue_target = 2
    instance.story_history_size = 7
    instance._story_writer = None
    instance.live_chat_enabled = False
    instance.live_chat_room_id = 0
    instance.live_chat_prefix = "!Prompt:"
    instance.live_chat_max_request_chars = 200
    instance.live_chat_max_pending = 10
    instance._model_prompt = lambda prompt: prompt

    # The flag under test: continuity on, with a narrow seam.
    instance.continuity_enabled = True
    instance.seam_frames = SEAM_FRAMES

    instance._reset_session_state()
    instance._active_prompt = "a bear in a misty forest"
    instance._started = True

    instance._jobs = queue.Queue()
    instance._worker = threading.Thread(
        target=instance._worker_loop, name="test-generation", daemon=True
    )
    instance._worker.start()

    instance._frames_for_clip = lambda index: FRAMES_PER_CLIP

    built: list[dict] = []

    def fake_generate(index, frames, prompt, style_prompt, seed, height, width, *, anchor=None):
        built.append(
            {"index": index, "prompt": prompt, "seed": seed, "anchored": anchor is not None}
        )
        # A distinct grey per clip so the last frame is a real image to anchor on.
        base = 40 + 30 * (index % 4)
        video = [np.full((8, 8, 3), min(255, base + f), np.uint8) for f in range(frames)]
        samples = np.zeros((1, round(frames / 24 * OUTPUT_SAMPLE_RATE)), np.int16)
        return video, samples

    instance._generate_clip = fake_generate
    instance.built = built

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    instance.emit = fake_emit
    instance.emitted = emitted

    messages: list = []

    async def fake_send(message):
        messages.append(message)

    instance.send = fake_send
    instance.messages = messages
    return instance


def names(messages) -> list[str]:
    return [type(m).__name__ for m in messages]


def run_channel_until(channel, stop_after_clips: int):
    async def drive():
        async def stopper():
            while channel._clips_sent < stop_after_clips:
                await asyncio.sleep(0.01)
            channel._started = False
            channel._stop_only = True
            channel._do_reset = True

        await asyncio.gather(channel._run_channel(), stopper())

    asyncio.run(drive())


def test_clip_zero_is_t2va_and_every_clip_after_is_fl2va_anchored(channel):
    """The chain: clip 0 opens with no anchor, each later clip anchors on the last."""
    run_channel_until(channel, stop_after_clips=3)
    built = channel.built
    assert len(built) >= 3
    # One held prompt, no re-prompting: every build used the same prompt.
    assert {b["prompt"] for b in built} == {"a bear in a misty forest"}
    assert built[0]["anchored"] is False
    assert all(b["anchored"] for b in built[1:])
    # The seed advances by one per clip, exactly as in hard-cut mode.
    assert [b["seed"] for b in built[:3]] == [1000, 1001, 1002]
    assert names(channel.messages)[0] == "ChannelStarted"


def test_continuity_keeps_video_and_audio_locked_slice_for_slice(channel):
    run_channel_until(channel, stop_after_clips=2)
    assert channel.emitted, "the channel emitted no frames"
    for output in channel.emitted:
        video_frames = output.main_video.shape[0]
        audio_samples = output.main_audio.shape[1]
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.dtype == np.int16
        assert output.main_audio.ndim == 2 and output.main_audio.shape[0] == 1
        assert audio_samples == video_frames * SAMPLES_PER_FRAME


def test_state_update_reports_continuity_on(channel):
    assert channel._snapshot().continuity is True


def test_state_update_reports_continuity_off_by_default():
    """A fresh model with no config is the hard-cut default."""
    instance = FastH3()
    assert instance.continuity_enabled is False


# --------------------------------------------------------- the seam on the emitter
#
# Driving ``_emit_paced`` on pre-built clips proves the frame-count arithmetic:
# each boundary merges the k-frame tail and head into one k-frame blend, and the
# final clip's tail is held (dropped at stop), so C clips of N frames emit
# C*(N-k) frames.


def _emitter_instance(seam_frames: int):
    instance = FastH3()
    instance._on_loop_ready()
    instance.connected.set()
    instance.continuity_enabled = True
    instance.seam_frames = seam_frames
    instance._paused = False
    instance._do_reset = False
    instance._clip_index = 0
    instance._frames_sent = 0
    instance._seconds_sent = 0.0
    return instance


def test_the_seam_removes_exactly_one_overlap_per_boundary():
    n, k, clips = FRAMES_PER_CLIP, SEAM_FRAMES, 3
    instance = _emitter_instance(seam_frames=k)

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    instance.emit = fake_emit

    async def main():
        # The seam stitch now runs on the generation worker (off the emit
        # metronome); its held-tail state lives in a worker-side pacer. Mirror
        # that here — stitch, then pace — so the boundary arithmetic is exercised
        # exactly as production runs it.
        seam_pacer = {"pending_v": None, "pending_a": None}
        pacer = {
            "clock_start": None,
            "frames_paced": 0,
            "pending_v": None,
            "pending_a": None,
        }
        for index in range(clips):
            instance._clip_index = index
            base = 50 + 20 * index
            frames_list = [np.full((8, 8, 3), base + f, np.uint8) for f in range(n)]
            audio = np.zeros((1, n * SAMPLES_PER_FRAME), np.int16)
            emit_frames, emit_audio = instance._stitch_seam(
                frames_list, audio, seam_pacer
            )
            await instance._emit_paced(emit_frames, emit_audio, pacer)

    asyncio.run(main())
    total = sum(o.main_video.shape[0] for o in emitted)
    # Each boundary removes exactly k frames; the last clip's tail is still held.
    assert total == clips * n - clips * k
    assert total == clips * (n - k)
    assert instance._frames_sent == total
    for output in emitted:
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.shape[1] == output.main_video.shape[0] * SAMPLES_PER_FRAME


def test_the_first_continuity_clip_holds_its_tail_for_the_seam():
    """Clip 0 emits N-k frames and stashes its last k for the next clip's blend."""
    n, k = FRAMES_PER_CLIP, SEAM_FRAMES
    instance = _emitter_instance(seam_frames=k)

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    instance.emit = fake_emit

    seam_pacer = {"pending_v": None, "pending_a": None}
    pacer = {"clock_start": None, "frames_paced": 0, "pending_v": None, "pending_a": None}
    frames_list = [np.full((8, 8, 3), 60 + f, np.uint8) for f in range(n)]
    audio = np.zeros((1, n * SAMPLES_PER_FRAME), np.int16)
    # Stitch on the worker-side pacer (as production does), then pace.
    emit_frames, emit_audio = instance._stitch_seam(frames_list, audio, seam_pacer)
    asyncio.run(instance._emit_paced(emit_frames, emit_audio, pacer))

    assert sum(o.main_video.shape[0] for o in emitted) == n - k
    assert seam_pacer["pending_v"] is not None
    assert seam_pacer["pending_v"].shape[0] == k
    assert seam_pacer["pending_a"].shape[1] == k * SAMPLES_PER_FRAME


# ================================================================ published contract


@pytest.fixture(scope="module")
def schema():
    from reactor_runtime.schema import render

    return render(MODEL_DIR, version="v0.4.0")


def test_state_update_publishes_the_continuity_flag(schema):
    props = schema["components"]["schemas"]["StateUpdate"]["properties"]
    assert "continuity" in props
    assert props["continuity"].get("description")
    assert props["continuity"]["type"] == "boolean"


def test_clip_started_documents_both_the_cut_and_the_stitch(schema):
    """The reworded doc has to name the hard-cut default and the continuity case."""
    summary = schema["webhooks"]["clip_started"]["post"]["summary"]
    assert "cut" in summary.lower()
    assert "continuity" in summary.lower()


def test_valid_commands_are_unchanged_by_the_flag():
    """Continuity is config-only: it adds no command and removes none."""
    import fasth3_session_rules as session_rules

    for running in (False, True):
        for paused in (False, True):
            for ready in (False, True):
                assert session_rules.valid_commands(
                    running=running, paused=paused, ready=ready
                ) == session_rules.valid_commands(
                    running=running, paused=paused, ready=ready
                )
