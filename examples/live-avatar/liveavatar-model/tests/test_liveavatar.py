"""LiveAvatar's application half and the model half's CPU helpers, without a GPU or weights.

The application half is driven through its planning and landing with a fake model and a hand-set clock; speech
arrives the way the LiveKit transport cuts a byte stream. Run from the model folder::

    PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ClientInfo, CommandError, InputFrame, UploadedFile
from reactor_runtime.interface.internal.reactor_core import MediaOps
from reactor_runtime.manifest import check_livekit_block, load_config

from avatar_app import SAMPLE_RATE, AvatarSettings, PlaybackFinished, PlaybackStarted
from avatar_contract import AvatarInput, AvatarResult
from liveavatar_app import LiveAvatar
from liveavatar_model import BLOCK_SAMPLES, LiveAvatarModel, crop_portrait

pytestmark = pytest.mark.asyncio

HERE = Path(__file__).resolve().parents[1]
BLOCK_FRAMES = 12
FRAME_SAMPLES = 240  # 10 ms at 24 kHz, the size the transport cuts a byte stream into


class FakeModel:
    """The model by shape: records each block and returns ``frames`` frames per block."""

    fps = 25
    block_frames = BLOCK_FRAMES
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.blocks: list[AvatarInput] = []
        self.resets = 0
        self.frames = BLOCK_FRAMES

    def generate(self, input: AvatarInput) -> AvatarResult:
        self.blocks.append(input)
        frames = np.zeros((self.frames, 8, 8, 3), dtype=np.uint8)
        return AvatarResult(
            frames=frames, block=len(self.blocks) - 1, avatar_id=input.avatar_id
        )

    def reset(self) -> None:
        self.resets += 1


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Harness:
    """The app with a live session, a fake model, and what it sent, flushed and emitted."""

    def __init__(self, **settings: Any) -> None:
        self.app = app = LiveAvatar()
        self.model = FakeModel()
        self.clock = Clock()
        self.sent: list[Any] = []
        self.flushes: list[bool] = []
        self.chunks: list[Any] = []
        app.settings = AvatarSettings(**settings)
        app.model = self.model
        app.block_frames = BLOCK_FRAMES
        app.samples_per_frame = SAMPLE_RATE // 25
        app.block_samples = BLOCK_SAMPLES
        app.default_avatar = b"portrait"
        app._init_loop_state()
        app.clock = self.clock
        app.session.active = True
        app._on_loop_ready()
        app.bind_output(
            broadcast=self.sent.append,
            addressed=lambda *args: None,
            media=self.chunks.append,
            media_ops=MediaOps(
                flush=self.flushes.append,
                set_rate=lambda fps: None,
                set_depth=lambda n: None,
            ),
        )

    def speak(self, stream: str, samples: np.ndarray, *, end: bool = False) -> None:
        meta = json.dumps(
            {"stream_id": stream, "attributes": {"sample_rate": "24000"}}
        ).encode()
        buffer = self.app.media.agent_audio
        for i in range(0, samples.size, FRAME_SAMPLES):
            buffer.push(
                InputFrame(
                    data=samples[i : i + FRAME_SAMPLES].reshape(1, -1), metadata=meta
                )
            )
        if end:
            ended = json.dumps({"stream_id": stream, "ended": True}).encode()
            buffer.push(
                InputFrame(data=np.zeros((1, 0), dtype=np.int16), metadata=ended)
            )

    async def step(self) -> Any:
        """One turn of the loop with the clock unmoved: plan, render here, land. ``"wait"`` if it would wait."""
        plan = self.app._plan(self.clock.now)
        if plan is None:
            return "wait"
        result = self.model.generate(plan.input)
        plan.elapsed = 0.4
        return await self.app._land(plan, result)


def voiced(n: int) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (600, 400), (120, 90, 60)).save(buf, format="PNG")
    return buf.getvalue()


# -- the contract -------------------------------------------------------------------------------------------------


async def test_the_contract_is_lkits_avatar_shape() -> None:
    contract = LiveAvatar.__reactor_contract__
    assert set(contract.commands) == {
        "clear_buffer",
        "user_speaking_state",
        "set_avatar",
    }
    assert {"playback_started", "playback_finished"} <= set(contract.messages)
    assert set(contract.tracks) == {"agent_audio", "main_video", "main_audio"}


async def test_the_manifest_livekit_block_names_the_contract() -> None:
    config = load_config(HERE / "reactor.yaml")
    check_livekit_block(config.livekit, LiveAvatar.__reactor_contract__)
    assert config.livekit.media_inputs == {"lk.audio_stream": "agent_audio"}


# -- speech into blocks -------------------------------------------------------------------------------------------


async def test_a_block_of_speech_becomes_a_speech_block_with_its_audio() -> None:
    h = Harness()
    speech = voiced(BLOCK_SAMPLES)
    h.speak("s1", speech)
    out = await h.step()
    block = h.model.blocks[-1]
    assert not block.silent and np.array_equal(block.audio, speech)
    assert np.array_equal(out.main_audio.ravel(), speech)


async def test_an_utterance_end_renders_its_short_tail_padded_with_silence() -> None:
    h = Harness()
    h.speak("s1", voiced(BLOCK_SAMPLES // 2), end=True)
    await h.step()
    block = h.model.blocks[-1]
    assert np.count_nonzero(block.audio[BLOCK_SAMPLES // 2 :]) == 0
    assert np.count_nonzero(block.audio[: BLOCK_SAMPLES // 2]) > 0


async def test_a_shorter_first_block_gives_the_rest_of_its_speech_to_the_next() -> None:
    h = Harness()
    h.model.frames = 9
    h.speak("s1", voiced(2 * BLOCK_SAMPLES), end=True)
    first = await h.step()
    assert first.main_audio.size == 9 * 960
    h.model.frames = BLOCK_FRAMES
    h.clock.now += 0.4
    await h.step()
    second = h.model.blocks[-1].audio
    assert np.array_equal(
        second[: 3 * 960], voiced(2 * BLOCK_SAMPLES)[9 * 960 : 12 * 960]
    )


async def test_without_speech_the_avatar_idles_then_listens() -> None:
    h = Harness(listen_after_blocks=1)
    for _ in range(3):
        await h.step()
        h.clock.now += 0.5
    flags = [(b.silent, b.listening) for b in h.model.blocks]
    assert flags == [(True, False), (True, True), (True, True)]


async def test_idle_blocks_wait_while_enough_video_is_queued() -> None:
    h = Harness()
    h.app.session.render_s = 0.2
    await h.step()
    assert await h.step() == "wait"
    h.clock.now += 0.3
    assert await h.step() != "wait"


# -- playback reports and interrupts ------------------------------------------------------------------------------


async def test_playback_is_reported_started_and_finished_on_the_clock() -> None:
    h = Harness()
    h.speak("s1", voiced(BLOCK_SAMPLES), end=True)
    await h.step()
    await h.app.report_playback()
    assert isinstance(h.sent[-1], PlaybackStarted)
    h.clock.now += 0.5
    await h.app.report_playback()
    assert isinstance(h.sent[-1], PlaybackFinished) and not h.sent[-1].interrupted
    assert h.sent[-1].playback_position == pytest.approx(0.48)


async def test_clear_buffer_cuts_the_utterance_and_reports_how_far_it_played() -> None:
    h = Harness()
    h.speak("s1", voiced(3 * BLOCK_SAMPLES))
    await h.step()
    h.clock.now += 0.2
    await h.app.clear_buffer()
    report = h.sent[-1]
    assert isinstance(report, PlaybackFinished) and report.interrupted
    assert report.playback_position == pytest.approx(0.2)
    assert h.flushes[-1] is True
    h.speak("s1", voiced(BLOCK_SAMPLES))
    await h.step()
    assert h.model.blocks[-1].silent


async def test_a_speech_block_rendered_across_an_interrupt_is_dropped() -> None:
    h = Harness()
    h.speak("s1", voiced(2 * BLOCK_SAMPLES))
    plan = h.app._plan(h.clock.now)
    h.app.session.rendering = plan
    await h.app.clear_buffer()
    assert await h.app._land(plan, h.model.generate(plan.input)) is None


# -- commands -----------------------------------------------------------------------------------------------------


async def test_set_avatar_accepts_an_image_from_the_starting_input_only() -> None:
    h = Harness()
    image = UploadedFile(name="me.png", mime_type="image/png", data=png())
    with pytest.raises(CommandError) as refused:
        await h.app.set_avatar(image=image, client=ClientInfo(id=1, joined_at=0.0))
    assert refused.value.code == "not_allowed"
    with pytest.raises(CommandError) as wrong:
        await h.app.set_avatar(
            image=UploadedFile(name="a.txt", mime_type="text/plain", data=b"x"),
            client=ClientInfo(id=0, joined_at=0.0, system=True),
        )
    assert wrong.value.code == "unsupported_media"
    await h.app.set_avatar(
        image=image, client=ClientInfo(id=0, joined_at=0.0, system=True)
    )
    await h.step()
    assert h.model.blocks[-1].avatar == image.data


async def test_user_speaking_state_makes_idle_blocks_listen() -> None:
    h = Harness(listen_after_blocks=5)
    h.app.user_speaking_state(state="speaking")
    await h.step()
    assert h.model.blocks[-1].listening


# -- the model half's CPU helpers ---------------------------------------------------------------------------------


async def test_crop_portrait_covers_and_centres_the_output_size() -> None:
    frame = crop_portrait(png(), 512, 512)
    assert frame.shape == (512, 512, 3) and frame.dtype == np.uint8


async def test_speech_resamples_to_the_encoders_block_without_an_edge() -> None:
    model = LiveAvatarModel()
    tone = voiced(2 * BLOCK_SAMPLES)
    a = model._to_encoder_rate(tone[:BLOCK_SAMPLES])
    b = model._to_encoder_rate(tone[BLOCK_SAMPLES:])
    assert a.shape == b.shape == (BLOCK_SAMPLES * 2 // 3,)
    joined = np.concatenate([a, b])
    assert np.abs(np.diff(joined)).max() < 0.05


class FakeEngine:
    """The engine by shape: records whether each block was rendered as listening."""

    width = height = 512
    bound = True
    block = 0

    def __init__(self) -> None:
        self.listening: list[bool] = []
        self.timing = type(
            "Timing",
            (),
            {"audio_ms": 0.0, "dit_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0},
        )()

    def render(self, audio: np.ndarray, *, listening: bool = False) -> np.ndarray:
        self.listening.append(listening)
        return np.zeros((BLOCK_FRAMES, 8, 8, 3), dtype=np.uint8)


async def test_silent_and_listening_blocks_render_on_the_listening_prompt() -> None:
    model = LiveAvatarModel()
    engine = FakeEngine()
    model.engine = engine  # type: ignore[assignment]  # the engine by shape
    model.avatar_id = 1
    for silent, listening in [
        (False, False),
        (True, False),
        (True, True),
        (False, True),
    ]:
        audio = (
            np.zeros(BLOCK_SAMPLES, dtype=np.int16) if silent else voiced(BLOCK_SAMPLES)
        )
        model.generate(
            AvatarInput(
                audio=audio,
                silent=silent,
                listening=listening,
                avatar=None,
                avatar_id=1,
            )
        )
    assert engine.listening == [False, True, True, True]
