"""AURA's application half and the model half's CPU helpers, without a GPU or weights.

The application half is the same file as the LiveAvatar model's, and its behaviour is tested there; these tests
check the copy, the contract, and that it drives AURA's 16-frame chunks. Run from the model folder::

    PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from reactor_runtime.interface.internal.reactor_core import MediaOps
from reactor_runtime.manifest import check_livekit_block, load_config

from aura_app import AuraAvatar
from aura_model import (
    BLOCK_FRAMES,
    BLOCK_SAMPLES,
    FPS,
    SAMPLE_RATE,
    AuraModel,
    NoAvatar,
    feature_positions,
)
from avatar_app import AvatarSettings
from avatar_contract import AvatarInput, AvatarResult

pytestmark = pytest.mark.asyncio

HERE = Path(__file__).resolve().parents[1]
LIVEAVATAR = HERE.parent / "liveavatar-model"


class FakeModel:
    """AURA by shape: 16 frames of 16 FPS video per 1.0 s chunk."""

    fps = FPS
    block_frames = BLOCK_FRAMES
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.blocks: list[AvatarInput] = []

    def generate(self, input: AvatarInput) -> AvatarResult:
        self.blocks.append(input)
        frames = np.zeros((BLOCK_FRAMES, 8, 8, 3), dtype=np.uint8)
        return AvatarResult(
            frames=frames, block=len(self.blocks) - 1, avatar_id=input.avatar_id
        )

    def reset(self) -> None:
        pass


async def test_the_application_half_is_the_one_the_liveavatar_model_uses() -> None:
    for name in ("avatar_app.py", "avatar_contract.py"):
        assert (HERE / name).read_bytes() == (LIVEAVATAR / name).read_bytes(), name


async def test_the_contract_is_lkits_avatar_shape() -> None:
    contract = AuraAvatar.__reactor_contract__
    assert set(contract.commands) == {
        "clear_buffer",
        "user_speaking_state",
        "set_avatar",
    }
    assert {"playback_started", "playback_finished", "avatar_failed"} <= set(
        contract.messages
    )
    assert set(contract.tracks) == {"agent_audio", "main_video", "main_audio"}


async def test_the_manifest_livekit_block_names_the_contract() -> None:
    config = load_config(HERE / "reactor.yaml")
    check_livekit_block(config.livekit, AuraAvatar.__reactor_contract__)
    assert config.livekit.media_inputs == {"lk.audio_stream": "agent_audio"}


async def test_a_second_of_speech_becomes_one_chunk_at_16_fps() -> None:
    app = AuraAvatar()
    model = FakeModel()
    app.settings = AvatarSettings()
    app.model = model
    app.block_frames = BLOCK_FRAMES
    app.samples_per_frame = SAMPLE_RATE // FPS
    app.block_samples = BLOCK_SAMPLES
    app.default_avatar = b"portrait"
    app._init_loop_state()
    app.clock = lambda: 100.0
    app.session.active = True
    app._on_loop_ready()
    chunks: list[object] = []
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda *args: None,
        media=chunks.append,
        media_ops=MediaOps(
            flush=lambda hold: None, set_rate=lambda fps: None, set_depth=lambda n: None
        ),
    )
    plan = app._plan(100.0)
    assert (
        plan is not None
        and plan.input.audio.size == BLOCK_SAMPLES
        and plan.input.avatar == b"portrait"
    )
    await app._land(plan, model.generate(plan.input))
    assert chunks


async def test_a_new_clip_without_a_portrait_is_refused_before_any_gpu_work() -> None:
    model = AuraModel()
    silence = np.zeros(BLOCK_SAMPLES, np.int16)
    with pytest.raises(NoAvatar):
        model.generate(
            AvatarInput(
                audio=silence, silent=True, listening=False, avatar=None, avatar_id=3
            )
        )


async def test_features_are_read_where_the_reference_reads_a_whole_file() -> None:
    first = feature_positions(0, 0)
    assert first[0] == 0 and np.all(np.diff(first) >= 0) and first[-1] < 12.5
    history = 4 * SAMPLE_RATE
    later = feature_positions(5, history)
    assert later[0] == pytest.approx(4 * 12.5)
    assert np.all(later >= 4 * 12.5) and np.all(later < 5 * 12.5)
