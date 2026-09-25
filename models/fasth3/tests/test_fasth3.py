"""FastH3's clip geometry, command contract, channel loop, schema and manifest.

Everything here runs on a laptop: the GPU work sits behind ``load()``, which
these tests never call, and the clip builder is replaced where the channel loop
itself is under test.

Run from the model folder: ``PYTHONPATH=. python -m pytest tests/ -q``.
"""

from __future__ import annotations

import asyncio
import io
import json
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import fasth3_clip_plan as clip_plan
import fasth3_session_rules as session_rules
from fasth3 import (
    EMIT_FRAMES,
    MAX_GENERATED_PROMPT_CHARS,
    SPONGEBOB_STORY_BIBLE,
    STORY_FIELDS,
    FastH3,
    _PromptOrigin,
    _story_plot,
    _StoryWriter,
)
from fasth3_live_chat import LivePromptRequest, parse_live_prompt
from fasth3_types import DEFAULT_STYLE_PROMPT

MODEL_DIR = Path(__file__).resolve().parents[1]


def test_runtime35_uses_the_public_custom_loop_interface():
    from reactor_runtime import ReactorApp

    assert FastH3.__bases__ == (ReactorApp,)
    assert FastH3.run is not ReactorApp.run


# --------------------------------------------------------------- clip geometry
#
# The rules a clip length has to satisfy to be generatable.


def test_bounds_are_the_generatable_range():
    """5 s aligns up to 124 frames; 15 s aligns up to 362 and is out of range."""
    assert clip_plan.MIN_FRAMES == 124
    assert clip_plan.MAX_FRAMES == 345
    # The ceiling is not 15 s: 360 frames aligns up to 362, which is 15.083 s
    # and past the cap, so 345 frames (14.375 s) is the longest clip there is.
    assert clip_plan.MAX_SECONDS == pytest.approx(14.375)
    assert 362 / clip_plan.FPS > 15.0


@pytest.mark.parametrize("frames", range(1, 400))
def test_align_frames_lands_on_the_chunk_grid(frames):
    aligned = clip_plan.align_frames(frames)
    assert aligned % 17 == 5
    assert aligned >= frames
    assert aligned - frames < 17


@pytest.mark.parametrize("seconds", [5.0, 5.167, 8.0, 10.0, 14.375, 15.0, 60.0])
def test_every_accepted_length_is_generatable(seconds):
    frames = clip_plan.frames_for_seconds(seconds)
    assert frames % 17 == 5
    assert clip_plan.MIN_FRAMES <= frames <= clip_plan.MAX_FRAMES
    assert 5.0 <= clip_plan.seconds_for_frames(frames) <= 15.0


def test_published_bounds_round_inward():
    """A client reading the schema sees tidy numbers that still snap cleanly."""
    assert clip_plan.MIN_SECONDS_PUBLISHED == 5.167
    assert clip_plan.MAX_SECONDS_PUBLISHED == 14.375
    assert clip_plan.MIN_SECONDS_PUBLISHED >= clip_plan.MIN_SECONDS
    assert clip_plan.MAX_SECONDS_PUBLISHED <= clip_plan.MAX_SECONDS
    for bound in (clip_plan.MIN_SECONDS_PUBLISHED, clip_plan.MAX_SECONDS_PUBLISHED):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_seconds_must_be_positive():
    with pytest.raises(ValueError):
        clip_plan.frames_for_seconds(0)


@pytest.mark.parametrize("aspect", clip_plan.ASPECT_CHOICES)
def test_every_offered_canvas_satisfies_the_checkpoint(aspect):
    height, width = clip_plan.canvas_for_choice(aspect)
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= 768 * 1344
    assert min(height, width) == 768 or height * width <= 768 * 1344
    assert 1 / 4 <= width / height <= 4


def test_default_canvas_is_the_measured_one():
    assert clip_plan.canvas_for_choice("16:9") == (768, 1344)


def test_unknown_aspect_is_rejected():
    with pytest.raises(ValueError):
        clip_plan.canvas_for_choice("32:9")


def test_ramp_opens_short_then_settles():
    ramp = clip_plan.parse_ramp([5.167])
    assert ramp == (124,)
    assert clip_plan.clip_frames(0, 345, ramp) == 124
    assert clip_plan.clip_frames(1, 345, ramp) == 345
    assert clip_plan.clip_frames(9, 345, ramp) == 345


def test_empty_ramp_is_a_uniform_cadence():
    assert clip_plan.clip_frames(0, 345, ()) == 345


def test_upstream_constants_have_not_drifted():
    """The duplicated constants must still match FastVideo's own.

    ``fasth3_clip_plan`` copies these rather than importing them, so that the
    schema renders without torch. This is the test that stops the copy going
    stale.
    """
    packing = pytest.importorskip(
        "fastvideo.pipelines.basic.minimax_h3.packing",
        reason="fastvideo is not installed on this machine",
    )
    assert clip_plan.FPS == packing.MINIMAX_H3_FPS
    assert clip_plan._FRAMES_PER_CHUNK == packing.MINIMAX_H3_FRAMES_PER_CHUNK
    assert clip_plan._LATENTS_PER_CHUNK == packing.MINIMAX_H3_LATENTS_PER_CHUNK
    assert clip_plan._MIN_DURATION == packing.MINIMAX_H3_MIN_DURATION
    assert clip_plan._MAX_DURATION == packing.MINIMAX_H3_MAX_DURATION
    assert clip_plan._SHORT_EDGE == packing.MINIMAX_H3_SHORT_EDGE
    assert clip_plan._MAX_PIXELS == packing.MINIMAX_H3_MAX_PIXELS
    assert clip_plan._CANVAS_MULTIPLE == packing.MINIMAX_H3_CANVAS_MULTIPLE

    for frames in (1, 100, 124, 200, 345):
        assert clip_plan.align_frames(frames) == packing.align_num_frames(frames)
    for aspect in clip_plan.ASPECT_CHOICES:
        ratio = clip_plan._ASPECT_RATIOS[aspect]
        assert clip_plan.canvas_for_choice(aspect) == packing.resolve_canvas_size(
            *ratio
        )


# --------------------------------------------------------------- session rules
#
# The command state machine clients read out of `state_update`.


def test_idle_without_a_prompt_cannot_start():
    commands = session_rules.valid_commands(running=False, paused=False, ready=False)
    assert "start" not in commands
    assert "set_prompt" in commands


def test_idle_with_a_prompt_can_start_and_set_the_canvas():
    commands = session_rules.valid_commands(running=False, paused=False, ready=True)
    assert "start" in commands
    assert "set_canvas" in commands


def test_the_canvas_is_locked_while_streaming():
    """The video track keeps one size for the life of a channel."""
    commands = session_rules.valid_commands(running=True, paused=False, ready=True)
    assert "set_canvas" not in commands
    assert "start" not in commands
    assert {"pause", "stop"} <= set(commands)


def test_a_paused_channel_offers_resume_not_pause():
    commands = session_rules.valid_commands(running=True, paused=True, ready=True)
    assert "resume" in commands
    assert "pause" not in commands


def test_conditions_and_reset_are_always_available():
    for running, paused in ((False, False), (True, False), (True, True)):
        commands = session_rules.valid_commands(
            running=running, paused=paused, ready=True
        )
        assert {
            "set_prompt",
            "set_auto_story",
            "set_style",
            "set_clip_seconds",
            "set_seed",
            "reset",
            "get_state",
        } <= set(commands)


# ------------------------------------------------------------ command contract
#
# The real handlers on a model whose ``load()`` never ran: everything they touch
# is session state and pure arithmetic, so the whole state machine — refusals
# included — is testable on a laptop.


def run(coro):
    """Drive one handler to completion."""
    return asyncio.run(coro)


def refusal(model):
    """The most recent `command_error` broadcast, or None if there is none.

    A refusal is not an exception here: a handler reports failure by
    broadcasting `command_error` and returning without a value, so that clients
    on every SDK generation can see it.
    """
    errors = [
        message for message in model.sent if type(message).__name__ == "CommandError"
    ]
    return errors[-1] if errors else None


@pytest.fixture
def model():
    """A FastH3 with the attributes ``load()`` would have set, and no engine."""
    instance = FastH3()
    # Loop-bound state the runtime creates when the model loop starts.
    instance._on_loop_ready()
    # The handful of values load() reads out of fasth3.yaml.
    instance.default_aspect = "16:9"
    instance.default_clip_frames = clip_plan.frames_for_seconds(clip_plan.MAX_SECONDS)
    instance.ramp_frames = (clip_plan.MIN_FRAMES,)
    instance.default_seed = 1000
    instance.default_style_prompt = DEFAULT_STYLE_PROMPT
    instance.num_inference_steps = 5
    instance.default_auto_story_enabled = True
    instance.story_start_delay = 20.0
    instance.story_queue_target = 2
    instance.story_history_size = 7
    instance._story_writer = None
    instance.live_chat_enabled = False
    instance.live_chat_room_id = 0
    instance.live_chat_prefix = "!Prompt:"
    instance.live_chat_max_request_chars = 200
    instance.live_chat_max_pending = 10

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, value, *, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": value.split()}

        def decode(self, token_ids, *, skip_special_tokens=False):
            del skip_special_tokens
            return " ".join(str(token_id) for token_id in token_ids)

    instance._prompt_tokenizer = Tokenizer()
    instance._reset_session_state()

    sent: list = []

    async def capture(message):
        sent.append(message)

    instance.send = capture
    instance.sent = sent
    return instance


def test_start_needs_a_prompt(model):
    """A fresh session has no prompt, so the channel waits for the client."""
    assert model._active_prompt == ""
    assert list(model._prompt_queue) == []
    assert run(model.start()) is None
    assert refusal(model).command == "start"
    assert model._started is False


def test_start_arms_the_channel(model):
    run(model.set_prompt(prompt="a lighthouse in fog"))
    run(model.start())
    assert model._started is True


def test_start_is_refused_while_a_channel_runs(model):
    model._running = True
    assert run(model.start()) is None
    assert refusal(model).command == "start"


def test_an_empty_prompt_clears_the_condition(model):
    run(model.set_prompt(prompt="a lighthouse in fog"))
    reply = run(model.set_prompt(prompt="   "))
    # "Unset" is null on the wire, never an empty string — the same convention
    # `state_update.prompt` uses, so a client reads one shape from both.
    assert reply.prompt is None
    assert model._active_prompt == ""
    assert list(model._prompt_queue) == []
    # And `start` goes back to being refused.
    assert run(model.start()) is None
    assert refusal(model).command == "start"


def test_an_overlong_tokenized_prompt_is_refused_without_entering_the_queue(model):
    reply = run(model.set_prompt(prompt=" ".join(["token"] * 257)))

    assert reply is None
    assert refusal(model).command == "set_prompt"
    assert "at most 256" in refusal(model).reason
    assert list(model._prompt_queue) == []


def test_story_prompt_normalization_keeps_only_the_three_model_fields(model):
    completion = """Here is the next scene:
integrated_multimodal_description: SpongeBob opens a letter.
overall_soundscape: Paper rustles.
non_diegetic_music: A playful sting.
integrated_multimodal_description: This accidental second scene is ignored.
```"""

    assert model._normalize_story_prompt(completion) == (
        "integrated_multimodal_description: SpongeBob opens a letter.\n"
        "overall_soundscape: Paper rustles.\n"
        "non_diegetic_music: A playful sting."
    )


def test_story_prompt_normalization_fits_verbose_completions(model):
    completion = (
        "integrated_multimodal_description: "
        + "SpongeBob and Patrick follow the glowing map through Bikini Bottom. " * 7
        + "\noverall_soundscape: "
        + "Clear English dialogue, bubbles, footsteps, shell hum, and ocean ambience. "
        * 3
        + "\nnon_diegetic_music: "
        + "Playful nautical cartoon music with an unresolved magical transition. " * 3
    )

    prompt = model._normalize_story_prompt(completion)

    assert len(prompt) <= MAX_GENERATED_PROMPT_CHARS
    assert prompt.count("integrated_multimodal_description:") == 1
    assert prompt.count("overall_soundscape:") == 1
    assert prompt.count("non_diegetic_music:") == 1
    assert "revealing a dimly." not in prompt


def test_story_context_uses_the_latest_assigned_and_waiting_scenes(model):
    model._story_history.extend([f"assigned {index}" for index in range(4)])
    model._prompt_queue.extend(["waiting 4", "waiting 5"])

    assert model._story_context() == [
        "assigned 0",
        "assigned 1",
        "assigned 2",
        "assigned 3",
        "waiting 4",
        "waiting 5",
    ]


def test_story_context_removes_exact_duplicates(model):
    model._story_history.extend(["opening", "treasure map", "opening"])
    model._prompt_queue.extend(["treasure map", "new clue"])

    assert model._story_context() == ["opening", "treasure map", "new clue"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("!Prompt: SpongeBob opens a glowing door", "SpongeBob opens a glowing door"),
        (" !prompt：派大星找到一张藏宝图 ", "派大星找到一张藏宝图"),
        (
            ":\r!Prompt: SpongeBob says Reactor is good",
            "SpongeBob says Reactor is good",
        ),
        (
            "viewer says !Prompt: SpongeBob visits Sandy's dome",
            "SpongeBob visits Sandy's dome",
        ),
        ("ordinary chat", None),
        ("!Prompt:", None),
    ],
)
def test_live_prompt_parser_accepts_only_explicit_commands(message, expected):
    request = parse_live_prompt(
        message,
        viewer_name="viewer",
        message_id="message-1",
        prefix="!Prompt:",
        max_chars=200,
    )

    assert (request.request if request else None) == expected


def test_live_prompt_parser_rejects_oversized_requests():
    request = parse_live_prompt(
        "!Prompt: " + "x" * 201,
        viewer_name="viewer",
        message_id="message-1",
        prefix="!Prompt:",
        max_chars=200,
    )

    assert request is None


def test_story_plot_removes_audio_fields_from_writer_context():
    scene = (
        "integrated_multimodal_description: SpongeBob follows a glowing shell.\n"
        "overall_soundscape: Bubbles and footsteps.\n"
        "non_diegetic_music: A curious motif."
    )

    assert _story_plot(scene) == "SpongeBob follows a glowing shell."


def test_story_prompt_rejects_rewrites_but_allows_new_events(model):
    scene = (
        "integrated_multimodal_description: SpongeBob opens a glowing shell box.\n"
        "overall_soundscape: The shell hums.\n"
        "non_diegetic_music: A mysterious chord."
    )
    rewrite = scene.replace("opens", "carefully opens")
    continuation = (
        "integrated_multimodal_description: A map rises from the shell and points "
        "SpongeBob toward a dark reef.\n"
        "overall_soundscape: Paper flutters through bubbles.\n"
        "non_diegetic_music: A quick nautical rhythm."
    )

    assert model._story_prompt_repeats(rewrite, [scene]) is True
    assert model._story_prompt_repeats(continuation, [scene]) is False


def test_story_prompt_rejects_a_copied_scene_with_a_short_extension(model):
    shared_action = (
        "At sunset in Jellyfish Fields, three jellyfish float nearby while the "
        "friends dance in separate positions. A wide camera slowly arcs. "
        "SpongeBob whispers that they should move slowly and Patrick laughs."
    )
    scene = (
        "integrated_multimodal_description: Exactly one SpongeBob SquarePants and "
        f"exactly one Patrick Star appear. {shared_action}\n"
        "overall_soundscape: Bubbles and laughter.\n"
        "non_diegetic_music: Playful strings."
    )
    copied = (
        f"integrated_multimodal_description: {shared_action} A stranger arrives "
        "with a glowing crystal.\n"
        "overall_soundscape: Bubbles and a magical chime.\n"
        "non_diegetic_music: Mysterious strings."
    )

    assert model._story_prompt_repeats(copied, [scene]) is True


def test_spongebob_bible_restores_aliases_and_character_anchors(model):
    prompt = model._normalize_story_prompt(
        "integrated_multimodal_description: S1 opens the door while S2 waves.\n"
        "overall_soundscape: A hinge squeaks.\n"
        "non_diegetic_music: Playful brass."
    )

    restored = model._apply_story_bible(prompt, SPONGEBOB_STORY_BIBLE)

    assert "S1" not in restored and "S2" not in restored
    assert "SpongeBob SquarePants" in restored
    assert "Patrick Star" in restored
    assert "small yellow square sea sponge" in restored
    assert "pink starfish" in restored


def test_human_spongebob_prompt_sets_a_persistent_story_bible(model):
    run(model.set_prompt(prompt="SpongeBob and Patrick Star enter the kitchen."))
    run(model.set_prompt(prompt="S1 flips a patty while S2 waits."))

    assert model._story_bible == SPONGEBOB_STORY_BIBLE


def test_openrouter_writer_requests_and_decodes_a_structured_scene(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        content = json.dumps(
            {
                "integrated_multimodal_description": "The shell opens a door.",
                "overall_soundscape": "Stone grinds and bubbles rise.",
                "non_diegetic_music": "A curious marimba motif.",
            }
        )
        response = {"choices": [{"message": {"content": content}}]}
        return Response(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    writer = _StoryWriter(
        model="openai/gpt-5.4-mini",
        endpoint="https://openrouter.example/chat/completions",
        api_key_env="TEST_OPENROUTER_KEY",
        api_key_file=tmp_path / "missing-key",
        max_tokens=450,
        reasoning_effort="minimal",
        timeout_seconds=45,
    )

    scene = writer.generate(
        [
            (
                "integrated_multimodal_description: SpongeBob finds a shell.\n"
                "overall_soundscape: Bubbles.\n"
                "non_diegetic_music: Playful strings."
            )
        ],
        story_bible=SPONGEBOB_STORY_BIBLE,
        viewer_request="Patrick Star discovers a singing jellyfish.",
    )

    request = captured["request"]
    payload = json.loads(request.data)
    assert captured["timeout"] == 45
    assert payload["model"] == "openai/gpt-5.4-mini"
    assert payload["reasoning"] == {"effort": "minimal", "exclude": True}
    assert payload["response_format"]["type"] == "json_schema"
    assert (
        "entirely in English regardless of the input language"
        in payload["messages"][0]["content"]
    )
    assert "Bubbles." not in payload["messages"][1]["content"]
    assert "<viewer_request>" in payload["messages"][1]["content"]
    assert (
        "Patrick Star discovers a singing jellyfish."
        in payload["messages"][1]["content"]
    )
    assert "Required progression:" not in payload["messages"][1]["content"]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert scene == (
        "integrated_multimodal_description: The shell opens a door.\n"
        "overall_soundscape: Stone grinds and bubbles rise.\n"
        "non_diegetic_music: A curious marimba motif."
    )


def test_repeated_playback_counts_as_one_story_scene(model):
    model._active_prompt = "opening scene"

    assert model._take_prompt() == "opening scene"
    assert model._take_prompt() == "opening scene"
    assert list(model._story_history) == ["opening scene"]


def test_spongebob_characters_are_anchored_for_independent_clips(model):
    prompt = model._normalize_story_prompt(
        "integrated_multimodal_description: SpongeBob and Patrick open a map.\n"
        "overall_soundscape: Paper rustle.\n"
        "non_diegetic_music: Playful music."
    )

    anchored = model._anchor_spongebob_characters(prompt)
    assert "exactly one spongebob squarepants" in anchored.casefold()
    assert "exactly one Patrick Star" in anchored
    assert "SpongeBob SquarePants, a small yellow square sea sponge" in anchored
    assert "Patrick Star, a pink starfish" in anchored
    assert len(anchored) <= MAX_GENERATED_PROMPT_CHARS


def test_spongebob_character_count_is_explicit_when_appearances_exist(model):
    prompt = model._normalize_story_prompt(
        "integrated_multimodal_description: SpongeBob SquarePants, a small yellow "
        "square sea sponge, and Patrick Star, a pink starfish, open a map.\n"
        "overall_soundscape: Paper rustle.\n"
        "non_diegetic_music: Playful music."
    )

    anchored = model._anchor_spongebob_characters(prompt)
    assert "exactly one spongebob squarepants" in anchored.casefold()
    assert "exactly one Patrick Star" in anchored


def test_spongebob_anchors_fit_a_nearly_full_prompt(model):
    prompt = model._normalize_story_prompt(
        "integrated_multimodal_description: SpongeBob and Patrick follow a glowing "
        + "map through a twisting reef corridor. " * 8
        + "\noverall_soundscape: Clear English dialogue, bubbles, and footsteps."
        + "\nnon_diegetic_music: Playful nautical music."
    )

    anchored = model._anchor_spongebob_characters(prompt)
    assert len(anchored) <= MAX_GENERATED_PROMPT_CHARS
    assert "Exactly one SpongeBob SquarePants" in anchored
    assert "exactly one Patrick Star" in anchored


def test_auto_story_can_be_disabled_and_restored(model):
    reply = run(model.set_auto_story(enabled=False))
    assert reply.enabled is False
    assert model._auto_story_enabled is False

    reply = run(model.set_auto_story(enabled=True))
    assert reply.enabled is True
    assert model._auto_story_enabled is True


def test_auto_story_retries_bad_output_without_queuing_a_fixed_fallback(model):
    class Writer:
        def __init__(self):
            self.calls = 0

        def generate(
            self,
            recent_scenes,
            *,
            story_bible="",
            viewer_request=None,
            retry=False,
        ):
            assert recent_scenes == ["opening scene"]
            assert story_bible == ""
            assert viewer_request is None
            self.calls += 1
            if self.calls <= 2:
                return "not a FastH3 prompt"
            return (
                "integrated_multimodal_description: A new door opens beyond the "
                "completed scene and the cast follows a moving light through it.\n"
                "overall_soundscape: Footsteps and a resonant door hinge.\n"
                "non_diegetic_music: A curious rising motif."
            )

    async def exercise():
        writer = Writer()
        model._story_writer = writer
        model.story_start_delay = 0
        model.story_queue_target = 1
        model._active_prompt = "opening scene"
        model._story_history.append("opening scene")
        model._started = True
        model._channel_started_at = time.monotonic()
        model.connected.set()
        task = asyncio.create_task(model._auto_story_loop())
        for _ in range(300):
            if model._fallback_prompt_queue:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return writer

    writer = run(exercise())
    assert writer.calls == 3
    assert len(model._fallback_prompt_queue) == 1
    assert "A new door opens" in model._fallback_prompt_queue[0]
    queued = [
        message
        for message in model.sent
        if type(message).__name__ == "AutoPromptQueued"
    ]
    assert queued[-1].fallback_used is False


def test_live_request_is_rewritten_before_automatic_fallbacks(model):
    class Writer:
        def __init__(self):
            self.requests = []

        def generate(
            self,
            recent_scenes,
            *,
            story_bible="",
            viewer_request=None,
            retry=False,
        ):
            del recent_scenes, story_bible, retry
            self.requests.append(viewer_request)
            return (
                "integrated_multimodal_description: A singing jellyfish enters "
                "the next room and points toward a newly opened tunnel.\n"
                "overall_soundscape: Bubbles, a bright hum, and a surprised laugh.\n"
                "non_diegetic_music: A quick curious marimba motif."
            )

    async def exercise():
        writer = Writer()
        model._story_writer = writer
        model._fallback_prompt_queue.extend(
            ["obsolete fallback one", "obsolete fallback two"]
        )
        model.connected.set()
        model._on_live_prompt(
            LivePromptRequest(
                request="a singing jellyfish reveals a tunnel",
                viewer_name="viewer",
                message_id="message-1",
            )
        )
        for _ in range(300):
            if model._prompt_queue:
                break
            await asyncio.sleep(0.01)
        if model._story_task is not None:
            await model._story_task
        await asyncio.sleep(0)
        return writer

    writer = run(exercise())

    assert writer.requests == ["a singing jellyfish reveals a tunnel"]
    assert not model._fallback_prompt_queue
    assert "singing jellyfish" in model._prompt_queue[0]
    assert not model._live_requests
    assert any(type(message).__name__ == "LivePromptReceived" for message in model.sent)
    assert any(type(message).__name__ == "LivePromptQueued" for message in model.sent)


def test_live_requests_are_rewritten_in_arrival_order(model):
    class Writer:
        def __init__(self):
            self.requests = []

        def generate(
            self,
            recent_scenes,
            *,
            story_bible="",
            viewer_request=None,
            retry=False,
        ):
            del recent_scenes, story_bible, retry
            self.requests.append(viewer_request)
            descriptions = {
                "open a treasure chest": (
                    "A brass treasure chest rises from a coral pit as the cast "
                    "pulls its seaweed-covered lid open and follows a blue beam "
                    "toward a map hidden inside."
                ),
                "ride a bubble bus": (
                    "A transparent bubble bus sweeps around a bright reef, opens "
                    "its round door, and carries the cast past a school of silver "
                    "fish toward a distant station."
                ),
            }
            return (
                f"integrated_multimodal_description: {descriptions[viewer_request]}\n"
                "overall_soundscape: Clear movement effects, bubbles, and dialogue.\n"
                "non_diegetic_music: A playful forward-moving cartoon cue."
            )

    async def exercise():
        writer = Writer()
        model._story_writer = writer
        model.connected.set()
        for index, request in enumerate(
            ("open a treasure chest", "ride a bubble bus"), start=1
        ):
            model._on_live_prompt(
                LivePromptRequest(
                    request=request,
                    viewer_name=f"viewer-{index}",
                    message_id=f"message-{index}",
                )
            )
        for _ in range(300):
            if len(model._prompt_queue) == 2:
                break
            await asyncio.sleep(0.01)
        if model._story_task is not None:
            await model._story_task
        await asyncio.sleep(0)
        return writer

    writer = run(exercise())

    assert writer.requests == ["open a treasure chest", "ride a bubble bus"]
    assert "treasure chest" in model._prompt_queue[0]
    assert "bubble bus" in model._prompt_queue[1]


def test_live_backlog_limit_counts_raw_and_rewritten_requests(model):
    model.live_chat_max_pending = 2
    model._prompt_queue.append("an already rewritten viewer scene")
    model._prompt_origins.append(
        _PromptOrigin(
            source="bilibili",
            viewer_name="first viewer",
            original_request="first request",
        )
    )

    async def exercise():
        model._on_live_prompt(
            LivePromptRequest(
                request="second request",
                viewer_name="second viewer",
                message_id="message-2",
            )
        )
        model._on_live_prompt(
            LivePromptRequest(
                request="third request",
                viewer_name="third viewer",
                message_id="message-3",
            )
        )
        await asyncio.sleep(0)

    run(exercise())

    assert model._live_prompt_backlog() == 2
    assert [request.request for request in model._live_requests] == ["second request"]


def test_live_request_uses_a_valid_local_rewrite_when_openrouter_fails(model):
    class Writer:
        def generate(
            self,
            recent_scenes,
            *,
            story_bible="",
            viewer_request=None,
            retry=False,
        ):
            del recent_scenes, story_bible, viewer_request, retry
            raise RuntimeError("temporary upstream failure")

    async def exercise():
        model._story_writer = Writer()
        model.connected.set()
        model._on_live_prompt(
            LivePromptRequest(
                request="a jellyfish parade crosses the street",
                viewer_name="viewer",
                message_id="message-1",
            )
        )
        for _ in range(300):
            if model._prompt_queue:
                break
            await asyncio.sleep(0.01)
        if model._story_task is not None:
            await model._story_task
        await asyncio.sleep(0)

    run(exercise())

    prompt = model._prompt_queue[0]
    assert "jellyfish parade crosses the street" in prompt
    assert all(field in prompt for field in STORY_FIELDS)
    queued = [
        message
        for message in model.sent
        if type(message).__name__ == "LivePromptQueued"
    ]
    assert queued[-1].fallback_used is True


def test_complete_viewer_prompts_precede_automatic_fallbacks(model):
    model._fallback_prompt_queue.append("automatic fallback")
    model._prompt_queue.append("viewer-directed scene")

    assert model._take_prompt() == "viewer-directed scene"
    assert model._take_prompt() == "automatic fallback"


def test_style_is_applied_without_changing_the_client_prompt(model):
    reply = run(model.set_style(style_prompt="flat 2D animation"))
    prompt_reply = run(model.set_prompt(prompt="SpongeBob walks into the Krusty Krab"))

    assert reply.style_prompt == "flat 2D animation"
    assert prompt_reply.prompt == "SpongeBob walks into the Krusty Krab"
    assert model._compose_prompt(prompt_reply.prompt, model._style_prompt) == (
        "SpongeBob walks into the Krusty Krab\nVisual style: flat 2D animation"
    )


def test_blank_style_disables_the_shared_style_instruction(model):
    reply = run(model.set_style(style_prompt="   "))

    assert reply.style_prompt == ""
    assert model._style_prompt == ""
    assert model._compose_prompt("a lighthouse", model._style_prompt) == "a lighthouse"


def test_style_set_while_idle_applies_to_the_first_clip(model):
    reply = run(model.set_style(style_prompt="stop-motion clay animation"))

    assert reply.effective_clip_index == 0
    assert reply.effective_in_seconds == 0.0


def test_style_rejects_a_combination_that_exceeds_the_token_budget(model):
    run(model.set_style(style_prompt=""))
    run(model.set_prompt(prompt=" ".join(["subject"] * 225)))
    reply = run(model.set_style(style_prompt=" ".join(["style"] * 35)))

    assert reply is None
    assert refusal(model).command == "set_style"
    assert model._style_prompt == ""


def test_clip_length_snaps_to_something_generatable(model):
    reply = run(model.set_clip_seconds(seconds=8.3))
    assert reply.frames % 17 == 5
    assert reply.clip_seconds == pytest.approx(reply.frames / 24, abs=1e-3)
    assert model._clip_frames == reply.frames


def test_the_canvas_is_locked_once_a_channel_runs(model):
    reply = run(model.set_canvas(aspect="9:16"))
    assert (reply.height, reply.width) == clip_plan.canvas_for_choice("9:16")
    model._running = True
    assert run(model.set_canvas(aspect="1:1")) is None
    assert refusal(model).command == "set_canvas"
    # The refused command had no effect.
    assert model._aspect == "9:16"


def test_pause_and_resume_only_apply_to_a_live_channel(model):
    assert run(model.pause()) is None
    assert refusal(model).command == "pause"

    model._running = True
    assert run(model.pause()) is not None  # accepted: a real reply comes back
    assert model._paused is True
    assert run(model.pause()) is None
    assert refusal(model).command == "pause"

    assert run(model.resume()) is not None
    assert model._paused is False
    assert run(model.resume()) is None
    assert refusal(model).command == "resume"


def test_stop_keeps_the_conditions(model):
    run(model.set_prompt(prompt="a lighthouse in fog"))
    run(model.set_clip_seconds(seconds=10.0))
    model._running = True
    run(model.stop_channel())
    assert model._started is False
    assert model._do_reset is True
    assert model._stop_only is True
    assert list(model._prompt_queue) == ["a lighthouse in fog"]
    assert model._clip_frames == clip_plan.frames_for_seconds(10.0)


def test_reset_restores_every_default(model):
    run(model.set_prompt(prompt="a lighthouse in fog"))
    run(model.set_clip_seconds(seconds=10.0))
    run(model.set_seed(seed=7))
    run(model.set_canvas(aspect="1:1"))

    reply = run(model.reset())
    assert reply.was_running is False
    assert model._active_prompt == ""
    assert list(model._prompt_queue) == []
    assert model._clip_frames == model.default_clip_frames
    assert model._seed == model.default_seed
    assert model._aspect == model.default_aspect
    assert model._style_prompt == model.default_style_prompt
    assert model._current_prompt == ""
    assert model._next_prompt == ""


def test_reset_while_idle_does_not_wedge_the_arm_loop(model):
    """`_do_reset` left set with nothing running would block every later start.

    ``_wait_until_armed`` deliberately does not read the flag, and ``reset``
    only raises it for a channel that is actually live. Both halves are asserted
    here because either alone would still hang the model loop.
    """
    run(model.reset())
    assert model._do_reset is False

    run(model.set_prompt(prompt="a lighthouse in fog"))
    run(model.start())
    assert model._started and model._ready()


def test_a_prompt_set_while_idle_lands_on_the_first_clip(model):
    reply = run(model.set_prompt(prompt="a lighthouse in fog"))
    assert reply.effective_clip_index == 0
    assert reply.effective_in_seconds == 0.0


def test_prompts_queued_while_idle_take_consecutive_clips(model):
    first = run(model.set_prompt(prompt="a lighthouse in fog"))
    second = run(model.set_prompt(prompt="a neon alley in rain"))

    assert first.queue_position == 0
    assert second.queue_position == 1
    assert second.effective_clip_index == 1
    assert second.effective_in_seconds == pytest.approx(
        clip_plan.MIN_FRAMES / 24, abs=0.01
    )
    assert list(model._prompt_queue) == [
        "a lighthouse in fog",
        "a neon alley in rain",
    ]


def test_a_prompt_set_before_the_first_clip_lands_on_the_second(model):
    """Clip 0 is already being built, so the earliest a change can land is clip 1."""
    model._running = True
    model._clip_index = -1
    reply = run(model.set_prompt(prompt="a lighthouse in fog"))
    assert reply.effective_clip_index == 1
    # The wait is the whole of clip 0 — the ramp's short opening clip.
    assert reply.effective_in_seconds == pytest.approx(
        clip_plan.MIN_FRAMES / 24, abs=0.01
    )


def test_a_prompt_set_mid_channel_lands_two_clips_out(model):
    """Clip k is playing and clip k+1 is already built, so a change lands on k+2."""
    model._running = True
    model._clip_index = 3
    model._current_clip_seconds = 14.375
    model._clip_start_seconds = 40.0
    model._seconds_sent = 44.0  # 4 s into clip 3

    reply = run(model.set_prompt(prompt="a lighthouse in fog"))
    assert reply.effective_clip_index == 5
    # 10.375 s left of clip 3, then the whole of clip 4.
    assert reply.effective_in_seconds == pytest.approx(10.375 + 14.375, abs=0.01)


def test_get_state_reports_the_same_snapshot_that_is_broadcast(model):
    run(model.set_prompt(prompt="a lighthouse in fog"))
    model.sent.clear()
    run(model._send_state_update())
    broadcast = model.sent[-1]
    direct = run(model.get_state())
    assert vars(direct) == vars(broadcast)


def test_the_snapshot_publishes_the_live_command_set(model):
    snapshot = run(model.get_state())
    assert snapshot.ready is False
    assert "start" not in snapshot.valid_commands  # no prompt yet

    run(model.set_prompt(prompt="a lighthouse in fog"))
    snapshot = run(model.get_state())
    assert snapshot.ready is True
    assert "start" in snapshot.valid_commands
    assert (snapshot.height, snapshot.width) == clip_plan.canvas_for_choice("16:9")


def test_the_ramp_shortens_only_the_opening_clip(model):
    assert model._frames_for_clip(0) == clip_plan.MIN_FRAMES
    assert model._frames_for_clip(1) == model.default_clip_frames
    assert model._frames_for_clip(50) == model.default_clip_frames


def test_a_refusal_never_masquerades_as_a_reply(model):
    """A handler must return only the type its annotation names.

    `pause` is annotated `-> ChannelPaused`; a refusal that returned a
    `CommandError` from it would reach the client typed as the message the
    schema promised, with every guaranteed field undefined.
    """
    assert run(model.pause()) is None
    error = refusal(model)
    assert type(error).__name__ == "CommandError"
    assert error.reason


def test_the_runtime_exception_is_never_raised(model):
    """Its correlated failure frame is withheld from v0 clients, so a raise is
    silence for anyone on the 2.x SDK. The broadcast is what reaches them."""
    source = (MODEL_DIR / "fasth3.py").read_text(encoding="utf-8")
    assert "raise CommandError" not in source
    # And the name in scope is the model's own message, not the runtime's.
    from fasth3 import CommandError as in_scope
    from fasth3_types import CommandError as ours

    assert in_scope is ours


# ----------------------------------------------------------------- the channel
#
# ``_run_channel`` is the concurrent part of this model — a worker thread
# building clips, an event loop pacing them out, and abort paths crossing both.
# These tests replace only the clip *builder*, so the real queueing, ordering,
# emission and teardown all run.
#
# Clips are shrunk to a few frames so a whole channel plays in well under a
# second; the pacer is a real 24 fps clock, so the numbers below are wall time.

FRAMES_PER_CLIP = 6  # 0.25 s of content at 24 fps


@pytest.fixture
def channel():
    instance = FastH3()
    instance._on_loop_ready()
    instance.connected.set()
    instance.default_aspect = "16:9"
    instance.default_clip_frames = clip_plan.frames_for_seconds(clip_plan.MAX_SECONDS)
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
    instance._reset_session_state()
    instance._active_prompt = "a lighthouse in fog"
    # `_run_channel` is driven directly here, so arm the session as `start` does.
    instance._started = True

    instance._jobs = queue.Queue()
    instance._worker = threading.Thread(
        target=instance._worker_loop, name="test-generation", daemon=True
    )
    instance._worker.start()

    # Every clip is a handful of tiny frames; shape does not matter here.
    instance._frames_for_clip = lambda index: FRAMES_PER_CLIP

    built: list[tuple[int, str, str, int]] = []

    def fake_generate(index, frames, prompt, style_prompt, seed, height, width):
        built.append((index, prompt, style_prompt, seed))
        video = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(frames)]
        samples = np.zeros((1, round(frames / 24 * 48_000)), dtype=np.int16)
        return video, samples

    instance._generate_clip = fake_generate
    instance.built = built

    emitted: list = []
    timeline: list[str] = []

    async def fake_emit(output):
        emitted.append(output)
        timeline.append("emit")

    instance.emit = fake_emit
    instance.emitted = emitted

    messages: list = []

    async def fake_send(message):
        messages.append(message)
        timeline.append(type(message).__name__)

    instance.send = fake_send
    instance.messages = messages
    instance.timeline = timeline
    return instance


def names(messages) -> list[str]:
    return [type(m).__name__ for m in messages]


def run_channel_until(channel, stop_after_clips: int):
    """Run a channel, stopping it once `stop_after_clips` clips have completed."""

    async def drive():
        async def stopper():
            while channel._clips_sent < stop_after_clips:
                await asyncio.sleep(0.01)
            channel._started = False
            channel._stop_only = True
            channel._do_reset = True

        await asyncio.gather(channel._run_channel(), stopper())

    asyncio.run(drive())


async def armed_within(channel, seconds: float) -> bool:
    """Whether `_wait_until_armed` would arm a channel inside `seconds`."""
    try:
        return await asyncio.wait_for(channel._wait_until_armed(), timeout=seconds)
    except TimeoutError:
        return False


def test_clips_stream_in_order_and_the_channel_ends_cleanly(channel):
    run_channel_until(channel, stop_after_clips=3)

    started = [m for m in channel.messages if type(m).__name__ == "ClipStarted"]
    assert [m.clip_index for m in started[:3]] == [0, 1, 2]
    assert names(channel.messages)[0] == "ChannelStarted"
    assert "ChannelStopped" in names(channel.messages)
    assert channel._running is False


def test_the_next_clip_is_built_while_the_current_one_plays(channel):
    """This is what makes the channel endless; without it every clip is a stall."""
    run_channel_until(channel, stop_after_clips=2)

    # Clip 0 is emitted only after clip 1 has been asked for.
    assert [index for index, _prompt, _style, _seed in channel.built][:3] == [0, 1, 2]
    assert len(channel.built) >= 3


def test_state_distinguishes_the_visible_clip_from_the_lookahead(channel):
    channel._prompt_queue.extend(["first scene", "second scene"])
    run_channel_until(channel, stop_after_clips=1)

    started_at = names(channel.messages).index("ClipStarted")
    started = channel.messages[started_at]
    snapshot = channel.messages[started_at + 1]
    assert type(snapshot).__name__ == "StateUpdate"
    assert started.prompt == "first scene"
    assert started.style_prompt == DEFAULT_STYLE_PROMPT
    assert snapshot.current_prompt == "first scene"
    assert snapshot.current_style_prompt == DEFAULT_STYLE_PROMPT
    assert snapshot.next_prompt == "second scene"
    assert snapshot.next_style_prompt == DEFAULT_STYLE_PROMPT
    assert channel.timeline.index("emit") < channel.timeline.index("ClipStarted")


def test_visible_clip_reports_its_bilibili_origin(channel):
    channel._prompt_queue.append("SpongeBob visits the boating school")
    channel._prompt_origins.append(
        _PromptOrigin(
            source="bilibili",
            viewer_name="NextTryTV",
            original_request="SpongeBob goes back to school",
        )
    )
    run_channel_until(channel, stop_after_clips=1)

    started_at = names(channel.messages).index("ClipStarted")
    started = channel.messages[started_at]
    snapshot = channel.messages[started_at + 1]
    assert started.source == "bilibili"
    assert started.viewer_name == "NextTryTV"
    assert started.original_request == "SpongeBob goes back to school"
    assert snapshot.current_prompt_source == "bilibili"
    assert snapshot.current_prompt_viewer_name == "NextTryTV"
    assert snapshot.current_prompt_original_request == "SpongeBob goes back to school"


def test_every_frame_of_every_clip_reaches_the_wire(channel):
    run_channel_until(channel, stop_after_clips=2)

    frames = sum(output.main_video.shape[0] for output in channel.emitted)
    # Every completed clip went out whole. The stop lands mid-clip, so the tally
    # can exceed that by part of the clip that was interrupted.
    assert frames >= channel._clips_sent * FRAMES_PER_CLIP
    assert channel._clips_sent == 2
    # The counters clients see are derived from what was actually emitted.
    assert channel._frames_sent == frames
    assert channel._seconds_sent == pytest.approx(frames / 24)


def test_video_and_audio_stay_locked_slice_for_slice(channel):
    run_channel_until(channel, stop_after_clips=2)

    for output in channel.emitted:
        video_frames = output.main_video.shape[0]
        audio_samples = output.main_audio.shape[1]
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.dtype == np.int16
        assert output.main_audio.ndim == 2 and output.main_audio.shape[0] == 1
        assert audio_samples == pytest.approx(video_frames * 48_000 / 24, abs=1)


def test_queued_prompts_are_consumed_once_in_order_then_the_last_repeats(channel):
    """Each queued prompt owns one clip, then the last one becomes the fallback."""

    async def drive():
        async def change():
            # Wait until clip 0 is streaming; clip 1 has been submitted by then.
            while channel._clip_index < 0:
                await asyncio.sleep(0.005)
            await channel.set_prompt(prompt="a neon alley in the rain")
            await channel.set_prompt(prompt="a coral reef full of fish")
            await channel.set_prompt(prompt="a rocket crossing the moon")
            while channel._clips_sent < 6:
                await asyncio.sleep(0.01)
            channel._started = False
            channel._stop_only = True
            channel._do_reset = True

        await asyncio.gather(channel._run_channel(), change())

    asyncio.run(drive())

    prompts = [prompt for _index, prompt, _style, _seed in channel.built]
    assert prompts[0] == "a lighthouse in fog"
    assert prompts[1] == "a lighthouse in fog"  # already in flight when it changed
    assert prompts[:7] == [
        "a lighthouse in fog",
        "a lighthouse in fog",  # already in flight when the queue changed
        "a neon alley in the rain",
        "a coral reef full of fish",
        "a rocket crossing the moon",
        "a rocket crossing the moon",
        "a rocket crossing the moon",
    ]


def test_each_clip_advances_the_seed(channel):
    run_channel_until(channel, stop_after_clips=2)
    seeds = [seed for _index, _prompt, _style, seed in channel.built]
    assert seeds[:3] == [1000, 1001, 1002]


def test_a_lost_audience_winds_the_channel_down(channel):
    """No client means no GPU spend; the channel stops at the next boundary."""

    async def drive():
        async def leave():
            while channel._clips_sent < 1:
                await asyncio.sleep(0.01)
            channel.connected.clear()

        await asyncio.gather(channel._run_channel(), leave())

    asyncio.run(drive())

    assert channel._running is False
    # A lost audience is not a `stop`, so no channel_stopped is sent.
    assert "ChannelStopped" not in names(channel.messages)
    # The conditions survive, so reconnecting resumes the same channel.
    assert channel._active_prompt == "a lighthouse in fog"
    assert channel._started is True


def test_a_failing_clip_reports_and_returns_the_model_to_idle(channel):
    def explode(index, frames, prompt, style_prompt, seed, height, width):
        raise RuntimeError("the engine fell over")

    channel._generate_clip = explode
    asyncio.run(channel._run_channel())

    failures = [m for m in channel.messages if type(m).__name__ == "ChannelFailed"]
    assert len(failures) == 1
    assert "the engine fell over" in failures[0].reason
    assert channel._running is False


def test_pause_holds_the_stream_and_resume_continues_it(channel):
    async def drive():
        async def hold():
            while not channel.emitted:
                await asyncio.sleep(0.005)
            channel._paused = True
            # The pause is read once per slice, and a slice already waiting on
            # the pacer is committed to going out — so it takes effect within
            # one slice interval (EMIT_FRAMES / 24, about 0.125 s), not
            # instantly. Wait past that before measuring.
            await asyncio.sleep(0.25)
            held = len(channel.emitted)
            await asyncio.sleep(0.3)
            assert len(channel.emitted) == held, "the stream moved while paused"
            channel._paused = False
            while channel._clips_sent < 1:
                await asyncio.sleep(0.01)
            channel._started = False
            channel._stop_only = True
            channel._do_reset = True

        await asyncio.gather(channel._run_channel(), hold())

    asyncio.run(drive())
    assert sum(o.main_video.shape[0] for o in channel.emitted) >= FRAMES_PER_CLIP


def test_the_pacer_holds_24_fps_across_a_clip_boundary(channel):
    """A seam must not cost time: the clock carries from one clip to the next."""
    started = time.monotonic()
    run_channel_until(channel, stop_after_clips=3)
    elapsed = time.monotonic() - started

    content = 3 * FRAMES_PER_CLIP / 24
    # A slice is handed over at the instant its content is due, so the run ends
    # one slice before the content it queued finishes playing.
    expected = content - EMIT_FRAMES / 24
    # Real-time, not faster: the metronome is what keeps the audio in sync.
    assert elapsed >= expected * 0.95
    # And no seam stall: each clip was built long before it was needed, so three
    # of them must not take appreciably longer than their own content.
    assert elapsed < content + 0.3


def test_a_failed_channel_does_not_restart_itself(channel):
    """`run()` re-arms from `_started`; leaving it set is an eight-GPU crash loop."""

    def explode(index, frames, prompt, style_prompt, seed, height, width):
        raise RuntimeError("the engine fell over")

    channel._generate_clip = explode
    assert channel._started is True
    asyncio.run(channel._run_channel())

    assert channel._started is False
    # And the arm loop agrees: it would not start another channel unattended.
    assert asyncio.run(armed_within(channel, seconds=0.2)) is False


# ---------------------------------------------------------- published contract
#
# ``reactor schema`` compiles this document out of the model class, and
# generated SDKs are built from nothing else. A change here is a change to every
# client, so these tests exist to make an accidental one fail loudly and a
# deliberate one obvious in review — together with the version bump it needs.

# Every command a client can send, and the message each answers with. `None`
# means the command is answered with a bare acknowledgement.
EXPECTED_COMMANDS = {
    "get_state": "StateUpdate",
    "pause": "ChannelPaused",
    "reset": "ChannelReset",
    "resume": "ChannelResumed",
    "set_canvas": "CanvasAccepted",
    "set_clip_seconds": "ClipLengthAccepted",
    "set_prompt": "PromptAccepted",
    "set_auto_story": "AutoStoryAccepted",
    "set_style": "StyleAccepted",
    "set_seed": "SeedAccepted",
    "start": None,
    "stop": None,
}

EXPECTED_MESSAGES = {
    "auto_prompt_queued",
    "auto_story_accepted",
    "canvas_accepted",
    "channel_failed",
    "channel_paused",
    "channel_reset",
    "channel_resumed",
    "channel_started",
    "channel_stopped",
    "clip_complete",
    "clip_length_accepted",
    "clip_started",
    "command_error",
    "live_chat_status",
    "live_prompt_queued",
    "live_prompt_received",
    "prompt_accepted",
    "seed_accepted",
    "state_update",
    "style_accepted",
}

# Commands that can be refused. Each one has to say so in its own summary, and
# name the message a client will actually receive.
EXPECTED_REJECTIONS = (
    "set_prompt",
    "set_style",
    "set_canvas",
    "start",
    "pause",
    "resume",
    "stop",
)


@pytest.fixture(scope="module")
def schema():
    """Render the document exactly as the release pipeline does.

    The renderer imports the model without loading it, so this needs no weights
    and no GPU — but it does need the model's own imports to stay light, which
    is itself part of what this asserts.
    """
    from reactor_runtime.schema import render

    return render(MODEL_DIR, version="v0.4.0")


def test_the_model_publishes_two_outbound_tracks(schema):
    tracks = schema["x-reactor"]["tracks"]
    assert [(t["name"], t["kind"], t["direction"]) for t in tracks] == [
        ("main_video", "video", "out"),
        ("main_audio", "audio", "out"),
    ]


def test_the_command_set_is_exactly_what_clients_expect(schema):
    published = {path.removeprefix("/events/") for path in schema["paths"]}
    assert published == set(EXPECTED_COMMANDS)


def test_every_command_answers_with_the_type_it_promises(schema):
    for name, message in EXPECTED_COMMANDS.items():
        operation = schema["paths"][f"/events/{name}"]["post"]
        responses = operation["responses"]
        if message is None:
            # A handler that returns nothing is answered with a bare 202, so an
            # awaiting client still resolves — it just learns nothing.
            assert set(responses) == {"202"}, name
            continue
        body = responses["200"]["content"]["application/json"]["schema"]
        assert body["$ref"] == f"#/components/schemas/{message}", name


def test_every_message_is_published_once(schema):
    assert set(schema["webhooks"]) == EXPECTED_MESSAGES
    for message in EXPECTED_COMMANDS.values():
        if message is not None:
            assert message in schema["components"]["schemas"]


def test_the_clip_length_bounds_a_client_reads_are_generatable(schema):
    seconds = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]["properties"]["seconds"]
    assert seconds["minimum"] == clip_plan.MIN_SECONDS_PUBLISHED
    assert seconds["maximum"] == clip_plan.MAX_SECONDS_PUBLISHED
    for bound in (seconds["minimum"], seconds["maximum"]):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_the_canvas_choices_are_published_as_an_enum(schema):
    aspect = schema["paths"]["/events/set_canvas"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["aspect"]
    assert aspect["enum"] == list(clip_plan.ASPECT_CHOICES)


def test_every_client_facing_string_is_documented(schema):
    """A frontend developer who cannot read this repo works from these alone."""
    for path, operations in schema["paths"].items():
        assert operations["post"].get("summary"), f"{path} has no description"
        body = operations["post"].get("requestBody")
        if not body:
            continue
        properties = body["content"]["application/json"]["schema"]["properties"]
        for name, field in properties.items():
            assert field.get("description"), (
                f"{path} parameter {name} has no description"
            )
    for name, message in schema["webhooks"].items():
        assert message["post"].get("summary"), f"message {name} has no description"
    # Only the messages this model declares. The runtime contributes its own
    # components (the upload reference, for one), and those are not ours to
    # document.
    ours = {
        message["post"]["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ].rsplit("/", 1)[-1]
        for message in schema["webhooks"].values()
    }
    assert ours, "no message definitions were resolved from the webhooks"
    for name in ours:
        definition = schema["components"]["schemas"][name]
        for field_name, field in definition.get("properties", {}).items():
            assert field.get("description"), f"{name}.{field_name} has no description"


def test_every_message_summary_says_when_it_is_emitted(schema):
    """The house style: a message docstring opens with "Emitted ..."."""
    for name, message in schema["webhooks"].items():
        summary = message["post"]["summary"]
        assert summary.startswith("Emitted "), f"{name}: {summary!r}"


def test_no_message_name_repeats_the_model_name(schema):
    """The SDK client already identifies the model; a prefix is dead weight."""
    for name in schema["webhooks"]:
        assert not name.startswith("fasth3"), name
    for name in schema["components"]["schemas"]:
        assert not name.lower().startswith("fasth3"), name


def test_every_command_summary_names_what_it_emits(schema):
    for name in EXPECTED_COMMANDS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        if name == "get_state":
            # A pure read: it answers, it does not emit.
            assert "state_update" in summary
            continue
        assert "`state_update`" in summary, (
            f"{name} does not say it broadcasts a snapshot"
        )


def test_every_refusable_command_documents_its_failure(schema):
    for name in EXPECTED_REJECTIONS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        assert "`command_error`" in summary, f"{name} does not document its failure"


def test_a_prompt_that_is_not_set_is_null_not_empty(schema):
    """`None` is the unset value on the wire; an empty string would be ambiguous."""
    for message in ("StateUpdate", "PromptAccepted"):
        prompt = schema["components"]["schemas"][message]["properties"]["prompt"]
        types = prompt.get("anyOf") or [prompt]
        assert any(entry.get("type") == "null" for entry in types), (
            f"{message}.prompt: {prompt}"
        )


def test_clip_length_prose_matches_the_published_bounds(schema):
    """The numbers in the description are generated from the same constants."""
    summary = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]["properties"]["seconds"]["description"]
    assert f"{clip_plan.MIN_SECONDS_PUBLISHED:g}" in summary
    assert f"{clip_plan.MAX_SECONDS_PUBLISHED:g}" in summary


# ------------------------------------------------------------------- manifest
#
# The workspace rules from GUIDELINES.md, checked here rather than discovered
# during a build.

# A stray checkpoint in the folder is a large, slow mistake to discover later.
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text(encoding="utf-8"))


def test_the_model_name_matches_the_folder(manifest):
    """The folder is the workspace, and the name is the published slug."""
    assert manifest["model"]["name"] == MODEL_DIR.name


def test_the_version_is_semver_with_a_v_prefix(manifest):
    version = manifest["model"]["version"]
    assert isinstance(version, str), "quote the version so it is not parsed as a number"
    assert re.fullmatch(r"v\d+\.\d+\.\d+", version), (
        f"{version!r} must be semver with a `v`"
    )


def test_the_manifest_carries_a_complete_resource_spec(manifest):
    resources = manifest["model"]["resources"]
    assert resources["gpu"]["type"] and resources["gpu"]["count"] >= 1
    assert resources["cpu"]["request"] and resources["cpu"]["limit"]
    assert resources["memory"]["request"] and resources["memory"]["limit"]


def test_the_image_is_built_from_the_manifest_not_a_dockerfile(manifest):
    """`reactor build` owns the image; a Dockerfile here would be ignored."""
    assert not (MODEL_DIR / "Dockerfile").exists()
    build = manifest["build"]
    assert build["python_requirements"] == "requirements.txt"
    assert (MODEL_DIR / build["python_requirements"]).is_file()


def test_the_config_the_runtime_hands_to_load_exists(manifest):
    config = manifest["runtime"]["config"]
    assert (MODEL_DIR / config).is_file(), (
        f"runtime.config points at a missing {config}"
    )


def test_the_config_and_schema_share_the_default_style(manifest, schema):
    document = yaml.safe_load(
        (MODEL_DIR / manifest["runtime"]["config"]).read_text(encoding="utf-8")
    )
    published = schema["paths"]["/events/set_style"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["style_prompt"]["default"]
    assert document["inference"]["style_prompt"] == DEFAULT_STYLE_PROMPT
    assert published == DEFAULT_STYLE_PROMPT


def test_the_runtime_pin_matches_the_rest_of_the_repo(manifest):
    """Every model here pins the same Reactor Runtime release."""
    assert manifest["build"]["runtime_version"] == "3.5.0"


def test_the_runtime_release_is_pinned_once(manifest):
    """`build.runtime_version` owns it; a second pin would let the two drift."""
    assert "reactor-runtime" not in (MODEL_DIR / "requirements.txt").read_text(
        encoding="utf-8"
    )


def test_the_runtime_import_resolves_to_the_model_class(manifest):
    module_name, _, class_name = manifest["runtime"]["import"].partition(":")
    module = __import__(module_name)
    assert getattr(module, class_name, None) is not None, (
        f"{module_name}.py has no {class_name}"
    )


def test_no_weights_are_committed_alongside_the_model():
    offenders = [
        path.relative_to(MODEL_DIR)
        for path in MODEL_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    ]
    assert not offenders, f"weights never live in git: {offenders}"
