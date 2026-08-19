"""Serve NVIDIA PersonaPlex as a full-duplex speech-to-speech Reactor model.

PersonaPlex is a Moshi-architecture model that encodes the participant's speech
and generates its own at the same time, on one 12.5 Hz token clock. That is what
makes it conversational rather than turn-based: because the agent is producing
audio while it is still hearing, interruptions, overlaps, and fast turn-taking
come out of the model itself. Nothing in this adapter arranges them.

The adapter's job is therefore small and mostly about clocks. Reactor's wire
carries 48 kHz int16 mono; Mimi works at 24 kHz float32. Every 80 ms the adapter
takes one wire frame, converts it down, runs one step of Mimi and the language
model, converts the generated frame back up, and emits it. The mic read is what
paces the loop — it cannot return faster than real time — so no rate limiter is
needed and the agent's speech leaves at exactly the rate the participant's
arrives.

Conditioning is separate from that loop. A voice prompt and a role prompt are
stepped through the language model before a conversation can start, which takes
seconds rather than milliseconds, so it is a distinct phase the adapter enters
on the first step of a session and again whenever a client changes the persona
or the voice.

Upstream's ``server.py`` and ``offline.py`` are the reference for the inference
sequence; where this file departs from them it says so.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import sentencepiece
import torch
from moshi.models import LMGen, loaders

from reactor_runtime import (
    ClientInfo,
    CommandError,
    Idle,
    InputField,
    ReactorPipeline,
    ReadMode,
    connected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.interface.pipeline.idle import _IdleType
from reactor_runtime.log import get_logger

from personaplex_assets import PersonaPlexAssets, prepare_assets, read_config
from personaplex_audio import (
    MODEL_SAMPLE_RATE,
    WIRE_SAMPLE_RATE,
    ModelToWire,
    WireFrameAssembler,
    WireToModel,
)
from personaplex_types import (
    AgentText,
    ConversationStarted,
    PersonaPlexConfig,
    PersonaPlexInput,
    PersonaPlexOutput,
    PersonaPlexState,
    StateUpdate,
)

logger = get_logger(__name__)

FRAME_RATE = 12.5
"""Mimi's token rate in Hz — one 80 ms frame per step, in and out."""

_PLAYOUT_FRAMES = 4
"""Frames of agent speech that may queue ahead of the wire, i.e. 320 ms.

The loop is paced by the mic, so it cannot outrun playout for long and this
bound rarely binds; what it buys is room to absorb a step that takes longer
than its 80 ms budget without the client hearing a gap. A deeper queue would
only add latency to a conversation, which is the one thing that must stay low.
"""

_WARMUP_STEPS = 4
"""Zero-input steps taken at load, to build the CUDA graphs before a client waits."""

_TEXT_CONTROL_TOKENS = frozenset({0, 1, 2, 3})
"""Text ids carrying no transcript: EPAD, BOS, EOS, and PAD.

Upstream filters only EPAD and PAD, so its client also receives the raw ``<s>``
and ``</s>`` pieces for BOS and EOS. Those are structural markers rather than
speech, so they are dropped here; this affects the transcript only, never audio.
"""

_SYSTEM_TAG = "<system>"

_BACKLOG_FRAMES = 25
"""Inbound blocks waiting before the model is reported as falling behind.

Counted in transport blocks, whose size the transport chooses — 10 ms each in
practice, so this is roughly a quarter second of speech. Deliberately well above
ordinary network jitter: the point is to name a model that cannot keep up with
real time, not to narrate a queue that drains on the next step.
"""

_BACKLOG_QUIET = 5.0
"""Seconds between repeats of the backlog warning, so a bad session logs once."""


class PersonaPlex(ReactorPipeline):
    """Hold a spoken conversation with a persona a client describes in text."""

    state: PersonaPlexState
    input: PersonaPlexInput

    # Pinning `fps` tells the runtime every chunk plays out at Mimi's own token
    # rate. Leaving it unpinned would tag each chunk with the measured step time
    # instead, which for audio is meaningless — the frame is 80 ms of speech
    # however long it took to compute, and pacing it any faster or slower would
    # resample the agent's voice.
    fps = FRAME_RATE
    buffer_size = _PLAYOUT_FRAMES

    def __init__(self) -> None:
        super().__init__()
        self._config: PersonaPlexConfig | None = None
        self._assets: PersonaPlexAssets | None = None
        self._device = torch.device("cpu")
        self._mimi: Any = None
        self._lm_gen: Any = None
        self._tokenizer: Any = None
        self._agent_codebooks = 0
        self._model_frame_samples = 0
        self._assembler: WireFrameAssembler | None = None
        self._to_model = WireToModel()
        self._to_wire = ModelToWire()
        self._lagging_since: float | None = None

    # -- load ------------------------------------------------------------------

    def load(self, config_path: Path | None) -> None:
        """Fetch the assets, place the model, and warm its CUDA graphs.

        Runs once before any client can connect, so the first conversation pays
        for downloads and graph capture only if this process has never served.

        Args:
            config_path: Path the runtime resolved from ``runtime.config``.

        Raises:
            RuntimeError: If Mimi's own frame rate disagrees with the rate this
                adapter pins its playout to. Every clock here derives from that
                one number, so a mismatch would emit correctly-sized frames at
                the wrong speed — audible as a pitch shift rather than an error.
        """
        config = read_config(config_path)
        assets = prepare_assets(config)
        device = _select_device(config.device)
        logger.info("loading PersonaPlex", device=str(device))

        mimi = loaders.get_mimi(str(assets.mimi_weights), device)
        tokenizer = sentencepiece.SentencePieceProcessor(str(assets.tokenizer))
        lm = loaders.get_moshi_lm(
            str(assets.lm_weights), device=device, cpu_offload=config.cpu_offload
        )
        lm.eval()

        if int(mimi.sample_rate) != MODEL_SAMPLE_RATE or not math.isclose(
            float(mimi.frame_rate), FRAME_RATE
        ):
            raise RuntimeError(
                f"Mimi reports {mimi.sample_rate} Hz at {mimi.frame_rate} fps; this "
                f"adapter is built for {MODEL_SAMPLE_RATE} Hz at {FRAME_RATE} fps"
            )

        lm_gen = LMGen(
            lm,
            device=device,
            sample_rate=int(mimi.sample_rate),
            frame_rate=mimi.frame_rate,
            audio_silence_frame_cnt=int(
                config.prompt_silence_seconds * mimi.frame_rate
            ),
            top_k=config.audio_top_k,
            top_k_text=config.text_top_k,
            save_voice_prompt_embeddings=False,
        )

        # Both modules stay in streaming mode for the life of the process;
        # `reset_streaming` is what starts a fresh conversation within it.
        mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        self._config = config
        self._assets = assets
        self._device = device
        self._mimi = mimi
        self._lm_gen = lm_gen
        self._tokenizer = tokenizer
        # The language model runs two audio streams — the agent's own speech and
        # its prediction of the participant's — so `lm.dep_q` counts both. Only
        # the agent's half is decoded, and its width is exactly the number of
        # codebooks Mimi was configured for, so read it from Mimi rather than
        # halving dep_q or hard-coding 8.
        agent_codebooks = int(mimi.num_codebooks)
        if int(lm.dep_q) < agent_codebooks:
            raise RuntimeError(
                f"the language model carries {lm.dep_q} audio codebooks, fewer than "
                f"the {agent_codebooks} Mimi decodes"
            )
        self._agent_codebooks = agent_codebooks
        self._model_frame_samples = int(mimi.sample_rate / mimi.frame_rate)
        self._assembler = WireFrameAssembler(
            self._model_frame_samples * (WIRE_SAMPLE_RATE // MODEL_SAMPLE_RATE)
        )

        self._warm_up()
        logger.info(
            "PersonaPlex loaded",
            voices=len(assets.voices),
            frame_samples=self._model_frame_samples,
        )

    def _warm_up(self) -> None:
        """Run silent steps so CUDA graph capture happens before a client connects.

        Mirrors upstream's warmup: encode silence, step the language model, and
        decode what it produces, enough times for every graph on the path to be
        captured. The streaming state this leaves behind is discarded by the
        `reset_streaming` that starts the first conversation.
        """
        silence = torch.zeros(
            1, 1, self._model_frame_samples, dtype=torch.float32, device=self._device
        )
        for _ in range(_WARMUP_STEPS):
            codes = self._mimi.encode(silence)
            for index in range(codes.shape[-1]):
                tokens = self._lm_gen.step(codes[:, :, index : index + 1])
                if tokens is not None:
                    self._mimi.decode(tokens[:, 1 : 1 + self._agent_codebooks])
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    # -- lifecycle -------------------------------------------------------------

    @session_started
    def on_session_started(self) -> None:
        """Arm the first conversation before the session's first client connects.

        Once-per-session work belongs here rather than in ``@connected``: a
        session that ends server-side is torn down without a per-client
        disconnect, so anything counted across connections would not return to
        its starting value for the next session.
        """
        self.state._conversation_pending = True
        self.state._conversation_index = 0

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Hand a joining client the current snapshot, including the voice list."""
        await client.send(self._snapshot())

    @session_ended
    def on_session_ended(self) -> None:
        """Drop the conversation's streaming state, keeping the weights resident.

        Fires on a server-side close as well as a natural end, so it is the only
        place that reliably sees every session out.
        """
        if self._lm_gen is not None:
            self._mimi.reset_streaming()
            self._lm_gen.reset_streaming()
        self._reset_stream_buffers()
        self._lagging_since = None

    # -- commands --------------------------------------------------------------

    @event(
        name="set_persona",
        description=(
            "Describe who the agent is, then start a conversation under that "
            "role. The model is conditioned on the text before it generates, so "
            "this restarts the conversation rather than steering the current one."
        ),
    )
    async def set_persona(
        self,
        persona: str = InputField(
            max_length=2000,
            description=(
                "Role prompt: who the agent is, its background, and the "
                "scenario. Pass an empty string to leave the agent "
                "unconditioned by any role."
            ),
        ),
    ) -> StateUpdate:
        """Store the role prompt and arm a conversation that runs under it."""
        if self.state is None:
            raise CommandError("no_session", "No session is running.")
        self.state.persona = persona
        self._arm_conversation()
        return self._snapshot()

    @event(
        name="set_voice",
        description=(
            "Select the voice the agent speaks in, then start a conversation "
            "using it. StateUpdate.voices lists the names available."
        ),
    )
    async def set_voice(
        self,
        voice: str = InputField(
            description=(
                "Voice prompt name, as published in StateUpdate.voices "
                "(for example NATF0 or VARM2)."
            ),
            min_length=1,
            max_length=64,
            # An identifier chosen from a published list, not authored prose.
            moderate=False,
        ),
    ) -> StateUpdate:
        """Store the voice and arm a conversation that speaks in it.

        The names are validated against what the model repository actually
        shipped rather than a list baked into this adapter, so the error tells a
        client what it may choose from instead of what a past release carried.
        """
        if self.state is None:
            raise CommandError("no_session", "No session is running.")
        assets = self._require_assets()
        if voice not in assets.voices:
            raise CommandError(
                "unknown_voice",
                f"Unknown voice {voice!r}. Available: {', '.join(assets.voice_names)}.",
            )
        self.state.voice = voice
        self._arm_conversation()
        return self._snapshot()

    @event(
        name="restart",
        description=(
            "Start a fresh conversation under the current persona and voice, "
            "discarding everything said so far. Queued agent speech is dropped "
            "so nothing from the old conversation plays after the cut."
        ),
    )
    async def restart(self) -> StateUpdate:
        """Arm a fresh conversation without changing any setting."""
        if self.state is None:
            raise CommandError("no_session", "No session is running.")
        self._arm_conversation()
        return self._snapshot()

    # -- inference -------------------------------------------------------------

    async def inference(self) -> AsyncIterator[PersonaPlexOutput | _IdleType]:
        """Step the conversation once per inbound 80 ms frame.

        Each turn reads one frame of the participant's speech, feeds it to the
        model, and yields the frame the model spoke in return. The read is the
        clock: it cannot complete faster than the mic delivers, so the loop runs
        at real time without measuring anything.

        A conversation that is armed is built first. That is the several-second
        conditioning phase, and it runs inside this turn on purpose — the
        pipeline holds its generator lock across a turn, so no command can
        change the persona halfway through the prompt it is being applied from.
        """
        while True:
            if self.state._conversation_pending:
                await self._begin_conversation()

            frame = await self._read_model_frame()
            speech, text = self._step_conversation(frame)

            if text:
                await self.send(
                    AgentText(
                        text=text, conversation_index=self.state._conversation_index
                    )
                )
            if speech is None:
                # The language model is still inside its own token delay and has
                # not produced a frame yet. Skipping the turn lets the transport
                # fill the gap with real-time silence, which keeps the audio
                # clock advancing; emitting an invented frame would not.
                yield Idle
                continue

            yield PersonaPlexOutput(voice=speech)

    async def _begin_conversation(self) -> None:
        """Condition the model on the current voice and persona, then open the floor.

        Follows upstream's order exactly: load the voice prompt, encode the role
        prompt, reset both streaming states, step the prompts through the
        language model, and reset Mimi again — the voice prompt is encoded with
        the same Mimi instance the conversation then uses, so its encoder state
        has to be cleared before the participant's first frame reaches it.
        """
        state = self.state
        assets = self._require_assets()
        config = self._require_config()

        voice_path = assets.voices.get(state.voice)
        if voice_path is None:
            # A voice can only be missing if it was valid when set and the
            # assets changed underneath, so fall back rather than end the
            # session: a conversation in the wrong voice beats no conversation.
            fallback = assets.voice_names[0]
            logger.warning(
                "voice is no longer available; falling back",
                requested=state.voice,
                fallback=fallback,
            )
            state.voice = fallback
            voice_path = assets.voices[fallback]

        state._conversation_index += 1
        logger.info(
            "conditioning conversation",
            index=state._conversation_index,
            voice=state.voice,
            persona_chars=len(state.persona),
        )

        if config.seed is not None:
            _seed_all(config.seed)

        lm_gen = self._lm_gen
        if lm_gen.voice_prompt != str(voice_path):
            if voice_path.suffix == ".pt":
                lm_gen.load_voice_prompt_embeddings(str(voice_path))
            else:
                lm_gen.load_voice_prompt(str(voice_path))
        lm_gen.text_prompt_tokens = (
            self._tokenizer.encode(_wrap_with_system_tags(state.persona))
            if state.persona.strip()
            else None
        )

        started = time.perf_counter()
        self._mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(self._mimi)
        self._mimi.reset_streaming()

        # Everything that arrived or was queued while conditioning ran belongs
        # to no conversation: drop the participant's backlog so the agent does
        # not answer speech from seconds ago, and cut queued playout so the old
        # conversation cannot be heard after the new one starts.
        self.output.flush()
        self._reset_stream_buffers()
        self.input.mic.clear()

        state._conversation_pending = False
        self._lagging_since = None
        logger.info(
            "conversation ready",
            index=state._conversation_index,
            seconds=round(time.perf_counter() - started, 2),
        )
        await self.send(
            ConversationStarted(
                persona=state.persona,
                voice=state.voice,
                conversation_index=state._conversation_index,
            )
        )
        await self.send(self._snapshot())

    async def _read_model_frame(self) -> npt.NDArray[np.float32]:
        """Return the next 80 ms of participant speech at the model's rate.

        Inbound blocks are whatever the transport decoded — 10 ms each in
        practice — so they are accumulated until one whole frame is held. The
        read blocks, which is what paces this model, and raises ``BufferClosed``
        on teardown, which the pipeline driver treats as the end of the session.
        """
        assembler = self._require_assembler()
        while True:
            frame = assembler.take()
            if frame is not None:
                return self._to_model.process(frame)
            blocks = await self.input.mic.read(1, mode=ReadMode.FIFO)
            assembler.push(blocks[0].data)
            self._note_backlog()

    def _step_conversation(
        self, frame: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.int16] | None, str]:
        """Run one model step over one frame, returning its speech and its text.

        Args:
            frame: One frame of participant speech, mono float32 at the model's
                sample rate.

        Returns:
            The agent's speech for this step as wire-rate int16 samples, or
            ``None`` when the model produced no frame, together with whatever
            text it produced (empty when it produced none).
        """
        state = self.state
        lm_gen = self._lm_gen
        # Read live each step: the temperatures are the two settings a client can
        # change without restarting, so they are applied here rather than cached.
        lm_gen.temp = float(state.audio_temperature)
        lm_gen.temp_text = float(state.text_temperature)

        chunk = torch.from_numpy(frame).to(self._device).reshape(1, 1, -1)
        codes = self._mimi.encode(chunk)

        speech: list[npt.NDArray[np.float32]] = []
        pieces: list[str] = []
        for index in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, index : index + 1])
            if tokens is None:
                continue
            # Channel 0 carries the text token; 1..agent_codebooks carry the
            # agent's speech. The channels above those are the model's prediction
            # of the participant's own stream, which nothing here decodes.
            decoded = self._mimi.decode(tokens[:, 1 : 1 + self._agent_codebooks])
            speech.append(decoded[0, 0].float().cpu().numpy())
            piece = self._text_piece(int(tokens[0, 0, 0].item()))
            if piece:
                pieces.append(piece)

        if not speech:
            return None, "".join(pieces)
        generated = speech[0] if len(speech) == 1 else np.concatenate(speech)
        # Shaped (1, M) — the runtime's documented form for a mono audio track.
        wire = self._to_wire.process(generated).reshape(1, -1)
        return wire, "".join(pieces)

    def _text_piece(self, token: int) -> str:
        """Render one text token as transcript text, or ``""`` for a control token."""
        if token in _TEXT_CONTROL_TOKENS:
            return ""
        return str(self._tokenizer.id_to_piece(token)).replace("▁", " ")

    # -- internals -------------------------------------------------------------

    def _arm_conversation(self) -> None:
        """Mark the conversation for rebuilding on the next inference turn."""
        self.state._conversation_pending = True

    def _snapshot(self) -> StateUpdate:
        """Build the complete client-facing snapshot of the conversation."""
        return StateUpdate.from_state(self.state, self._require_assets().voice_names)

    def _reset_stream_buffers(self) -> None:
        """Clear the resampler history and the partial inbound frame."""
        self._to_model.reset()
        self._to_wire.reset()
        if self._assembler is not None:
            self._assembler.reset()

    def _note_backlog(self) -> None:
        """Log once while the model is falling behind the participant's speech.

        The inbound buffer drops its oldest frames when it fills, so a model
        that cannot keep up with real time loses audio rather than growing a
        queue. That is the right behaviour for a conversation, but it is silent,
        so it is worth saying out loud — a session that reports this is a
        session where the agent is missing parts of what was said.
        """
        available = self.input.mic.available
        if available < _BACKLOG_FRAMES:
            self._lagging_since = None
            return
        now = time.monotonic()
        if self._lagging_since is None or now - self._lagging_since > _BACKLOG_QUIET:
            self._lagging_since = now
            logger.warning(
                "inbound audio is backing up; the agent is missing speech",
                blocks_waiting=available,
            )

    def _require_config(self) -> PersonaPlexConfig:
        """Return the loaded config, or fail clearly if ``load`` never ran."""
        if self._config is None:
            raise RuntimeError("PersonaPlex was not loaded")
        return self._config

    def _require_assets(self) -> PersonaPlexAssets:
        """Return the resolved assets, or fail clearly if ``load`` never ran."""
        if self._assets is None:
            raise RuntimeError("PersonaPlex was not loaded")
        return self._assets

    def _require_assembler(self) -> WireFrameAssembler:
        """Return the inbound frame assembler, or fail clearly if ``load`` never ran."""
        if self._assembler is None:
            raise RuntimeError("PersonaPlex was not loaded")
        return self._assembler


def _wrap_with_system_tags(persona: str) -> str:
    """Wrap a role prompt in the system tags the model was trained to read.

    Kept byte-identical to upstream's helper, including the closing tag being
    ``<system>`` rather than ``</system>``: the tokenizer was trained that way,
    and a well-formed closing tag would not match what the model expects.
    """
    cleaned = persona.strip()
    if cleaned.startswith(_SYSTEM_TAG) and cleaned.endswith(_SYSTEM_TAG):
        return cleaned
    return f"{_SYSTEM_TAG} {cleaned} {_SYSTEM_TAG}"


def _select_device(requested: str) -> torch.device:
    """Resolve the configured device, preferring CUDA when asked for ``auto``."""
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    """Seed torch, NumPy, and Python so a conversation is reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
