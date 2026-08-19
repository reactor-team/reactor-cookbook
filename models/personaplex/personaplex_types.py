"""Client-facing contract for the PersonaPlex adapter.

Everything a connected client can see or change lives here: the two audio
tracks, the session state its commands write, and the messages the model sends
back. Keeping them in one file makes the published schema readable as a single
document rather than something reassembled from the adapter's control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reactor_runtime import (
    Audio,
    Input,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
)

DEFAULT_PERSONA = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)
"""Role prompt used until a client sets its own — upstream's assistant template."""

DEFAULT_VOICE = "NATF0"
"""Voice prompt used until a client selects another, from upstream's natural set."""


# -- Tracks --------------------------------------------------------------------


class PersonaPlexInput(Input):
    """The audio this model listens to.

    ``mic`` is the participant's microphone. PersonaPlex encodes it
    incrementally while generating its own reply, so speech that overlaps the
    agent is heard rather than queued — interruptions and barge-ins are part of
    the model, not something the adapter arranges.
    """

    mic: Audio


class PersonaPlexOutput(Output):
    """The audio this model speaks.

    ``voice`` carries the agent's own generated speech. Its text is sent
    separately as :class:`AgentText`, so a client can show a transcript without
    running speech recognition over the track.
    """

    voice: Audio


# -- Session state -------------------------------------------------------------


class PersonaPlexState(InputState):
    """Conversation settings a client may change mid-session.

    ``persona`` and ``voice`` condition the model before it generates anything,
    so a change to either takes effect at the next conversation — either the
    ``restart`` command, or the setter itself, which starts one. The two
    temperatures are read on every step and take effect immediately.
    """

    persona: str = InputField(
        default=DEFAULT_PERSONA,
        max_length=2000,
        description=(
            "Role prompt describing who the agent is: its name, background, "
            "and the scenario it is in. Applied at the next conversation start."
        ),
    )
    voice: str = InputField(
        default=DEFAULT_VOICE,
        min_length=1,
        max_length=64,
        # A voice name is an identifier the model repository published, not
        # prose a client authored, so it is not a moderation candidate.
        moderate=False,
        description=(
            "Name of the packaged voice prompt establishing the agent's vocal "
            "characteristics. StateUpdate.voices lists the names this "
            "deployment carries. Applied at the next conversation start."
        ),
    )
    audio_temperature: float = InputField(
        default=0.8,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for the agent's speech tokens. Lower is "
            "steadier, higher is more varied. Takes effect immediately."
        ),
    )
    text_temperature: float = InputField(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for the agent's text tokens. Takes effect "
            "immediately."
        ),
    )

    # Private: the adapter's own bookkeeping, never surfaced as a command.
    # A conversation is pending until the model has run its voice and role
    # prompts through the language model, which it does on the first step of a
    # session and again after any restart.
    _conversation_pending: bool = True
    _conversation_index: int = 0


# -- Messages ------------------------------------------------------------------


class StateUpdate(ModelMessage):
    """Complete snapshot of the conversation's settings.

    Sent to a client when it joins, and returned by every command, so a client
    renders its controls from one message instead of reconstructing them from a
    sequence of partial updates.
    """

    persona: str = MessageField(description="Role prompt the agent is conditioned on.")
    voice: str = MessageField(description="Voice prompt selected for the agent.")
    voices: list[str] = MessageField(
        description="Every voice prompt name this deployment carries, sorted."
    )
    audio_temperature: float = MessageField(
        description="Sampling temperature now applied to speech tokens."
    )
    text_temperature: float = MessageField(
        description="Sampling temperature now applied to text tokens."
    )
    conversation_pending: bool = MessageField(
        description=(
            "True while the model is conditioning on the persona and voice. "
            "No agent speech is produced until it turns false."
        )
    )
    conversation_index: int = MessageField(
        description=(
            "How many conversations this session has started. Increments on "
            "every restart, so a client can discard text from a prior one."
        )
    )

    @classmethod
    def from_state(cls, state: PersonaPlexState, voices: list[str]) -> StateUpdate:
        """Build a snapshot from the live session state and the loaded voices."""
        return cls(
            persona=state.persona,
            voice=state.voice,
            voices=voices,
            audio_temperature=state.audio_temperature,
            text_temperature=state.text_temperature,
            conversation_pending=state._conversation_pending,
            conversation_index=state._conversation_index,
        )


class ConversationStarted(ModelMessage):
    """The model has finished conditioning and is now listening.

    Broadcast once per conversation, after the voice and role prompts have been
    stepped through the language model. Agent audio starts flowing from here.
    """

    persona: str = MessageField(description="Role prompt this conversation runs under.")
    voice: str = MessageField(description="Voice prompt this conversation runs under.")
    conversation_index: int = MessageField(
        description="Index of the conversation that just started."
    )


class AgentText(ModelMessage):
    """One piece of the agent's own speech, as text.

    PersonaPlex generates a text token alongside every 80 ms audio frame, so
    this arrives incrementally and slightly ahead of the audio that renders it.
    Concatenating the pieces in arrival order reproduces the transcript; the
    pieces already carry their leading spaces, so no separator is needed.
    """

    text: str = MessageField(description="Text fragment the agent just produced.")
    conversation_index: int = MessageField(
        description="Which conversation produced it, matching StateUpdate."
    )


# -- Configuration -------------------------------------------------------------


@dataclass(frozen=True)
class PersonaPlexConfig:
    """Deployment settings read from the adapter's config file.

    These are the choices a client does not make: where the weights live, what
    device to place them on, and the sampling bounds the temperatures move
    within. Client-settable defaults live on :class:`PersonaPlexState` instead,
    so the published schema and the running model cannot disagree about them.

    Attributes:
        repo_id: Hugging Face repository holding the checkpoint and voices.
        revision: Immutable commit of that repository to download.
        assets_path: Directory the assets are downloaded into, resolved under
            the runtime's weights root.
        device: Torch device string, or ``"auto"`` to prefer CUDA.
        audio_top_k: Top-k cutoff for speech-token sampling.
        text_top_k: Top-k cutoff for text-token sampling.
        prompt_silence_seconds: Silence held between the prompt phases, matching
            upstream's spacer.
        seed: Seed applied before each conversation, or ``None`` for unseeded.
        cpu_offload: Offload language-model layers to host memory when GPU
            memory is short.
    """

    repo_id: str
    revision: str
    assets_path: Path
    device: str
    audio_top_k: int
    text_top_k: int
    prompt_silence_seconds: float
    seed: int | None
    cpu_offload: bool
