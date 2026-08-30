"""Which commands the session would accept right now.

Kept out of ``fasth3.py`` so the state machine is readable on its own and can be
tested without a GPU. ``StateUpdate.valid_commands`` carries the result to
clients, so a frontend enables and greys out controls from the snapshot instead
of re-deriving these rules.
"""

from __future__ import annotations

# Always available: they only ever record a value or report one.
_ALWAYS = ("set_prompt", "set_clip_seconds", "set_seed", "reset", "get_state")


def valid_commands(*, running: bool, paused: bool, ready: bool) -> list[str]:
    """Name every command the session would accept in this state.

    Args:
        running: A channel is live and streaming clips.
        paused: The output stream is held; only meaningful while ``running``.
        ready: A prompt is set, so ``start`` has everything it needs.
    """
    commands = list(_ALWAYS)
    if running:
        commands.append("resume" if paused else "pause")
        commands.append("stop")
    else:
        # The canvas fixes the video track's geometry, so it can only change
        # while nothing is streaming.
        commands.append("set_canvas")
        if ready:
            commands.append("start")
    return sorted(commands)


__all__ = ["valid_commands"]
