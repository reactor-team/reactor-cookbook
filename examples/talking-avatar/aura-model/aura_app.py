"""AURA-1.3B as a LiveKit avatar: :class:`avatar_app.AvatarApp` with AURA's model half.

The application half (speech in, the playout clock, the playback reports, the commands) is
:class:`avatar_app.AvatarApp`; this module only builds the model from ``config.yml`` and the mounted weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reactor_runtime import get_logger, get_weights_path

from aura_assets import prepare_weights
from aura_model import DEFAULT_PROMPT, FPS, AuraModel
from avatar_app import AvatarApp
from avatar_contract import AvatarModel

logger = get_logger(__name__)


class AuraAvatar(AvatarApp):
    """A talking avatar from one portrait, lip-synced live to speech streamed from a LiveKit room."""

    fps = FPS

    def create_model(
        self, config: dict[str, Any], config_dir: Path
    ) -> tuple[AvatarModel, bytes]:
        """Load AURA-1.3B on the GPU from the mounted weights and render its warm-up chunks.

        ``config.yml`` keys: ``avatar`` (the default portrait, relative to this folder), ``prompt``, ``seed``,
        ``audio_context_s``, ``compile`` and ``warmup_blocks``.
        """
        weights = prepare_weights(get_weights_path())
        avatar = (config_dir / str(config.get("avatar", "portrait.jpg"))).read_bytes()
        model = AuraModel()
        model.load(
            weights,
            prompt=str(config.get("prompt") or DEFAULT_PROMPT),
            seed=int(config.get("seed", 42)),
            audio_context_s=float(config.get("audio_context_s", 4.0)),
            compile_dit=bool(config.get("compile", True)),
            warmup_avatar=avatar,
            warmup_blocks=int(config.get("warmup_blocks", 6)),
        )
        logger.info("AURA-1.3B loaded", compile=bool(config.get("compile", True)))
        return model, avatar
