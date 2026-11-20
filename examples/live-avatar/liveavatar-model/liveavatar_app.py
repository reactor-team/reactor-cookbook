"""LiveAvatar as a LiveKit avatar: :class:`avatar_app.AvatarApp` with LiveAvatar's model half.

The application half (speech in, the playout clock, the playback reports, the
commands) is :class:`avatar_app.AvatarApp`; this module only builds the model
from ``config.yml`` and the mounted weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from reactor_runtime import get_logger, get_weights_path

from avatar_app import AvatarApp
from avatar_contract import AvatarModel
from liveavatar_assets import prepare_weights
from liveavatar_model import DEFAULT_PROMPT, LiveAvatarModel

logger = get_logger(__name__)


class LiveAvatar(AvatarApp):
    """A talking avatar from one portrait, lip-synced live to speech streamed from a LiveKit room."""

    def create_model(
        self, config: dict[str, Any], config_dir: Path
    ) -> tuple[AvatarModel, bytes]:
        """Load LiveAvatar on the GPU from the mounted weights and render its warm-up blocks.

        ``config.yml`` keys: ``avatar`` (the default portrait; relative paths resolve against the upstream
        source checkout), ``prompt``, ``seed``, ``width`` and ``height``, ``fp8``, ``warmup_blocks``.
        """
        source = Path(os.environ.get("LIVEAVATAR_SOURCE", "/opt/liveavatar-source"))
        weights = prepare_weights(get_weights_path(), source)
        avatar_path = Path(config.get("avatar", "examples/livestream_1.png"))
        if not avatar_path.is_absolute():
            avatar_path = source / avatar_path
        avatar = avatar_path.read_bytes()
        model = LiveAvatarModel()
        model.load(
            weights,
            source=source,
            width=int(config.get("width", 512)),
            height=int(config.get("height", 512)),
            prompt=str(config.get("prompt") or DEFAULT_PROMPT),
            seed=int(config.get("seed", 420)),
            fp8=bool(config.get("fp8", True)),
            warmup_avatar=avatar,
            warmup_blocks=int(config.get("warmup_blocks", 12)),
        )
        logger.info(
            "LiveAvatar loaded", fp8=bool(config.get("fp8", True)), size=model.size
        )
        return model, avatar
