"""Check the StreamRender-H3 weights folder that prepare_weights.py builds. Nothing here downloads."""

import json
from pathlib import Path

PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
REPO_ID = "Efficient-Large-Model/StreamRender-H3"
REVISION = "9f81b42fb6bac3509ccc6ebf4cb6c8aec7469c62"
# The merged backbone that the bundle's download_assets.py writes next to the shards.
MERGED_BACKBONE = "backbone/1008b_step224_backbone_ema_merged.safetensors"


def check_weights(assets_path: Path) -> None:
    """Raise when ``assets.json`` or a file or folder that it names is absent."""
    if not assets_path.is_file():
        raise FileNotFoundError(
            f"StreamRender-H3 weights lack {assets_path}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )
    assets = json.loads(assets_path.read_text())["assets"]
    root = assets_path.parent
    missing = [
        str(root / value)
        for key, value in assets.items()
        if key != "prompt" and not (root / value).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"StreamRender-H3 weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )
