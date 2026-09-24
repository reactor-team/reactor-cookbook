"""Configuration and checkpoint resolution for Cosmos3-Policy-DROID.

The model half reads ``cosmos3_policy_droid.yaml`` through :func:`read_config`
and routes every Hugging Face lookup the vendored ``cosmos_framework`` makes
through :func:`route_checkpoint_downloads`, so the policy checkpoint and the
video tokenizer it depends on land in the deployment's weights directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PolicyConfig:
    """Describe which checkpoint to serve and how to sample it.

    Attributes:
        checkpoint: Hugging Face repository id of a DROID policy checkpoint,
            or a local directory holding one.
        hf_revision: Revision of the repository to download.
        format_prompt_as_json: Serve the task as the structured JSON prompt the
            checkpoint was trained on. ``None`` keeps the checkpoint's default.
        guidance_interval: Inclusive denoising-timestep range that applies
            classifier-free guidance, or ``None`` to apply it at every step.
        guidance: Classifier-free guidance scale.
        num_steps: Denoising steps per prediction.
        resolution: Action transform resolution tier.
        action_chunk_size: Number of action steps in a chunk.
        device_id: CUDA device the policy loads on.
        warmup: Run two throwaway predictions at load so compilation happens
            before the first session.
    """

    checkpoint: str
    hf_revision: str
    format_prompt_as_json: bool | None
    guidance_interval: tuple[int, int] | None
    guidance: float
    num_steps: int
    resolution: str
    action_chunk_size: int
    device_id: int
    warmup: bool


def read_config(config_path: Path | None) -> PolicyConfig:
    """Parse the model's YAML config, applying the Edge checkpoint's defaults."""
    raw: dict = {}
    if config_path is not None:
        raw = yaml.safe_load(config_path.read_text()) or {}
    interval = raw.get("guidance_interval")
    if interval is not None:
        if len(interval) != 2:
            raise ValueError("guidance_interval must be a pair [low, high]")
        interval = (int(interval[0]), int(interval[1]))
    fmt = raw.get("format_prompt_as_json")
    return PolicyConfig(
        checkpoint=str(raw.get("checkpoint", "nvidia/Cosmos3-Edge-Policy-DROID")),
        hf_revision=str(raw.get("hf_revision", "main")),
        format_prompt_as_json=None if fmt is None else bool(fmt),
        guidance_interval=interval,
        guidance=float(raw.get("guidance", 3.0)),
        num_steps=int(raw.get("num_steps", 4)),
        resolution=str(raw.get("resolution", "480")),
        action_chunk_size=int(raw.get("action_chunk_size", 32)),
        device_id=int(raw.get("device_id", 0)),
        warmup=bool(raw.get("warmup", True)),
    )


def route_checkpoint_downloads(cache_dir: Path) -> None:
    """Resolve the vendored framework's Hugging Face lookups through ``huggingface_hub``.

    Upstream's ``checkpoint_db`` downloads by shelling out to a separate ``uv``
    project that is not vendored. Both of its Hugging Face resolvers are
    replaced with ``huggingface_hub`` calls that cache under ``cache_dir``, so
    a checkpoint already present is reused and one that is absent is fetched.
    Idempotent: a second call leaves the first routing in place.
    """
    import huggingface_hub as hub

    from cosmos_framework.utils import checkpoint_db as db

    if getattr(db.CheckpointDirHf._download, "_routed", False):
        return
    cache = str(cache_dir)

    def dir_download(self) -> str:
        include = list(self.include) or None
        exclude = list(self.exclude) or None
        if self.subdirectory:
            include = [os.path.join(self.subdirectory, p) for p in (include or ["*"])]
        path = hub.snapshot_download(
            self.repository,
            revision=self.revision,
            repo_type=self.repository_type.value,
            allow_patterns=include,
            ignore_patterns=exclude,
            cache_dir=cache,
        )
        return os.path.join(path, self.subdirectory) if self.subdirectory else path

    def file_download(self) -> str:
        return hub.hf_hub_download(
            self.repository,
            self.filename,
            revision=self.revision,
            repo_type=self.repository_type.value,
            cache_dir=cache,
        )

    dir_download._routed = True  # type: ignore[attr-defined]
    file_download._routed = True  # type: ignore[attr-defined]
    db.CheckpointDirHf._download = dir_download
    db.CheckpointFileHf._download = file_download
