"""Resolving the PersonaPlex config and fetching its published assets.

The adapter needs four files from the model repository — the language model, the
Mimi codec checkpoint, the text tokenizer, and the archive of voice prompts —
plus the voices unpacked to disk. All of it is downloaded once into the
directory the Reactor CLI mounts as the weights root, so a rebuilt image
reuses what is already there and nothing large is baked into a layer.

The repository is gated: accept its licence on Hugging Face and pass a read
token in as ``HF_TOKEN``. Without one the first download fails with an
authorisation error rather than anything about the model.
"""

from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

from personaplex_types import PersonaPlexConfig

logger = get_logger(__name__)

_VOICES_ARCHIVE = "voices.tgz"
_VOICES_DIR = "voices"
_VOICE_SUFFIXES = (".pt", ".wav")
_UNPACK_MARKER = ".reactor-voices.json"


@dataclass(frozen=True)
class PersonaPlexAssets:
    """Where each downloaded asset landed on disk.

    Attributes:
        lm_weights: The 7B language-model checkpoint.
        mimi_weights: The Mimi encoder/decoder checkpoint.
        tokenizer: The SentencePiece text tokenizer.
        voices: Voice prompt name to the file conditioning on it.
    """

    lm_weights: Path
    mimi_weights: Path
    tokenizer: Path
    voices: dict[str, Path]

    @property
    def voice_names(self) -> list[str]:
        """Every voice prompt name available, sorted."""
        return sorted(self.voices)


def read_config(config_path: Path | None) -> PersonaPlexConfig:
    """Read the adapter's config file into a :class:`PersonaPlexConfig`.

    The runtime resolves ``runtime.config`` from ``reactor.yaml`` to a path and
    hands it over unparsed, so the format is this adapter's own. Every key has a
    default, and ``assets.path`` is resolved under the runtime's weights root so
    relocating the assets is a one-line change in ``reactor.yaml``.

    Args:
        config_path: Path the runtime resolved, or ``None`` when the manifest
            names no config file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If ``assets.path`` is absolute, or a numeric field is out of
            range — either would only surface later as a confusing failure.
    """
    document: dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            document = loaded

    assets = document.get("assets") if isinstance(document.get("assets"), dict) else {}
    sampling = (
        document.get("sampling") if isinstance(document.get("sampling"), dict) else {}
    )

    relative = str(assets.get("path", "personaplex"))
    if Path(relative).is_absolute():
        raise ValueError(
            f"assets.path must be relative to the weights root, got {relative!r}"
        )

    audio_top_k = int(sampling.get("audio_top_k", 250))
    text_top_k = int(sampling.get("text_top_k", 25))
    if audio_top_k < 1 or text_top_k < 1:
        raise ValueError("sampling.audio_top_k and sampling.text_top_k must be >= 1")

    silence = float(document.get("prompt_silence_seconds", 0.5))
    if silence < 0:
        raise ValueError("prompt_silence_seconds must not be negative")

    raw_seed = document.get("seed")
    seed = None if raw_seed is None or int(raw_seed) < 0 else int(raw_seed)

    return PersonaPlexConfig(
        repo_id=str(assets.get("repo_id", "nvidia/personaplex-7b-v1")),
        revision=str(assets.get("revision", "main")),
        assets_path=get_weights_path() / relative,
        device=str(document.get("device", "auto")),
        audio_top_k=audio_top_k,
        text_top_k=text_top_k,
        prompt_silence_seconds=silence,
        seed=seed,
        cpu_offload=bool(document.get("cpu_offload", False)),
    )


def prepare_assets(config: PersonaPlexConfig) -> PersonaPlexAssets:
    """Download and unpack everything the adapter loads, reusing what is present.

    Each file is fetched at the pinned revision into ``config.assets_path``.
    Hugging Face skips a file already complete there and resumes a partial one,
    so an interrupted first start continues rather than starting over. The voice
    archive is unpacked once, guarded by a marker recording the revision it came
    from, so a revision bump re-unpacks instead of silently mixing two sets.

    Args:
        config: The parsed adapter configuration.

    Returns:
        The resolved asset paths and the voice prompts found on disk.

    Raises:
        RuntimeError: If the archive carried no voice prompt files — the model
            cannot be conditioned without one, and failing here names the cause.
    """
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    root = config.assets_path
    root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "preparing PersonaPlex assets",
        repo_id=config.repo_id,
        revision=config.revision,
        path=str(root),
    )

    def fetch(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=config.repo_id,
                filename=filename,
                revision=config.revision,
                local_dir=root,
            )
        )

    lm_weights = fetch(loaders.MOSHI_NAME)
    mimi_weights = fetch(loaders.MIMI_NAME)
    tokenizer = fetch(loaders.TEXT_TOKENIZER_NAME)
    archive = fetch(_VOICES_ARCHIVE)

    voices_dir = _unpack_voices(archive, root, config.revision)
    voices = _collect_voices(voices_dir)
    if not voices:
        raise RuntimeError(
            f"{_VOICES_ARCHIVE} unpacked no voice prompts into {voices_dir}; "
            f"expected files ending in {' or '.join(_VOICE_SUFFIXES)}"
        )

    logger.info("PersonaPlex assets ready", voices=len(voices))
    return PersonaPlexAssets(
        lm_weights=lm_weights,
        mimi_weights=mimi_weights,
        tokenizer=tokenizer,
        voices=voices,
    )


def _unpack_voices(archive: Path, root: Path, revision: str) -> Path:
    """Extract the voice archive under *root*, once per revision.

    Extraction is filtered to plain data members, so a malformed archive cannot
    write outside the destination. The marker is written last and replaced
    atomically, so an interrupted extraction is retried on the next start rather
    than mistaken for a finished one.
    """
    voices_dir = root / _VOICES_DIR
    marker = root / _UNPACK_MARKER
    if marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recorded = {}
        if recorded.get("revision") == revision and voices_dir.is_dir():
            return voices_dir

    logger.info("unpacking voice prompts", archive=str(archive))
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=root, filter="data")
    if not voices_dir.is_dir():
        raise RuntimeError(f"{archive} did not contain a '{_VOICES_DIR}/' directory")

    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps({"revision": revision}), encoding="utf-8")
    os.replace(pending, marker)
    return voices_dir


def _collect_voices(voices_dir: Path) -> dict[str, Path]:
    """Map each voice prompt's name to its file, preferring precomputed embeddings.

    Upstream ships both a ``.wav`` of the reference speech and a ``.pt`` of the
    embeddings already encoded from it. The ``.pt`` skips a few seconds of Mimi
    encoding at every conversation start, so it wins when both are present.

    The names are read off disk rather than hard-coded: the set of packaged
    voices belongs to the model repository, and a list baked into the adapter
    would start rejecting valid voices the moment upstream adds one.
    """
    voices: dict[str, Path] = {}
    for suffix in reversed(_VOICE_SUFFIXES):
        for path in sorted(voices_dir.rglob(f"*{suffix}")):
            voices[path.stem] = path
    return voices
