"""Drive the unmodified Matrix-Game 3.0 interactive inference loop."""

from __future__ import annotations

import importlib
import io
import queue
import re
import sys
import threading
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np
from matrix_game_3_0_assets import MatrixGame30Config
from numpy.typing import NDArray
from PIL import Image

_CHUNK_PATH = re.compile(r"_current_iteration_(\d+)\.mp4$")
_STOP = object()
_HOOK_LOCK = threading.Lock()


@dataclass(frozen=True)
class NativeAction:
    """Hold one Matrix keyboard vector and mouse vector for a native chunk."""

    keyboard: tuple[float, float, float, float, float, float]
    mouse: tuple[float, float]


@dataclass(frozen=True)
class _Chunk:
    """Carry one decoded chunk from the upstream visualization boundary."""

    index: int
    frames: NDArray[np.uint8]


@dataclass(frozen=True)
class _Failure:
    """Carry an upstream generation failure across the rollout thread boundary."""

    error: BaseException


class _StopRollout(BaseException):
    """Stop the upstream loop at its next native action boundary."""


class _InteractiveModule(Protocol):
    """Describe the upstream globals bridged at native iteration boundaries."""

    MatrixGame3Pipeline: type[Any]
    get_current_action: Callable[[], dict[str, Any]]
    process_video: Callable[..., None]


class _AttentionModule(Protocol):
    """Describe the upstream attention dispatcher selected by configuration."""

    attention: Callable[..., Any]


class _ModelModule(Protocol):
    """Describe the attention symbol imported by upstream model layers."""

    flash_attention: Callable[..., Any]


def action_from_controls(
    pressed_keys: AbstractSet[str], pitch: float, yaw: float
) -> NativeAction:
    """Map discrete keys and normalized camera axes to native conditions."""
    unsupported = set(pressed_keys).difference(("w", "a", "s", "d"))
    if unsupported:
        raise ValueError(f"unsupported Matrix key: {min(unsupported)}")
    if not -1.0 <= pitch <= 1.0 or not -1.0 <= yaw <= 1.0:
        raise ValueError("Matrix pitch and yaw must be in [-1, 1]")
    keyboard = (
        float("w" in pressed_keys),
        float("s" in pressed_keys),
        float("a" in pressed_keys),
        float("d" in pressed_keys),
        0.0,
        0.0,
    )
    return NativeAction(keyboard=keyboard, mouse=(pitch * 0.1, yaw * 0.1))


class MatrixGame30Backend:
    """Own one loaded upstream pipeline and one resumable official rollout."""

    def __init__(self, config: MatrixGame30Config) -> None:
        self._config = config
        self._module: _InteractiveModule | None = None
        self._pipeline: Any = None
        self._args: SimpleNamespace | None = None
        self._action_queue: queue.Queue[NativeAction | object] | None = None
        self._output_queue: queue.Queue[_Chunk | _Failure] | None = None
        self._thread: threading.Thread | None = None
        self._expected_chunk = 0

    def load(self) -> None:
        """Load the fast distilled model once without changing upstream source."""
        source_root = self._config.source_path / "Matrix-Game-3"
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        module = cast(
            _InteractiveModule,
            importlib.import_module("pipeline.inference_interactive_pipeline"),
        )
        configs = importlib.import_module("wan.configs")
        attention = cast(
            _AttentionModule, importlib.import_module("wan.modules.attention")
        )
        model_module = cast(_ModelModule, importlib.import_module("wan.modules.model"))
        model_module.flash_attention = attention.attention
        config = configs.WAN_CONFIGS["matrix_game3"]
        args = self._build_args()
        pipeline_type = module.MatrixGame3Pipeline
        self._pipeline = pipeline_type(
            config=config,
            checkpoint_dir=str(self._config.checkpoint_path),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=False,
            args=args,
            fa_version="0",
            use_base_model=False,
        )
        self._module = module
        self._args = args

    def reset(
        self,
        prompt: str,
        seed: int,
        anchor_image: Path | bytes,
    ) -> None:
        """Start a fresh official rollout while retaining loaded model weights."""
        if self._pipeline is None or self._module is None or self._args is None:
            raise RuntimeError("Matrix-Game 3.0 backend is not loaded")
        self._stop_rollout()
        image = _read_image(anchor_image)
        self._action_queue = queue.Queue()
        self._output_queue = queue.Queue()
        self._expected_chunk = 0
        self._thread = threading.Thread(
            target=self._run_rollout,
            args=(prompt, seed, image),
            name="matrix-game-3-0-rollout",
            daemon=True,
        )
        self._thread.start()

    def generate_chunk(self, action: NativeAction) -> NDArray[np.uint8]:
        """Generate exactly one native upstream iteration for the supplied action."""
        action_queue = self._action_queue
        output_queue = self._output_queue
        thread = self._thread
        if action_queue is None or output_queue is None or thread is None:
            raise RuntimeError("Matrix-Game 3.0 rollout has not been reset")
        if not thread.is_alive() and output_queue.empty():
            raise RuntimeError("Matrix-Game 3.0 rollout ended before the next chunk")
        action_queue.put(action)
        try:
            result = output_queue.get(timeout=self._config.chunk_timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError(
                "Matrix-Game 3.0 did not finish a native chunk within "
                f"{self._config.chunk_timeout_seconds:g} seconds"
            ) from error
        if isinstance(result, _Failure):
            raise RuntimeError(  # noqa: TRY004 - the upstream inference operation failed.
                "Matrix-Game 3.0 rollout failed"
            ) from result.error
        if result.index != self._expected_chunk:
            raise RuntimeError(
                f"Matrix returned chunk {result.index + 1}; expected {self._expected_chunk + 1}"
            )
        self._expected_chunk += 1
        return result.frames

    def end_session(self) -> None:
        """Release the active rollout state while retaining loaded model weights."""
        self._stop_rollout()

    def _build_args(self) -> SimpleNamespace:
        """Build the arguments consumed by the official interactive pipeline."""
        config = self._config
        return SimpleNamespace(
            size=config.size,
            ckpt_dir=str(config.checkpoint_path),
            ulysses_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            prompt="",
            seed=config.seed,
            image="",
            num_iterations=config.max_chunks,
            convert_model_dtype=False,
            output_dir=str(config.checkpoint_path / "reactor-output"),
            save_name="reactor",
            sample_shift=config.sample_shift,
            sample_guide_scale=config.guide_scale,
            num_inference_steps=config.num_inference_steps,
            lightvae_pruning_rate=config.lightvae_pruning_rate,
            use_async_vae=False,
            async_vae_warmup_iters=0,
            compile_vae=False,
            vae_type=config.vae_type,
            use_int8=config.use_int8,
            verify_quant=False,
            fa_version="0",
            interactive=True,
            use_base_model=False,
        )

    def _run_rollout(self, prompt: str, seed: int, image: Image.Image) -> None:
        """Run upstream generation and bridge its two interactive I/O hooks."""
        module = self._module
        pipeline = self._pipeline
        args = self._args
        action_queue = self._action_queue
        output_queue = self._output_queue
        if (
            module is None
            or pipeline is None
            or args is None
            or action_queue is None
            or output_queue is None
        ):
            return

        def get_current_action() -> dict[str, Any]:
            action = action_queue.get()
            if action is _STOP:
                raise _StopRollout
            if not isinstance(action, NativeAction):
                raise TypeError("Matrix action queue received an invalid value")
            torch = importlib.import_module("torch")
            return {
                "keyboard": torch.tensor(action.keyboard),
                "mouse": torch.tensor(action.mouse),
            }

        def capture_video(
            frames: NDArray[np.uint8],
            output_path: str,
            _config: object,
            _mouse_icon: object,
            **_kwargs: object,
        ) -> None:
            match = _CHUNK_PATH.search(str(output_path))
            if match is None:
                return
            output_queue.put(
                _Chunk(
                    index=int(match.group(1)),
                    frames=np.ascontiguousarray(np.asarray(frames, dtype=np.uint8)),
                )
            )

        with _HOOK_LOCK:
            original_action = module.get_current_action
            original_process_video = module.process_video
            module.get_current_action = get_current_action
            module.process_video = capture_video
            try:
                max_area = int(self._config.size.split("*")[0]) * int(
                    self._config.size.split("*")[1]
                )
                pipeline.generate(
                    prompt,
                    image,
                    max_area=max_area,
                    shift=self._config.sample_shift,
                    num_inference_steps=self._config.num_inference_steps,
                    guide_scale=self._config.guide_scale,
                    seed=seed,
                    use_base_model=False,
                    args=args,
                )
            except (_StopRollout, SystemExit):
                pass
            except Exception as error:  # noqa: BLE001 - transport any upstream failure.
                output_queue.put(_Failure(error))
            finally:
                module.get_current_action = original_action
                module.process_video = original_process_video

    def _stop_rollout(self) -> None:
        """Stop a rollout at its next action boundary and clear its queues."""
        thread = self._thread
        action_queue = self._action_queue
        if thread is not None and thread.is_alive():
            if action_queue is not None:
                action_queue.put(_STOP)
            thread.join(timeout=self._config.chunk_timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(
                    "Matrix-Game 3.0 rollout did not stop at a chunk boundary"
                )
        self._thread = None
        self._action_queue = None
        self._output_queue = None
        self._expected_chunk = 0


def _read_image(value: Path | bytes) -> Image.Image:
    """Return an owned RGB PIL image from a built-in path or uploaded bytes."""
    source: Path | io.BytesIO
    source = io.BytesIO(value) if isinstance(value, bytes) else value
    with Image.open(source) as image:
        return image.convert("RGB").copy()
