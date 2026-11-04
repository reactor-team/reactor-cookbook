"""StreamForce, the model half.

A plain class with load / generate / reset that imports nothing from reactor_runtime. It wraps
the upstream StreamForce source (pinned revision plus ``streamforce.patch``, which adds the
per-window stream) and steps it one rolling window per ``generate()``:

* window w denoises blocks w-3 .. w at the four noise levels of the distilled student; block w
  enters as fresh noise and block w-3 leaves clean, is decoded by the causal VAE and returned
  (9 video frames for block 0, 12 after; nothing on the first three steps of a run);
* the force current at the step is built into block w's conditions, so a force change shows up
  in the block that enters next and every later one (what the offline source does);
* the run ends at the temporal RoPE table (1024 latent frames = 341 blocks): the step that would
  pass it switches to the source's closing windows, which finish the blocks still in flight.

The two halves meet on ``StreamForceInput`` / ``StreamForceResult``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Force:
    """One force, in the training data's units.

    mode: "point" (a push at a place) or "wind" (a global field).
    strength: 0..1 of the strongest force in training. angle: degrees, counter-clockwise from +x
    (0 = right, 90 = up). x, y: point only, 0..1 over the output frame, origin top-left.
    """

    mode: str
    strength: float
    angle: float
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class StreamForceInput:
    run_id: int                      # a new id starts a new run (stream)
    first_frame: np.ndarray | None   # RGB uint8 HxWx3, any size; rides only on a run's first step
    seed: int                        # applied at run start
    prompt: str                      # current text; applied at the step boundary when it changes
    force: Force                     # current force; applied to the block that enters this step


@dataclass(frozen=True)
class StreamForceResult:
    run_id: int
    window: int                      # the window this step ran
    block: int | None                # the block whose frames are returned
    frames: np.ndarray | None        # uint8 (T, H, W, 3), or None while the first blocks are in flight
    prompt: str                      # the prompt this window was conditioned on
    force: Force                     # the force built into the block that entered this window
    seed: int
    finished: bool                   # the last block of the run was returned


class NotStarted(Exception):
    """The step names a run the model does not hold, and carries no first frame to start it."""


class StreamForceModel:
    """The distilled StreamForce student, one rolling window per step."""

    def load(self, config_path: str | Path, source_path: str | Path, wan_root: str | Path,
             checkpoint_path: str | Path) -> None:
        """Load the generator, text encoder and VAE once, and warm the real path.

        ``config_path`` is the model YAML (its ``model``, ``freeze`` and ``warmup_windows``
        keys); ``source_path`` the patched upstream checkout; ``wan_root`` the directory holding
        ``Wan2.2-TI2V-5B/``; ``checkpoint_path`` the StreamForce checkpoint.
        """
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        src = Path(source_path).resolve()

        # Freeze nondeterminism exactly as the baseline did (before CUDA is touched).
        fz = cfg["freeze"]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = fz["cublas_workspace_config"]
        import torch

        torch.backends.cudnn.benchmark = fz["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = fz["cudnn_deterministic"]
        if fz["use_deterministic_algorithms"]:
            torch.use_deterministic_algorithms(True)
        torch.set_grad_enabled(False)
        self.torch = torch

        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import utils.wan_wrapper as wan_wrapper

        wan_wrapper.WAN_MODELS_ROOT = str(wan_root)
        from omegaconf import OmegaConf
        from pipeline import force_conditions as fc
        from pipeline.rolling_forcing_inference import ControlRollingForcingInferencePipeline
        from pipeline.rolling_forcing_step import (
            BlockDecoder,
            PositionLimitReached,
            RollingForcingStream,
            encode_first_frame,
        )
        from utils.misc import set_seed

        self.fc, self.set_seed = fc, set_seed
        self.PositionLimitReached, self.encode_first_frame = PositionLimitReached, encode_first_frame

        m = cfg["model"]
        self.H, self.W = int(m["height"]), int(m["width"])
        config = OmegaConf.merge(OmegaConf.load(src / "configs/default_config.yaml"),
                                 OmegaConf.load(src / m["config"]))
        if "generator_model_kwargs" not in config:
            config.generator_model_kwargs = {}
        config.generator_model_kwargs.rolling_forcing_attention = True
        config.generator_model_kwargs.rolling_forcing_block_frames = m["rolling_forcing_block_frames"]
        config.generator_model_kwargs.rolling_forcing_max_frames = m["rolling_forcing_max_frames"]
        config.rolling_forcing_cache_frames = m["rolling_forcing_cache_frames"]
        config.model_kwargs.control_weight = m["control_weight"]
        config.gradient_checkpointing = False

        self.device = torch.device("cuda")
        pipe = ControlRollingForcingInferencePipeline(config, device=self.device)

        ckpt = Path(checkpoint_path)
        state = torch.load(ckpt, map_location="cpu", mmap=True)
        key = "generator_ema" if m["use_ema"] else "generator"
        if key not in state:
            raise RuntimeError(f"{ckpt} has no '{key}' state dict (keys: {sorted(state)})")
        ren = lambda n: n.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        pipe.generator.load_state_dict({ren(n): p for n, p in state[key].items()})
        del state
        self.pipe = pipe.to(device=self.device, dtype=torch.bfloat16)

        self.stream = RollingForcingStream(self.pipe)
        self.decoder = BlockDecoder(self.pipe.vae)
        self.tokenizer = self.pipe.text_encoder.tokenizer
        self.max_blocks = self.stream.max_blocks
        self.run_id: int | None = None
        self._clear_run()
        self._warm_up(int(cfg.get("warmup_windows", 0)))

    # ------------------------------------------------------------ public helpers ---
    def prompt_tokens(self, text: str) -> int:
        """Tokens the text encoder reads for this text, special tokens included (its budget is 512)."""
        tok = self.tokenizer
        return len(tok.tokenizer(tok._clean(text), add_special_tokens=True).input_ids)

    @property
    def frames_per_session(self) -> int:
        return 9 + 12 * (self.max_blocks - 1)

    # ------------------------------------------------------------------- step ---
    def generate(self, inp: StreamForceInput) -> StreamForceResult:
        # Grad mode is per thread, and the runtime calls generate() from its own thread: turn it
        # off here, as the source does for its whole process.
        with self.torch.no_grad():
            return self._generate(inp)

    def _generate(self, inp: StreamForceInput) -> StreamForceResult:
        if inp.run_id != self.run_id:
            if inp.first_frame is None:
                raise NotStarted(f"run {inp.run_id} has no first frame")
            self._start_run(inp)

        if inp.prompt != self.prompt:
            self.stream.set_text(self._text(inp.prompt))
            self.prompt = inp.prompt

        hint, masked = self._conditions(inp.force)
        clean = None
        if not self.closing:
            try:
                clean = self.stream.window(hint, masked)
            except self.PositionLimitReached:
                self.closing = True
        if self.closing:
            clean = self.stream.window(None, masked, total_blocks=self.max_blocks)

        frames, block = None, None
        if clean is not None:
            block, latents = clean
            video = self.decoder.decode(latents)                      # [1, T, 3, H, W] in 0..1
            # The source writes frames as (x * 255).astype(uint8) on the host; the same float32
            # multiply and truncating cast on the GPU gives the same bytes and copies 1/4 of them.
            frames = (video[0].permute(0, 2, 3, 1) * 255).to(self.torch.uint8).cpu().numpy()
        finished = self.closing and block == self.max_blocks - 1
        return StreamForceResult(run_id=self.run_id, window=self.stream.window_index - 1, block=block,
                          frames=frames, prompt=self.prompt, force=inp.force, seed=self.seed,
                          finished=finished)

    def reset(self) -> None:
        """Forget the run. The next step must start one."""
        self.run_id = None
        self._clear_run()

    # ---------------------------------------------------------------- internals ---
    def _clear_run(self) -> None:
        self.prompt = None
        self.seed = None
        self.image = None
        self._text_cache: dict[str, Any] = {}
        self._cond_cache: dict[Force, Any] = {}
        self.closing = False
        self.decoder.reset()
        self.stream.started = False
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _start_run(self, inp: StreamForceInput) -> None:
        self._clear_run()
        self.set_seed(int(inp.seed))                     # python, numpy, torch CPU + every CUDA generator
        self.seed = int(inp.seed)
        self.image = self.fc.fit_image(np.array(inp.first_frame), self.H, self.W)   # uint8 [1, 3, H, W]
        initial_latent = self.encode_first_frame(self.pipe, self.image.float().unsqueeze(1))
        self.stream.begin(initial_latent, self._text(inp.prompt))
        self.prompt = inp.prompt
        self.run_id = inp.run_id

    def _text(self, prompt: str):
        """Encode once per text for the run; a handful of recent encodings are kept."""
        if prompt not in self._text_cache:
            if len(self._text_cache) >= 8:
                self._text_cache.pop(next(iter(self._text_cache)))
            self._text_cache[prompt] = self.pipe.text_encoder(text_prompts=[prompt])
        return self._text_cache[prompt]

    def _conditions(self, f: Force):
        """Build a force's block conditions once; only the current force is kept (a drag sends many)."""
        if f not in self._cond_cache:
            self._cond_cache.clear()
            if f.mode == "point":
                xp, yp = self.fc.frame_to_canvas(f.x, f.y, self.H, self.W)
            else:
                xp, yp = 0.0, 0.0
            sig = self.fc.frame_signal(f.mode, f.strength, f.angle, xp, yp, self.H, self.W)
            self._cond_cache[f] = self.fc.block_conditions(sig, self.image, self.device)
        return self._cond_cache[f]

    def _warm_up(self, windows: int) -> None:
        """Run the real path once per force mode at a client-like image size, then forget it."""
        if windows <= 0:
            return
        yy, xx = np.mgrid[0:720, 0:1280]
        img = np.stack([(xx * 255 // 1279), (yy * 255 // 719), ((xx + yy) % 256)], -1).astype(np.uint8)
        for i, force in enumerate([Force("point", 0.6, 30.0, 0.5, 0.6), Force("wind", 0.6, 200.0)]):
            for w in range(windows):
                self.generate(StreamForceInput(run_id=-(i + 1), first_frame=img if w == 0 else None, seed=0,
                                        prompt="warm-up" if w < windows // 2 else "warm-up, changed",
                                        force=force))
        self.reset()
