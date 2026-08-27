#!/usr/bin/env python3
"""Run EVOKE inference behind a small JSON-line request protocol."""

from __future__ import annotations

import importlib
import inspect
import json
import queue
import sys
import threading
import traceback
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_RESPONSE_PREFIX = "REACTOR_EVOKE_RESPONSE "
_NEGATIVE_PROMPT = (
    "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, "
    "color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene "
    "changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, "
    "works, paintings, images, deformed, disfigured, messy background"
)


class _InteractiveStopError(Exception):
    """Signal normal termination while a rollout waits for another chunk."""


@dataclass(frozen=True)
class _ChunkInput:
    index: int
    trajectory: np.ndarray | None
    prompt: str


@dataclass(frozen=True)
class _GeneratedChunk:
    index: int
    frames: np.ndarray


class _Stop:
    pass


class _InteractiveSession:
    """Bridge JSON requests into one live EVOKE pipeline call."""

    def __init__(self, pose_base: np.ndarray | None) -> None:
        self._pose_base = pose_base
        self._inputs: queue.Queue[_ChunkInput | _Stop] = queue.Queue(maxsize=1)
        self._results: queue.Queue[_GeneratedChunk | Exception] = queue.Queue(maxsize=1)
        self._stop = _Stop()

    def conditions_for_chunk(
        self,
        chunk_index: int,
        *,
        frames_per_chunk: int,
    ) -> tuple[np.ndarray | None, str]:
        """Wait for the next chunk's camera trajectory and prompt."""
        if frames_per_chunk != 36:
            raise RuntimeError(
                "EVOKE interactive rollout expects 36 camera slots per chunk"
            )
        value = self._inputs.get()
        if isinstance(value, _Stop):
            raise _InteractiveStopError
        if value.index != chunk_index:
            raise RuntimeError(
                f"interactive input index {value.index} does not match {chunk_index}"
            )
        trajectory = value.trajectory
        if trajectory is not None and self._pose_base is not None:
            trajectory = np.einsum("ij,njk->nik", self._pose_base, trajectory).astype(
                np.float32
            )
        return trajectory, value.prompt

    def publish_chunk(self, chunk_index: int, decoded: Any) -> None:
        """Publish one decoded RGB chunk to the waiting request."""
        value = decoded.detach().float().cpu().numpy()
        if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 3:
            raise RuntimeError(f"unexpected EVOKE decoded shape: {value.shape}")
        frames = np.transpose(value[0], (1, 2, 3, 0))
        if float(frames.min()) < -0.05:
            frames = (frames + 1.0) * 127.5
        elif float(frames.max()) <= 1.5:
            frames = frames * 255.0
        frames = np.clip(frames, 0.0, 255.0).round().astype(np.uint8)
        self._results.put(_GeneratedChunk(chunk_index, np.ascontiguousarray(frames)))

    def generate(
        self, chunk_index: int, trajectory: np.ndarray | None, prompt: str
    ) -> np.ndarray:
        """Submit one chunk condition and wait for its decoded frames."""
        self._inputs.put(_ChunkInput(chunk_index, trajectory, prompt))
        result = self._results.get()
        if isinstance(result, Exception):
            raise result
        if result.index != chunk_index:
            raise RuntimeError(
                f"interactive output index {result.index} does not match {chunk_index}"
            )
        return result.frames

    def fail(self, error: Exception) -> None:
        """Wake a request waiting on a failed rollout."""
        with suppress(queue.Full):
            self._results.put_nowait(error)

    def stop(self) -> None:
        """Wake a rollout waiting at the next native chunk boundary."""
        with suppress(queue.Full):
            self._inputs.put_nowait(self._stop)


class EvokeRuntime:
    """Keep the post-distillation model and one autoregressive rollout resident."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings = settings
        source_path = Path(settings["source_path"]).resolve()
        adapter_path = Path(__file__).resolve().parent
        sys.path[:] = [
            str(source_path),
            *(
                entry
                for entry in sys.path
                if entry and Path(entry).resolve() not in {source_path, adapter_path}
            ),
        ]
        self._runtime_root = Path(settings["runtime_root"]).resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._default_image = Path(settings["default_image"]).resolve()
        self._stability_prompt = str(settings["stability_prompt"]).strip()
        if not self._stability_prompt:
            raise ValueError("EVOKE stability prompt must not be empty")
        self._max_chunks = int(settings["max_chunks"])
        self._reference_seconds = float(settings["reference_seconds"])
        self._active_seed = int(settings["seed"])
        self._session: _InteractiveSession | None = None
        self._thread: threading.Thread | None = None
        self._chunk_index = 0
        self._counter = 0
        self._mode = "i2v"

        infer_single = importlib.import_module("scripts.inference.infer_single")
        argv = self._build_load_argv()
        self._args = infer_single.parse_args(argv)
        pipe, transformer, vae, device = infer_single.build_pipe(self._args)
        if "interactive_session" not in inspect.signature(pipe.__call__).parameters:
            raise RuntimeError(
                "EVOKE source is missing the Reactor stateful rollout patch"
            )
        self._pipe = pipe
        self._transformer = transformer
        self._vae = vae
        self._device = device
        self._torch = importlib.import_module("torch")

    def reset(
        self,
        *,
        mode: str,
        media: Path | None,
        pose: Path | None,
        prompt: str,
        seed: int,
        source_fps: int,
        source_height: int,
        source_width: int,
    ) -> None:
        """Start a fresh i2v, v2v, or t2v rollout."""
        if mode not in {"i2v", "v2v", "t2v"}:
            raise ValueError(f"unsupported EVOKE mode: {mode}")
        prompt = prompt.strip() or self._stability_prompt
        if mode == "i2v" and (media is None or not media.is_file()):
            raise FileNotFoundError("i2v mode requires an image")
        if mode == "v2v" and (
            media is None or not media.is_file() or pose is None or not pose.is_file()
        ):
            raise FileNotFoundError(
                "v2v mode requires a reference video and pose track"
            )
        self._stop_session()
        self._active_seed = int(seed)
        self._mode = mode
        self._chunk_index = 0
        prepared = self._prepare_conditioning(
            mode,
            media,
            pose,
            source_fps,
            source_height,
            source_width,
        )
        session = _InteractiveSession(prepared["pose_base"])
        self._session = session

        def run() -> None:
            try:
                self._run_pipeline(session, prompt, prepared)
            except _InteractiveStopError:
                return
            except Exception as error:  # noqa: BLE001 - forward rollout failures to the request thread
                traceback.print_exc()
                session.fail(error)
            finally:
                self._clear_rollout_caches()

        self._thread = threading.Thread(
            target=run, name="evoke-stateful-rollout", daemon=True
        )
        self._thread.start()

    def generate(self, trajectory: list[Any] | None, seed: int, prompt: str) -> Path:
        """Generate one native chunk while preserving all upstream rollout state."""
        if seed != self._active_seed:
            raise ValueError("EVOKE seed can change only at reset")
        session = self._session
        thread = self._thread
        if session is None or thread is None or not thread.is_alive():
            raise RuntimeError("EVOKE interactive rollout is not running")
        poses = None if trajectory is None else np.asarray(trajectory, dtype=np.float32)
        frames = session.generate(self._chunk_index, poses, prompt.strip())
        self._chunk_index += 1
        self._counter += 1
        output = self._runtime_root / "outputs" / f"chunk_{self._counter:06d}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, frames, allow_pickle=False)
        return output

    def end_session(self) -> None:
        """Release autoregressive and geometric state without unloading weights."""
        self._stop_session()

    def shutdown(self) -> None:
        """Stop the active rollout before process exit."""
        self._stop_session()

    def _prepare_conditioning(
        self,
        mode: str,
        media: Path | None,
        pose: Path | None,
        source_fps: int,
        source_height: int,
        source_width: int,
    ) -> dict[str, Any]:
        torch = self._torch
        if mode == "t2v":
            return {
                "image": None,
                "video": None,
                "Ks": None,
                "c2ws": None,
                "pose_base": None,
            }
        if mode == "i2v":
            from PIL import Image

            from evoke.utils.ev_validation import load_pose_for_v2v

            assert media is not None
            image = (
                Image.open(media)
                .convert("RGB")
                .resize((640, 384), Image.Resampling.LANCZOS)
            )
            template_pose = Path(self._settings["default_image"]).parent / "pose.npz"
            Ks, _ = load_pose_for_v2v(
                str(template_pose),
                target_height=384,
                target_width=640,
                source_resolution=(720, 1280),
                pose_type="vipe",
                num_target_frames=1,
                target_fps=24,
                source_fps=30,
            )
            c2ws = torch.eye(4, dtype=torch.float32).repeat(self._max_chunks * 36, 1, 1)
            return {
                "image": image,
                "video": None,
                "Ks": Ks,
                "c2ws": c2ws,
                "pose_base": np.eye(4, dtype=np.float32),
            }

        from evoke.utils.ev_validation import load_pose_for_v2v, load_ref_video_for_v2v

        assert media is not None and pose is not None
        video = load_ref_video_for_v2v(
            str(media),
            height=384,
            width=640,
            seconds=self._reference_seconds,
            target_fps=24,
            source_fps=source_fps,
            start_seconds=0.0,
        )
        ref_frames = int(video.shape[0])
        Ks, c2ws = load_pose_for_v2v(
            str(pose),
            target_height=384,
            target_width=640,
            source_resolution=(source_height, source_width),
            pose_type="vipe",
            num_target_frames=ref_frames + self._max_chunks * 36,
            target_fps=24,
            source_fps=source_fps,
            pose_extend_mode="clamp",
        )
        pose_base = c2ws[max(0, ref_frames - 1)].cpu().numpy().astype(np.float32)
        return {
            "image": video[0:1].clone(),
            "video": video,
            "Ks": Ks,
            "c2ws": c2ws,
            "pose_base": pose_base,
        }

    def _run_pipeline(
        self,
        session: _InteractiveSession,
        prompt: str,
        prepared: dict[str, Any],
    ) -> None:
        use_geo = self._mode != "t2v"
        if use_geo:
            self._pipe._geo_vsnoise_cfg["vigeo_scale_mode"] = (
                "anchor" if self._mode == "v2v" else "depth_median"
            )
            self._pipe._geo_vsnoise_cfg["warp_patch_drop_seed"] = self._active_seed
        generator = self._torch.Generator(device=self._device).manual_seed(
            self._active_seed
        )
        self._pipe(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
            height=384,
            width=640,
            num_frames=33 * self._max_chunks,
            num_inference_steps=3,
            guidance_scale=1.0,
            image=prepared["image"],
            image_noise_sigma_min=0.0,
            image_noise_sigma_max=0.0,
            video=prepared["video"],
            video_noise_sigma_min=0.0,
            video_noise_sigma_max=0.0,
            lingbot_Ks=prepared["Ks"],
            lingbot_c2ws=prepared["c2ws"],
            use_dynamic_shifting=True,
            time_shift_type="exponential",
            is_keep_x0=True,
            history_sizes=[16, 2, 1],
            is_enable_stage2=True,
            stage2_num_stages=3,
            stage2_num_inference_steps_list=[1, 1, 1],
            vae_decode_type="persistent",
            output_type="np",
            return_dict=True,
            generator=generator,
            use_kv_cache=False,
            stream_output=True,
            use_dmd=False,
            is_amplify_first_chunk=False,
            attention_kwargs={
                "stage2_warp_compression_mode": "fixed_mem",
                "history_visible_token_threshold": 0.1,
            },
            use_geometric_state=use_geo,
            geo_disable_prev_short=False,
            geo_score="v1",
            geo_nearby_k=0,
            geo_select_k=5,
            geo_top_k=5,
            geo_bank_max=None,
            geo_init_k=10,
            chunk_prompts=None,
            interactive_session=session,
        )

    def _clear_rollout_caches(self) -> None:
        with suppress(Exception):
            self._vae.clear_cache()
        with suppress(Exception):
            self._transformer.clear_kv_cache()

    def _stop_session(self) -> None:
        session = self._session
        thread = self._thread
        self._session = None
        self._thread = None
        if session is not None:
            session.stop()
        if thread is not None:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("EVOKE rollout did not stop at its chunk boundary")
        self._clear_rollout_caches()

    def _build_load_argv(self) -> list[str]:
        runtime_root = self._runtime_root
        return [
            "--ckpt_path",
            str(Path(self._settings["base_model"])),
            "--transformer_path",
            str(Path(self._settings["transformer"])),
            "--sample_type",
            "i2v",
            "--prompt",
            self._stability_prompt,
            "--image_path",
            str(self._default_image),
            "--output_folder",
            str(runtime_root / "unused"),
            "--height",
            "384",
            "--width",
            "640",
            "--num_frames",
            "33",
            "--num_inference_steps",
            "3",
            "--guidance_scale",
            "1.0",
            "--is_enable_stage2",
            "--stage2_num_stages",
            "3",
            "--stage2_steps",
            "1",
            "1",
            "1",
            "--stage2_warp_compression_mode",
            "fixed_mem",
            "--vae_decode_type",
            "persistent",
            "--no_raw_sink_frames",
            "--use_geometric_state",
            "--visibility_aware_noise",
            "--warp_noise_sigma_invisible",
            "1.0",
            "--warp_noise_sigma_min",
            "0.0",
            "--warp_noise_sigma_max",
            "0.135",
            "--visible_token_threshold",
            "0.5",
            "--prefix_idx_mode",
            "zero",
            "--warp_rope_mode",
            "overlap_noise",
            "--geo_recon_backend",
            "da3",
            "--geo_cloud_update_n",
            "12",
            "--geo_depth_backend",
            "vigeo",
            "--geo_vigeo_weights",
            str(Path(self._settings["vigeo_path"])),
            "--geo_vigeo_mode",
            "chunk",
            "--geo_vigeo_scale_mode",
            "depth_median",
            "--geo_vigeo_depth_median_target",
            "5",
            "--geo_vigeo_anchor_windows",
            "4",
            "--geo_vigeo_cache_keep_frames",
            "6",
            "--geo_vigeo_intr_source",
            "gt",
            "--geo_da3_render_mode",
            "backward_zbuf",
            "--geo_bw_fill_iters",
            "12",
            "--geo_warp_stage0_only",
            "--image_noise_sigma_min",
            "0.0",
            "--image_noise_sigma_max",
            "0.0",
        ]


def _respond(request_id: int, *, ok: bool, **payload: object) -> None:
    print(
        _RESPONSE_PREFIX + json.dumps({"id": request_id, "ok": ok, **payload}),
        flush=True,
    )


def main() -> None:
    """Read JSON-line commands until shutdown or stdin closes."""
    runtime: EvokeRuntime | None = None
    for line in sys.stdin:
        request: dict[str, Any] = json.loads(line)
        request_id = int(request["id"])
        command = str(request["command"])
        try:
            if command == "shutdown":
                if runtime is not None:
                    runtime.shutdown()
                _respond(request_id, ok=True)
                return
            if command == "initialize":
                runtime = EvokeRuntime(request)
                _respond(request_id, ok=True)
                continue
            if runtime is None:
                raise RuntimeError("EVOKE worker is not initialized")
            if command == "reset":
                runtime.reset(
                    mode=str(request["mode"]),
                    media=Path(request["media"]) if request.get("media") else None,
                    pose=Path(request["pose"]) if request.get("pose") else None,
                    prompt=str(request["prompt"]),
                    seed=int(request["seed"]),
                    source_fps=int(request["source_fps"]),
                    source_height=int(request["source_height"]),
                    source_width=int(request["source_width"]),
                )
                _respond(request_id, ok=True)
            elif command == "generate":
                output = runtime.generate(
                    request.get("trajectory"),
                    int(request["seed"]),
                    str(request["prompt"]),
                )
                _respond(request_id, ok=True, output=str(output))
            elif command == "end_session":
                runtime.end_session()
                _respond(request_id, ok=True)
            else:
                raise ValueError(f"unknown EVOKE worker command: {command}")
        except Exception as error:  # noqa: BLE001 - serialize command failures for the parent process
            traceback.print_exc()
            _respond(request_id, ok=False, error=f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
