"""Call the pinned upstream GPU worker in the process owned by DistributedRunner."""

from __future__ import annotations

import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from inworld_model import (
    CHUNK_FRAMES,
    FPS,
    HEIGHT,
    PREFIX_FRAMES,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    WIDTH,
    InWorldAnchor,
    InWorldSettings,
)


class InWorldBackend:
    def __init__(
        self, settings: InWorldSettings, *, rank: int = 0, world_size: int = 1
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self._submitted = False
        self._cuda_graph = settings.cuda_graph
        from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )

        self.args = ServerArgs.from_dict(
            {
                "model_path": str(settings.base),
                "model_id": "LTX-2.5-Diffusers",
                "pipeline_class_name": "LTX2Pipeline",
                "transformer_weights_path": str(settings.checkpoint),
                "component_paths": {"connectors": str(settings.connectors)},
                "num_gpus": world_size,
                "ulysses_degree": world_size,
                "attention_backend": settings.attention_backend,
                "enable_torch_compile": False,
                "component_residency": {"vae": "resident"},
                "master_port": settings.master_port,
                "output_path": tempfile.gettempdir(),
            }
        )
        if settings.cuda_graph:
            # The pinned LTX25 implementation inherits LTX2's graph path but is
            # absent from its name allowlist. Enable the validated path explicitly.
            if type(self.args.pipeline_config).__name__ != "LTX25PipelineConfig":
                raise RuntimeError("CUDA Graph requires the pinned LTX25 pipeline")
            self.args.enable_breakable_cuda_graph = True
        set_global_server_args(self.args)
        self.worker = GPUWorker(
            local_rank=rank,
            rank=rank,
            master_port=settings.master_port,
            server_args=self.args,
        )
        install_output_processor(self.worker)

    def warmup(self) -> None:
        """Prepare first and continuation requests for every offered chunk length."""
        for count in (120, *(n for n in CHUNK_FRAMES if n != 120)):
            if self.rank != 0:
                self.follow_step()
                self.follow_step()
                continue
            with tempfile.TemporaryDirectory(prefix="inworld-warmup-") as directory:
                root = Path(directory)
                anchor = InWorldAnchor(seed=0, new_frames=count)
                parent = None
                for index in (1, 2):
                    self.begin_step()
                    try:
                        parent, _, _ = self.generate_clip(
                            root=root,
                            index=index,
                            prompt="A quiet garden with birds singing.",
                            anchor=anchor,
                            parent=parent,
                            references=(),
                            warmup=self._cuda_graph,
                        )
                    except Exception as exc:
                        if self.world_size > 1:
                            self.cancel_unsubmitted_step(str(exc))
                        raise

    def begin_step(self) -> None:
        self._submitted = False

    def _broadcast(self, value):
        import torch.distributed as dist

        values = [value]
        dist.broadcast_object_list(values, src=0, group=self.worker.sp_cpu_group)
        return values[0]

    def cancel_unsubmitted_step(self, message: str) -> None:
        """Release followers if preparing the rank-zero request failed."""
        if not self._submitted:
            self._submitted = True
            self._broadcast(("error", message))

    def follow_step(self) -> None:
        kind, value = self._broadcast(None)
        if kind == "error":
            raise RuntimeError(value)
        output = self.worker.execute_forward(value)
        if output.error:
            raise RuntimeError(output.error)

    def _execute(self, requests):
        if self.world_size > 1:
            self._submitted = True
            self._broadcast(("run", requests))
        return self.worker.execute_forward(requests)

    @staticmethod
    def store_references(
        root: Path, references: tuple[np.ndarray, ...]
    ) -> tuple[Path, ...]:
        paths = []
        for i, rgb in enumerate(references):
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("References must be RGB uint8 HWC arrays")
            path = root / f"reference-{i}.png"
            # The pinned upstream API takes file paths. PNG preserves the decoded pixels;
            # the original upstream image fitting still runs in its image-encoding stage.
            Image.fromarray(rgb).save(path, compress_level=1)
            paths.append(path)
        return tuple(paths)

    def generate_clip(
        self,
        *,
        root: Path,
        index: int,
        prompt: str,
        anchor: InWorldAnchor,
        parent: Path | None,
        references: tuple[Path, ...],
        warmup: bool = False,
    ):
        from murea_fast.client import prepare_tail
        from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
            build_sampling_params,
        )
        from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

        prefix = PREFIX_FRAMES if parent is not None else 0
        tail = root / f"prefix-{index}.mp4"
        if parent is not None:
            cached_tail = parent.with_suffix(".prefix.mp4")
            if cached_tail.is_file():
                tail = cached_tail
            else:
                prepare_tail(parent, tail, prefix)
        params = build_sampling_params(
            f"segment-{index}",
            prompt=prompt,
            seed=anchor.seed,
            width=WIDTH,
            height=HEIGHT,
            num_frames=anchor.new_frames + (prefix or 1),
            fps=FPS,
            num_inference_steps=8,
            guidance_scale=1.0,
            generator_device="cuda",
            video_path=str(tail) if prefix else None,
            continuation_prefix_frames=prefix if prefix else None,
            reference_image_paths=[str(p) for p in references]
            if references and not prefix
            else None,
            output_path=str(root),
            output_file_name=f"segment-{index}",
            save_output=True,
            return_file_paths_only=True,
        )
        req = prepare_request(self.args, params)
        req.is_warmup = warmup
        try:
            params.prepare_video_request_for_queue(req)
            output = self._execute([req])
            if output.error:
                raise RuntimeError(output.error)
            if len(output.output_file_paths) != 1:
                raise RuntimeError("Native inference did not produce exactly one clip")
            path = Path(output.output_file_paths[0])
            frames, audio = decode_clip(
                path, prefix, tail_path=path.with_suffix(".prefix.mp4")
            )
            expected = anchor.new_frames if prefix else anchor.new_frames + 1
            if frames.shape != (expected, HEIGHT, WIDTH, 3):
                raise RuntimeError(f"Unexpected native video shape: {frames.shape}")
            return path, frames, audio
        finally:
            params.cleanup_video_request(req)
            tail.unlink(missing_ok=True)


def decode_clip(
    path: Path, prefix: int, *, tail_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep native MP4 continuation semantics and emit only the new AV interval."""
    tail_frames: deque[av.VideoFrame] = deque(maxlen=PREFIX_FRAMES)
    with av.open(str(path)) as container:
        # Frame threading changes scheduling, not H.264 reconstruction. Bound
        # the worker's CPU use independently of the shared host's core count.
        container.streams.video[0].thread_type = "AUTO"
        container.streams.video[0].thread_count = 4
        decoded = []
        total_frames = 0
        for index, frame in enumerate(container.decode(video=0)):
            total_frames += 1
            if index >= prefix:
                decoded.append(frame)
            if tail_path is not None:
                tail_frames.append(frame)
    if not decoded:
        raise RuntimeError("Native clip has no new video frames")
    frames = np.empty(
        (len(decoded), decoded[0].height, decoded[0].width, 3), dtype=np.uint8
    )

    def convert(index: int) -> None:
        frames[index] = decoded[index].to_ndarray(format="rgb24")

    # Each frame owns its reformatter. Workers write disjoint output slices;
    # waiting for every conversion also propagates decoder failures.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(convert, range(len(decoded))))
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError("Native clip has no audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        blocks = []
        for frame in container.decode(audio=0):
            blocks.extend(f.to_ndarray() for f in resampler.resample(frame))
        blocks.extend(f.to_ndarray() for f in resampler.resample(None))
    if not blocks:
        raise RuntimeError("Native clip has no decoded audio samples")
    audio = np.concatenate(blocks, axis=1)
    start, end = prefix * SAMPLES_PER_FRAME, total_frames * SAMPLES_PER_FRAME
    valid_samples = min(audio.shape[1], end) - start
    if audio.shape[1] < end:
        missing = end - audio.shape[1]
        # The pinned audio pipeline rounds to a 40ms latent grid; its causal
        # decoder emits 4*n-3 mel frames at 10ms each. The resulting trailing
        # deficit is at most 20ms rounding + 30ms causal trimming. Preserve all
        # generated samples and fill only this bounded native gap for transport.
        if missing > SAMPLE_RATE // 20:
            raise RuntimeError(
                f"Native audio is over 50ms shorter than the clip: "
                f"{audio.shape[1]} samples for {total_frames} frames "
                f"({missing} samples missing)"
            )
        audio = np.pad(audio, ((0, 0), (0, missing)))
    if tail_path is not None and len(tail_frames) == PREFIX_FRAMES:
        write_prefix(tail_path, tail_frames)
    audio = np.ascontiguousarray(audio[:, start:end])
    # Five milliseconds smooth the transport seam; native saved audio and the
    # continuation video stay unchanged, as does the exact sample count.
    fade = min(SAMPLE_RATE // 200, valid_samples)
    if prefix:
        audio[:, :fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    audio[:, valid_samples - fade : valid_samples] *= np.linspace(
        1, 0, fade, dtype=np.float32
    )
    return frames, audio


def install_output_processor(worker) -> None:
    """Use the native decoder and encoder with a smaller intermediate transfer."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av import (
        LTX2AVDecodingStage,
    )

    stages = [s for s in worker.pipeline.stages if isinstance(s, LTX2AVDecodingStage)]
    if len(stages) != 1:
        raise RuntimeError("Expected one native LTX-2 audio-video decoding stage")
    stage = stages[0]
    stage.video_processor = Uint8VideoProcessor(stage.video_processor)


class Uint8VideoProcessor:
    """Preserve native quantization while transferring uint8 instead of FP32.

    Native postprocessing denormalizes in the decoder dtype, casts to FP32,
    then clips, multiplies by 255 and truncates before the MP4 encoder. Keep
    that order and encoder; moving quantization does not replace the compressed
    continuation history with uncompressed frames.
    """

    def __init__(self, original) -> None:
        self.original = original

    def postprocess_video(self, video, output_type="np", **kwargs):
        import torch

        if output_type != "np":
            return self.original.postprocess_video(video, output_type, **kwargs)
        normalized = self.original.postprocess_video(video, "pt", **kwargs)
        frames = (normalized.float().clamp(0, 1) * 255.0).to(torch.uint8)
        frames = frames.permute(0, 1, 3, 4, 2).contiguous()
        if frames.device.type == "cuda":
            host = torch.empty(
                frames.shape, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            host.copy_(frames, non_blocking=True)
            torch.cuda.current_stream(frames.device).synchronize()
            return host.numpy()
        return frames.numpy()


def write_prefix(path: Path, frames: deque[av.VideoFrame]) -> None:
    """Preserve the native last-17-frame YUV420 prefix without an RGB round trip.

    The upstream continuation client uses libx264 CRF 0 and yuv420p. Retaining
    the decoded YUV frames avoids decoding the same parent a second time.
    """
    if len(frames) != PREFIX_FRAMES:
        raise ValueError("A continuation prefix requires 17 frames")
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=FPS)
        stream.width, stream.height = frames[0].width, frames[0].height
        stream.pix_fmt = "yuv420p"
        # This is only the lossless continuation prefix, never the generated
        # output's native lossy encode. CRF 0 preserves decoded YUV at any preset.
        stream.options = {"crf": "0", "preset": "ultrafast"}
        stream.thread_count = 4
        for index, frame in enumerate(frames):
            if frame.format.name != "yuv420p":
                raise ValueError("Native continuation frames must be yuv420p")
            frame.pts = index
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
