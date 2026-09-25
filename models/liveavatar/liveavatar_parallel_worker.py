"""One native TPP rank; only rank four delivers decoded clips to Runtime."""

import ctypes
import logging
import os
import signal
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np


def run_worker(rank, parent_pid, directory, base, lora, commands, results, ack):
    # Linux parent-death signal also prevents GPU orphans after abrupt Runtime exit.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent_pid:
        return
    try:
        import torch
        import torch.distributed as dist
        import yaml

        from liveavatar_assets import SOURCE, configure_cache_environment
        from liveavatar_turbo import install_dit_compile, turbo_plan

        configure_cache_environment()
        plan = turbo_plan()
        output_rank = plan["output_rank"]
        torch.set_num_threads(4)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method=f"file://{directory}/nccl-init",
            rank=rank,
            world_size=plan["world_size"],
            timeout=timedelta(seconds=120),
            device_id=torch.device(f"cuda:{rank}"),
        )
        sys.path.insert(0, str(SOURCE))
        # Turbo compiles the DiT before the pinned decorators are imported.
        if plan["compile"]:
            install_dit_compile()
        from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS

        from liveavatar_streaming import StreamingWanS2V

        model = StreamingWanS2V(
            config=WAN_CONFIGS["s2v-14B"],
            checkpoint_dir=base,
            device_id=rank,
            rank=rank,
            single_gpu=False,
            sp_size=1,
            convert_model_dtype=True,
            init_on_cpu=False,
            offload_kv_cache=False,
        )
        settings = yaml.safe_load(
            (SOURCE / "liveavatar/configs/s2v_causal_sft.yaml").read_text()
        )
        model.noise_model = model.add_lora_to_model(
            model.noise_model,
            lora_rank=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
            lora_target_modules=settings["lora_target_modules"],
            init_lora_weights=settings["init_lora_weights"],
            pretrained_lora_path=str(Path(lora) / "liveavatar.safetensors"),
            load_lora_weight_only=False,
        )
        from liveavatar_acceleration import install_fa4

        counts = install_fa4()
        dist.barrier()
        if rank == output_rank:
            results.put(("ready",))
        while True:
            # User uploads may take arbitrarily long; no NCCL collective may
            # remain pending while a rank waits for the next interactive take.
            job = commands.get()
            model.vae.model.clear_cache()
            model.vae.model.first_decode = True
            model.vae.model.first_encode = True
            blocks = []
            previous = time.perf_counter()
            chunk = 0

            def emit(clip, blocks=blocks):
                nonlocal previous, chunk
                video = (
                    ((clip.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                )
                blocks.append(video.permute(1, 2, 3, 0).contiguous())
                if len(blocks) != 4:
                    return
                video = torch.cat(blocks).cpu().numpy()
                blocks.clear()
                chunk += 1
                np.save(Path(directory) / "chunk.npy", video, allow_pickle=False)
                now = time.perf_counter()
                logging.getLogger(__name__).warning(
                    "TPP clip=%s frames=%s build_seconds=%.3f realtime_x=%.3f",
                    chunk,
                    len(video),
                    now - previous,
                    len(video) / 25 / (now - previous),
                )
                results.put(("chunk", len(video)))
                ack.get()
                previous = time.perf_counter()

            model.generate(
                input_prompt=job["prompt"],
                ref_image_path=job["image"],
                audio_path=job["audio"],
                pose_video=job["pose"],
                n_prompt=job["negative_prompt"],
                num_repeat=job["max_chunks"],
                max_repeat=job["max_chunks"],
                max_area=704 * 384,
                infer_frames=48,
                sampling_steps=plan["sampling_steps"],
                sample_solver="euler",
                shift=3.0,
                guide_scale=0,
                seed=job["seed"],
                offload_model=False,
                num_gpus_dit=plan["num_gpus_dit"],
                enable_vae_parallel=True,
                chunk_callback=emit,
            )
            logging.getLogger(__name__).warning(
                "TPP rank=%s cumulative_attention_counts=%s", rank, counts
            )
            model.kv_cache1 = model.crossattn_cache = None
            model._sampler_timesteps = model._sampler_sigmas = None
            dist.barrier()
            if rank == output_rank:
                results.put(("end",))
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        results.put(("error", f"rank {rank}: {error}"))
        raise
