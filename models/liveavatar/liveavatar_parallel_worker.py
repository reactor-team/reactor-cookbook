"""Native TPP setup and continuous inference inside Runtime-owned ranks."""

import sys


def load_rank(rank, deployment):
    import torch
    import yaml

    from liveavatar_turbo import install_dit_compile, turbo_plan

    plan = turbo_plan(deployment.turbo)
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    sys.path.insert(0, str(deployment.source))
    # Turbo compiles the DiT before the pinned decorators are imported.
    if plan["compile"]:
        install_dit_compile()
    from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS

    from liveavatar_streaming import StreamingWanS2V

    model = StreamingWanS2V(
        config=WAN_CONFIGS["s2v-14B"],
        checkpoint_dir=str(deployment.base),
        device_id=rank,
        rank=rank,
        single_gpu=False,
        sp_size=1,
        convert_model_dtype=True,
        init_on_cpu=False,
        offload_kv_cache=False,
    )
    settings = yaml.safe_load(
        (deployment.source / "liveavatar/configs/s2v_causal_sft.yaml").read_text()
    )
    model.noise_model = model.add_lora_to_model(
        model.noise_model,
        lora_rank=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"],
        lora_target_modules=settings["lora_target_modules"],
        init_lora_weights=settings["init_lora_weights"],
        pretrained_lora_path=str(deployment.lora / "liveavatar.safetensors"),
        load_lora_weight_only=False,
    )
    from liveavatar_acceleration import install_fa4

    install_fa4()

    return model, plan


def generate_take(model, plan, rank, job, deliver):
    import torch

    # CUDA's current device is thread-local.
    torch.cuda.set_device(rank)
    model.vae.model.clear_cache()
    model.vae.model.first_decode = True
    model.vae.model.first_encode = True
    blocks = []

    def emit(clip):
        video = ((clip.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        blocks.append(video.permute(1, 2, 3, 0).contiguous())
        if len(blocks) == 4:
            frames = torch.cat(blocks).cpu().numpy()
            blocks.clear()
            deliver(frames)

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
    model.kv_cache1 = model.crossattn_cache = None
    model._stage_kv = {}
    model._stage_cross = {}
    model.vae.model.clear_cache()
    model._sampler_timesteps = model._sampler_sigmas = None
