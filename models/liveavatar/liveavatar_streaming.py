# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F841, RUF059, TRY201 -- retain upstream generation bookkeeping.
"""Streaming TPP adapter for pinned LiveAvatar source c3c47d031d8bf3247428333c1fe610579c71a551.

The generate method follows the upstream causal_s2v_pipeline_tpp.WanS2V
implementation, with explicit stage ownership and per-block delivery.
Weight loading, conditioning helpers and cache allocation are inherited.
"""

import gc
import os
import random
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
from liveavatar.utils.router.utils import process_masks_to_routing_logits
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from liveavatar_stage_packing import StageRandomStreams, stage_groups


class StreamingWanS2V(WanS2V):
    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=1,
        pose_video=None,
        generate_size=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
        use_dataset=False,
        dataset_sample_idx=0,
        drop_motion_noisy=False,
        num_gpus_dit=4,
        max_repeat=1000000,
        enable_vae_parallel=False,
        mask=None,
        input_video_for_sam2=None,
        enable_online_decode=False,
        chunk_callback=None,
    ):
        """
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            pose_video ('str'):
                If provided, uses a sequence of poses to drive the generated video
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        output_rank = num_gpus_dit - 1 + int(enable_vae_parallel)
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None,
        )
        HEIGHT, WIDTH = size
        channel = 3
        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()
        ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
        if enable_tts is True:
            audio_path = self.tts(tts_prompt_audio, tts_prompt_text, tts_text)
        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()
        if "+" in audio_path:
            audio_paths = audio_path.split("+")
            audio_embs = []
            nr_list = []
            for path in audio_paths:
                audio_emb_i, nr_i = self.encode_audio(path, infer_frames=infer_frames)
                audio_embs.append(audio_emb_i)
                nr_list.append(nr_i)
            min_frames = min(emb.shape[-1] for emb in audio_embs)
            audio_embs = [emb[..., :min_frames] for emb in audio_embs]
            nr = min(nr_list)
            audio_emb = torch.cat(audio_embs, dim=0)
            print(f"rank {dist.get_rank()} processing SAM2")
            input_video_for_sam2 = (
                input_video_for_sam2
                if input_video_for_sam2 is not None
                else ref_image_path
            )
            routing_logits = None
            rank = dist.get_rank()
            if rank == 0:
                video_path_bytes = input_video_for_sam2.encode("utf-8")
                path_length = torch.tensor(
                    [len(video_path_bytes)], dtype=torch.long, device=self.device
                )
            else:
                path_length = torch.tensor([0], dtype=torch.long, device=self.device)
            dist.broadcast(path_length, src=0)
            if rank == 0:
                path_tensor = torch.ByteTensor(list(video_path_bytes)).to(self.device)
            else:
                path_tensor = torch.zeros(
                    path_length.item(), dtype=torch.uint8, device=self.device
                )
            dist.broadcast(path_tensor, src=0)
            video_path = path_tensor.cpu().numpy().tobytes().decode("utf-8")
            print(f"Rank {rank}: video_path: {video_path}")
            parent_dir = os.path.dirname(video_path)
            sam2_output_base = parent_dir
            if rank == 0:
                sam2_cmd = [
                    "python",
                    "liveavatar/utils/router/sam2_tools.py",
                    "--video_folder",
                    video_path,
                    "--output_path",
                    sam2_output_base,
                ]
                try:
                    subprocess.run(sam2_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Rank {rank}: SAM2 processing failed: {e}")
                    raise e
                dist.barrier()
            else:
                dist.barrier()
            base_name = os.path.basename(video_path).split(".")[0]
            tracking_mask_results_dir = os.path.join(
                sam2_output_base, base_name, "tracking_mask_results"
            )
            print(f"Rank {rank}: Looking for masks in: {tracking_mask_results_dir}")
            target_shape = (1, infer_frames // 4, HEIGHT // 8, WIDTH // 8)
            routing_logits = process_masks_to_routing_logits(
                tracking_mask_results_dir, shape=target_shape
            )
            num_actors = routing_logits.shape[-1]
            routing_logits = routing_logits.reshape(
                1, infer_frames // 4, HEIGHT // 8 // 2, WIDTH // 8 // 2, num_actors
            )
            routing_logits = routing_logits.to(
                device=self.device, dtype=self.param_dtype
            )
            mask = routing_logits.permute(4, 1, 2, 3, 0)

            def dilate_mask_by_ratio(
                mask_tensor: torch.Tensor, ratio: float = 0.3, thr: float = 0.5
            ) -> torch.Tensor:
                A, T, H, W, _ = mask_tensor.shape
                out = torch.zeros_like(mask_tensor)
                bin_mask = (mask_tensor > thr).to(dtype=mask_tensor.dtype)
                for a in range(A):
                    for t in range(T):
                        m2d = bin_mask[a, t, :, :, 0]
                        if m2d.any():
                            ys, xs = torch.where(m2d)
                            box_h = int(ys.max() - ys.min() + 1)
                            box_w = int(xs.max() - xs.min() + 1)
                            radius = max(1, int(ratio * max(box_h, box_w) + 0.9999))
                            k = 2 * radius + 1
                            x = m2d[None, None, :, :]
                            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
                            out[a, t, :, :, 0] = (x[0, 0] > 0).to(mask_tensor.dtype)
                        else:
                            out[a, t, :, :, 0] = m2d
                return out

            mask = dilate_mask_by_ratio(mask, ratio=0.1, thr=0.5)
            mask_bool = mask > 0.5
            total_count = mask_bool.sum(dim=0, keepdim=True)
            others_present = total_count - mask_bool.to(total_count.dtype) > 0
            mask = (~others_present).to(dtype=mask.dtype)
            m = (mask[0][0].detach().to(torch.float16).cpu().numpy() > 0.5).astype(
                np.uint8
            ) * 255
            Image.fromarray(m.squeeze()).save("tmp/mask/mask.png")
        else:
            audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        self.audio_encoder.model.to("cpu")
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr
        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))
        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))
        drop_first_motion = False
        motion_latents = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
        videos_last_frames = motion_latents.detach()
        motion_latents = torch.stack(self.vae.encode(motion_latents))
        if drop_motion_noisy:
            zero_motion_latents = torch.zeros_like(motion_latents)
        COND = self.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=size,
        )
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        context, context_null = self.encode_prompt(
            input_prompt, n_prompt, offload_model
        )
        dataset_info = {}
        print("complete prepare conditional inputs")
        if sample_solver == "euler":
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=3
            )
        else:
            raise NotImplementedError("Unsupported solver.")
        self._initialize_comm_group(
            num_gpus_dit=num_gpus_dit, enable_vae_parallel=enable_vae_parallel
        )
        in_dit_device = dist.get_rank() < num_gpus_dit
        owned_steps = (
            stage_groups(dist.get_world_size())[dist.get_rank()]
            if in_dit_device
            else []
        )
        self._stage_kv = {}
        self._stage_cross = {}
        stage_random = StageRandomStreams(owned_steps)
        dist.barrier()
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            out = []
            self.kv_cache1 = None
            active_nr = min(max_repeat, num_repeat)
            for r in range(active_nr):
                if r == 0 or in_dit_device:
                    seed_g = torch.Generator(device=self.device)
                    seed_g.manual_seed(seed + r)
                    lat_target_frames = (
                        infer_frames + 3 + self.motion_frames
                    ) // 4 - lat_motion_frames
                    target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                    frame_seq_length = HEIGHT // 8 * WIDTH // 8 // 2 // 2
                    clip_noise = [
                        torch.randn(
                            16,
                            target_shape[0],
                            target_shape[1],
                            target_shape[2],
                            dtype=self.param_dtype,
                            device=self.device,
                            generator=seed_g,
                        )
                    ]
                    clip_output = torch.zeros_like(clip_noise[0])
                    max_seq_len = np.prod(target_shape) // 4
                    if self.kv_cache1 is None:
                        local_rank = torch.distributed.get_rank()
                        if local_rank < num_gpus_dit:
                            self._initialize_kv_cache(
                                batch_size=1,
                                dtype=self.param_dtype,
                                device=f"cuda:{local_rank}",
                                kv_cache_size=max_seq_len,
                            )
                        self._initialize_crossattn_cache(
                            batch_size=1, dtype=self.param_dtype, device=self.device
                        )
                if r == 0 or in_dit_device:
                    clip_latents = deepcopy(clip_noise)
                    with torch.no_grad():
                        left_idx = r * infer_frames
                        right_idx = r * infer_frames + infer_frames
                        cond_latents = COND[r] if pose_video else COND[0] * 0
                        cond_latents = cond_latents.to(
                            dtype=self.param_dtype, device=self.device
                        )
                        audio_input = audio_emb[..., left_idx:right_idx]
                    input_motion_latents = motion_latents.clone()
                if (r == 0 or r == 1) and in_dit_device:
                    if r == 1:
                        ref_latents = torch.empty_like(ref_latents).type_as(
                            clip_latents[0]
                        )
                        dist.broadcast(ref_latents, src=output_rank)
                    block_index = 0
                    block_latents = clip_latents[0][
                        :,
                        block_index * self.num_frames_per_block : (block_index + 1)
                        * self.num_frames_per_block,
                    ]
                    left_idx = block_index * (self.num_frames_per_block * 4)
                    right_idx = (block_index + 1) * (self.num_frames_per_block * 4)
                    block_arg_c = {
                        "context": context[0:1],
                        "seq_len": None,
                        "cond_states": cond_latents[
                            :,
                            :,
                            block_index * self.num_frames_per_block : (block_index + 1)
                            * self.num_frames_per_block,
                        ],
                        "motion_latents": input_motion_latents,
                        "ref_latents": ref_latents,
                        "audio_input": audio_input[..., left_idx:right_idx],
                        "motion_frames": [self.motion_frames, lat_motion_frames],
                        "drop_motion_frames": drop_first_motion and r == 0,
                        "sink_flag": True,
                    }
                    timestep = (
                        torch.ones(
                            [1, self.num_frames_per_block],
                            device=self.device,
                            dtype=self.param_dtype,
                        )
                        * 0
                    )
                    if not self._stage_kv:
                        for stage in owned_steps:
                            self._stage_kv[stage] = deepcopy(self.kv_cache1)
                            self._stage_cross[stage] = deepcopy(self.crossattn_cache)
                        self.kv_cache1 = self._stage_kv[owned_steps[0]]
                        self.crossattn_cache = self._stage_cross[owned_steps[0]]
                    for stage in owned_steps:
                        stage_random.call(
                            stage,
                            self.noise_model,
                            [block_latents],
                            t=timestep * 0,
                            **block_arg_c,
                            kv_cache=self._stage_kv[stage],
                            crossattn_cache=self._stage_cross[stage],
                            current_start=block_index
                            * self.num_frames_per_block
                            * frame_seq_length,
                            current_end=(block_index + 1)
                            * self.num_frames_per_block
                            * frame_seq_length,
                        )
                num_blocks = target_shape[0] // self.num_frames_per_block
                for block_index in range(num_blocks):
                    if getattr(self, "_sampler_timesteps", None) is None:
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=self.device
                        )
                        self._sampler_timesteps = sample_scheduler.timesteps
                        self._sampler_sigmas = sample_scheduler.sigmas
                    timesteps = self._sampler_timesteps
                    sample_scheduler.timesteps = timesteps
                    sample_scheduler.sigmas = self._sampler_sigmas
                    sample_scheduler._step_index = dist.get_rank()
                    sample_scheduler._begin_index = 0
                    block_latents = clip_latents[0][
                        :,
                        block_index * self.num_frames_per_block : (block_index + 1)
                        * self.num_frames_per_block,
                    ]
                    if r == 0 or in_dit_device:
                        left_idx = block_index * (self.num_frames_per_block * 4)
                        right_idx = (block_index + 1) * (self.num_frames_per_block * 4)
                        block_arg_c = {
                            "context": context[0:1],
                            "seq_len": None,
                            "cond_states": cond_latents[
                                :,
                                :,
                                block_index * self.num_frames_per_block : (
                                    block_index + 1
                                )
                                * self.num_frames_per_block,
                            ],
                            "motion_latents": input_motion_latents,
                            "ref_latents": ref_latents,
                            "audio_input": audio_input[..., left_idx:right_idx],
                            "motion_frames": [self.motion_frames, lat_motion_frames],
                            "drop_motion_frames": drop_first_motion and r == 0,
                        }
                    for i, t in enumerate(tqdm(timesteps)):
                        if i not in owned_steps:
                            continue
                        sample_scheduler._step_index = i
                        if self.src_gpu is None or i != owned_steps[0]:
                            latent_model_input = block_latents
                        else:
                            latent_model_input = torch.empty_like(block_latents)
                            dist.recv(latent_model_input, self.src_gpu)
                        timestep = [t] * self.num_frames_per_block
                        timestep = torch.tensor(timestep).to(self.device).unsqueeze(0)
                        noise_pred_cond = stage_random.call(
                            i,
                            self.noise_model,
                            [latent_model_input],
                            t=timestep,
                            **block_arg_c,
                            kv_cache=self._stage_kv[i],
                            crossattn_cache=self._stage_cross[i],
                            current_start=block_index
                            * self.num_frames_per_block
                            * frame_seq_length
                            + r
                            * num_blocks
                            * self.num_frames_per_block
                            * frame_seq_length,
                            current_end=(block_index + 1)
                            * self.num_frames_per_block
                            * frame_seq_length
                            + r
                            * num_blocks
                            * self.num_frames_per_block
                            * frame_seq_length,
                            mask=mask,
                        )
                        noise_pred = [torch.cat(noise_pred_cond, dim=0)]
                        temp_x0 = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0),
                            t,
                            latent_model_input.unsqueeze(0),
                            return_dict=False,
                            generator=seed_g,
                        )[0]
                        block_latents = temp_x0.squeeze(0)
                        if self.tgt_gpu is None or i != owned_steps[-1]:
                            pass
                        else:
                            dist.send(block_latents.contiguous(), self.tgt_gpu)
                    if enable_vae_parallel and dist.get_rank() == output_rank:
                        vae_wait_start = time.time()
                        block_latents = torch.empty_like(block_latents)
                        dist.recv(block_latents, self.src_gpu)
                        torch.cuda.synchronize()
                        if time.time() - vae_wait_start < 0.01:
                            print("WARNING: VAE serves as a bottleneck!")
                    if enable_vae_parallel and dist.get_rank() == output_rank:
                        if offload_model:
                            print("offloading model to cpu")
                            self.noise_model.cpu()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        if r == 0 and active_nr != 1:
                            if block_index == 0:
                                ref_latents = block_latents.unsqueeze(0)[:, :, 0:1]
                            elif block_index == num_blocks - 1:
                                dist.broadcast(
                                    ref_latents.contiguous(), src=output_rank
                                )
                            else:
                                pass
                        if r == 0 and block_index == 0:
                            decode_latents = motion_latents[:, :, :7]
                            self.vae.stream_decode(decode_latents)
                        decode_latents = block_latents.unsqueeze(0)
                        image = torch.stack(self.vae.stream_decode(decode_latents))
                        image = image[:, :, -infer_frames // num_blocks :]
                        if r == 0 and block_index == 0:
                            image = image[:, :, 3:]
                        if chunk_callback is None:
                            out.append(image.cpu())
                        else:
                            chunk_callback(image[0])
        if dist.is_initialized():
            dist.barrier()
        if dist.get_rank() == output_rank:
            videos = torch.cat(out, dim=2) if chunk_callback is None else None
            del clip_noise, clip_latents, clip_output, block_latents
            self._sampler_timesteps = None
            self._sampler_sigmas = None
            if offload_model:
                gc.collect()
                torch.cuda.synchronize()
            return (videos[0] if videos is not None else None, dataset_info)
        else:
            return (None, dataset_info)
