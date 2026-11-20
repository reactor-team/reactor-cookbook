#!/usr/bin/env python3
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Prove this model directory computes what the upstream JoyAI-Video-Edit runtime computes.

Two runtimes, the same clips, the same frames, the same seed. The upstream
runtime (`deploy/` of https://github.com/jd-opensource/JoyAI-Video-Edit, mounted
read-only) runs with its B200 serving recipe. This directory runs as the
Reactor runtime runs it, minus the
network: the application class (`joy_video_edit.JoyVideoEdit`) is loaded from a
config path, its own step loop runs (`ReactorApp.run` on the model thread), and
each clip is one session opened with the runtime's events (SessionStarted,
ClientConnected), driven by the client's commands (set_prompt, set_seed, start)
validated by the model's contract, fed through the `camera` input buffer
(`push_media`), read by `process_input()` with its configured `camera_read`, and
collected from the media sink the runtime binds. Nothing is overridden unless
`--config-set KEY=VALUE` says so.

Layers exercised on this side: joy_video_edit.yaml, load() and its warm-up and CUDA
graph capture, command validation and handlers, the session state, the input
buffer and the camera read policy, the three step hooks, emit(). Not exercised: WebRTC
(the video encode/decode and the network in both directions), per-connection
playout pacing, uploads (no reference image is sent), and the recording.
Each side runs in its own subprocess (both trees ship a package named
`xvideo`) and plays every clip in order. The script then compares, chunk by
chunk:

  enc     latent of the source window after VAE encode   (input to the DiT)
  dit     denoised latent the DiT returns                 (input to the VAE decode)
  pseudo  latent re-encoded from the last decoded frame   (history for the next decode)
  pixels  output frames as each runtime hands them out (uint8)

and reports the first stage that leaves tolerance in each chunk, so a failure
points at encode, denoise or decode rather than at "the video looks different".

## What is controlled, and why

Every control is applied to BOTH sides identically, and each removes a source
of run-to-run difference that has nothing to do with the code under test:

* **VAE compilation** (`--vae-compile off`, default) -- the VAE is compiled with
  `max-autotune`, which picks conv kernels by timing them; two processes with
  different compile caches can pick different kernels, and this model amplifies
  that bf16-level difference into ~22-33 dB between two runs of the SAME code.
  Off, both sides run the same eager cuDNN convolutions (weights in
  channels_last_3d, the layout the compiled path uses).
* **Attention** (`--attention cudnn`, default) -- upstream uses
  FlashAttention-4, this directory cuDNN SDPA. The upstream side is patched to
  the same cuDNN-only SDPA call. `--attention native` keeps each tree's kernel.
* **VAE posterior** -- `--posterior mean` (default) replaces
  `DiagonalGaussianDistribution.sample()` by the mean. `--posterior sample`
  keeps the draw: it uses the global CUDA RNG, so parity then also requires both
  runtimes to make the same draws in the same order from the same RNG state.
  Upstream draws on its encode and pseudo-encode worker threads; the drive
  waits for the pseudo latent of chunk N to be stored before pushing chunk N+1,
  so the two threads never race for the generator. This directory runs every
  stage of a chunk in order on one thread, so its draws are ordered by construction.
* **Frames** -- each clip is decoded and resized once (PIL bicubic, as the
  session does) and both sides read the same array.
* **Drive** -- `--feed gated` (default): a chunk's frames are pushed, its
  output awaited, then the next, so no frame is dropped and both sides see the
  same frames. `--feed realtime` pushes this directory's frames at
  `--feed-fps` whatever the model does, which exercises the camera read policy
  and can drop frames (then the frame-by-frame comparison is not meaningful).

`--self-check` runs each side a second time in a fresh process and reports the
side-against-itself difference. A tolerance means nothing until both sides meet
it against themselves.

## Tolerances

PASS requires, for every chunk of every clip, `dit` relative L2 error <=
`--tol-latent` (default 1e-2) and every frame PSNR >= `--tol-psnr` dB (default 50).

## Run it inside the model image

    docker run --rm --gpus '"device=0"' --ipc=host \\
      -v <weights>:/weights:ro -v <JoyAI-Video-Edit>/deploy:/upstream:ro \\
      -v <this model dir>:/app:ro -v <clips>:/clips:ro -v <out>:/out \\
      -e PYTHONPATH=/app -w /app --entrypoint python <model image> \\
      scripts/prove_parity.py --upstream /upstream --weights /weights --out /out \\
        --clip /clips/input.mp4 --prompt "Turn the video into a watercolor wash style." \\
        [--clip ... --prompt ...]

It writes `<out>/report.json`, ONE `<out>/parity.mp4` with every clip in order
(source | upstream | this directory, labels, instruction and per-clip PSNR
burned in) and each side's raw captures under `<out>/<side>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this model directory
UPSTREAM_ENV = {  # the upstream launcher's B200 recipe
    "JOYOMNI_FP8_IMG": "1", "JOYOMNI_FP8_TXT": "1", "JOYOMNI_CUDA_GRAPH": "1",
    "JOYOMNI_SAGE_ATTN": "0", "JOYOMNI_FP8_FAST_ACCUM": "0", "JOYOMNI_LOW_VRAM": "0",
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", required=True, help="JoyAI-Video-Edit/deploy directory")
    ap.add_argument("--weights", required=True, help="dir with dit/, vae/, MiMo-VL-7B-RL-2508/")
    ap.add_argument("--dit", default="dit/joyai_video_edit_dit_0811.pth")
    ap.add_argument("--clip", action="append", required=True, help="repeatable; paired with --prompt")
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1248)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=0, help="per clip; 0 = whole clip")
    ap.add_argument("--vae-compile", choices=["off", "on"], default="off")
    ap.add_argument("--attention", choices=["cudnn", "native"], default="cudnn")
    ap.add_argument("--posterior", choices=["mean", "sample"], default="mean")
    ap.add_argument("--cuda-graph", choices=["on", "off"], default="on", help="upstream side only")
    ap.add_argument("--config-set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one joy_video_edit.yaml key for this directory's side (YAML value, repeatable), "
                         "e.g. text_tokens=null; by default the model runs exactly as joy_video_edit.yaml serves it")
    ap.add_argument("--feed", choices=["gated", "realtime"], default="gated",
                    help="gated (default): a chunk's frames are pushed once the previous chunk is emitted, "
                         "so both sides see the same frames; realtime: one frame every 1/--feed-fps")
    ap.add_argument("--feed-fps", type=float, default=24.0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--tol-latent", type=float, default=1e-2)
    ap.add_argument("--tol-psnr", type=float, default=50.0)
    ap.add_argument("--side", choices=["upstream", "integration"], help=argparse.SUPPRESS)
    ap.add_argument("--side-out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if len(args.clip) != len(args.prompt):
        ap.error("--clip and --prompt must be given the same number of times")
    return args


def clip_names(args) -> list[str]:
    return [os.path.splitext(os.path.basename(c))[0] for c in args.clip]


# =============================================================================== one side


def run_side(args) -> None:
    import torch
    from PIL import Image

    t0 = time.time()
    out = args.side_out
    os.makedirs(out, exist_ok=True)
    if args.side == "upstream":
        sys.path.insert(0, args.upstream)
        import xvideo.models.dit.dit as D

        if args.attention == "cudnn":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def _cudnn_attention(query, key, value):
                dtype = query.dtype
                if dtype not in (torch.float16, torch.bfloat16):
                    query, key, value = (t.to(torch.bfloat16) for t in (query, key, value))
                q, k, v = (t.transpose(1, 2) for t in (query, key, value))
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
                return o.transpose(1, 2).to(dtype)

            D._flash_attention4 = _cudnn_attention
            D._FA4_DISABLED = True
            D.attention_backend = lambda: "cudnn (forced)"
        _control_vae_compile(args)
        from xvideo.serving.joyomni_streaming import JoyOmniRuntime, JoyOmniV2VStreamingSession, StreamingSettings

        dev = "cuda:0"
        runtime = JoyOmniRuntime.load(
            os.path.join(args.weights, args.dit), vae_ckpt=os.path.join(args.weights, "vae"),
            text_encoder_ckpt=os.path.join(args.weights, "MiMo-VL-7B-RL-2508"),
            device=dev, vae_encode_device=dev, vae_decode_device=dev, vae_pseudo_device=dev,
            postprocess_device=dev, seed=args.seed, warmup_height=args.height, warmup_width=args.width,
        )

        def make_settings():
            return StreamingSettings(
                height=args.height, width=args.width, num_inference_steps=args.steps, seed=args.seed,
                max_temporal_ids=8, freeze_kv_on_static=True, static_diff_thresh=0.5,
                profile_timings=False, output_codec="h264",
            )
    else:
        sys.path.insert(0, HERE)
        import yaml

        with open(os.path.join(HERE, "joy_video_edit.yaml")) as f:
            cfg = yaml.safe_load(f)
        cfg["checkpoints"]["joyai"]["dit"] = args.dit
        cfg.update({"checkpoint_dir": os.path.abspath(args.weights),
                    "height": args.height, "width": args.width,
                    "warmup_height": args.height, "warmup_width": args.width})
        # dit.py reads the FP8 switches at import time. The model half's load() force-sets them from config
        # before its own xvideo import; the VAE control below imports xvideo first, so set the same
        # values here (load() then sets them again, unchanged, and refuses FP8 if the kernel is absent).
        os.environ["JOYOMNI_FP8_IMG"] = "1" if cfg.get("fp8_img", True) else "0"
        os.environ["JOYOMNI_FP8_TXT"] = "1" if cfg.get("fp8_txt", True) else "0"
        _control_vae_compile(args)
        import joy_video_edit as P

        for kv in args.config_set:
            key, _, val = kv.partition("=")
            cfg[key] = yaml.safe_load(val)
        cfg_path = os.path.join(out, "joy_video_edit.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        if int(cfg.get("num_inference_steps", 2)) != args.steps:
            raise SystemExit(f"config num_inference_steps={cfg.get('num_inference_steps')} but --steps={args.steps}: "
                             "both sides must denoise with the same step count")
        # The Reactor application class itself, loaded as the runtime loads it (a config path).
        from pathlib import Path
        model = P.JoyVideoEdit()
        model.load(Path(cfg_path))
        runtime = model.engine._runtime
        from xvideo.serving.joyomni_streaming import JoyOmniV2VStreamingSession
    load_s = time.time() - t0
    import xvideo.models.dit.dit as D

    backend = D.attention_backend()
    import xvideo.models.vae.vae as V
    if args.posterior == "mean":
        V.DiagonalGaussianDistribution.sample = lambda self, generator=None: self.mode()
    for vae in {id(v): v for v in (runtime.pipeline.vae, runtime.decode_vae, runtime.pseudo_encode_vae)}.values():
        for m in vae.modules():
            if isinstance(m, torch.nn.Conv3d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)

    cap: dict[str, np.ndarray] = {}

    def wrap(name, fn):
        orig = getattr(JoyOmniV2VStreamingSession, name)

        def w(self, *a, **k):
            r = orig(self, *a, **k)
            fn(a, k, r)
            return r
        setattr(JoyOmniV2VStreamingSession, name, w)

    def lat(key):
        return lambda a, k, r: cap.__setitem__(f"{key}_{int(k['chunk_idx']):03d}", r.detach().float().cpu().numpy())
    wrap("_encode_reference_chunk", lat("enc"))
    wrap("_denoise_chunk", lat("dit"))
    wrap("_store_decode_pseudo_latent",
         lambda a, k, r: cap.__setitem__(f"pseudo_{int(a[0]):03d}", a[1].detach().float().cpu().numpy()))

    info = {"side": args.side, "attention": backend, "load_s": load_s, "clips": {}}
    if args.side == "integration":
        _drive_reactor_model(args, model, cap, info, out)
    for name in (clip_names(args) if args.side == "upstream" else []):
        prompt = args.prompt[clip_names(args).index(name)]
        frames = np.load(os.path.join(args.out, "frames", f"{name}.npy"), mmap_mode="r")
        session = runtime.create_v2v_session(prompt, settings=make_settings())
        ft = int(session.ffactor_t)
        outs: dict[int, list] = {}
        lat_s: dict[int, float] = {}
        state = {"sub": 0, "rec": 0}

        def collect(results):
            for r in results:
                idx = int(r.profile["chunk_idx"])
                outs[idx] = [np.asarray(x) for x in r.jpegs]
                lat_s[idx] = time.perf_counter() - lat_s[idx]
                state["rec"] += 1

        for i in range(len(frames)):
            completes = i == 0 or (i - 1) % ft == ft - 1
            if completes:
                lat_s[state["sub"]] = time.perf_counter()
            collect(session.push_frame(Image.fromarray(np.ascontiguousarray(frames[i])),
                                       {"seq": i + 1, "t_capture_ms": 0.0}))
            if completes:
                state["sub"] += 1
                while state["rec"] < state["sub"]:
                    r = session.wait_async_result(timeout=0.5)
                    if r is not None:
                        collect([r])
                # upstream draws the pseudo latent for the next chunk on its own thread; let it land
                # before the next encode draws, so the RNG is consumed in one fixed order (this
                # directory returns from push_frame with the chunk, pseudo latent included)
                deadline = time.time() + 60
                while session._pseudo_latent_chunk_idx != state["sub"] and time.time() < deadline:
                    time.sleep(0.001)
        session.close()
        np.savez_compressed(os.path.join(out, f"{name}_latents.npz"), **cap)
        cap.clear()
        np.savez_compressed(os.path.join(out, f"{name}_frames.npz"),
                            frames=np.concatenate([np.stack(outs[k]) for k in sorted(outs)]))
        info["clips"][name] = {"ffactor_t": ft, "chunk_latency_s": [lat_s[k] for k in sorted(lat_s)],
                               "dit_text_tokens": [getattr(session, "text_tokens", None), None
                                                   if getattr(session, "streaming_cond_embeds", None) is None
                                                   else int(session.streaming_cond_embeds.shape[1])],
                               "graph_chunks": getattr(session, "graph_chunks", None)}
    _write_side_info(args, runtime, info, out)


def _write_side_info(args, runtime, info, out) -> None:
    import torch

    tf = runtime.pipeline.transformer
    blocks = [getattr(b, "_orig_mod", b) for b in tf.double_blocks]
    info.update({
        "fp8_img_blocks": sum(bool(getattr(b, "_fp8_img_installed", False)) for b in blocks),
        "fp8_txt_blocks": sum(bool(getattr(b, "_fp8_txt_installed", False)) for b in blocks),
        "compiled_dit_blocks": sum(hasattr(b, "_orig_mod") for b in tf.double_blocks),
        "vae_objects": len({id(v) for v in (runtime.pipeline.vae, runtime.decode_vae, runtime.pseudo_encode_vae)}),
        "torch": torch.__version__,
    })
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "clips"}), flush=True)


def _drive_reactor_model(args, model, cap, info, out) -> None:
    """Play every clip through the Reactor model the way the runtime drives it, minus WebRTC.

    The model's own step loop (ReactorCore.start_thread -> ReactorApp.run) runs on its
    thread. For each clip: SessionStarted + ClientConnected events, then the client commands
    set_prompt / set_seed / start, validated by the model's contract exactly as the runtime
    validates wire commands. Frames are pushed into the `camera` input buffer
    (ReactorCore.push_media). process_input() reads them with its configured camera_read policy,
    and the frames each step emits are collected from the media sink the runtime binds. The session
    then ends with ClientDisconnected + SessionEnded.
    """
    from reactor_runtime.core.model import ClientConnected, ClientDisconnected, EndReason, SessionEnded, SessionStarted
    from reactor_runtime.core.values import ConnId, InputFrame

    contract = type(model).__reactor_contract__
    msgs: list = []
    emitted: list = []
    model.bind_output(broadcast=msgs.append, addressed=lambda *a: None,
                      media=lambda chunk: emitted.append(np.array(chunk.bundle.tracks["main_video"].data)))
    model.start_thread()
    while model._command_q is None:
        time.sleep(0.01)
    conn = ConnId(1)

    def n_emitted() -> int:
        return sum(len(x) for x in emitted)

    def wait(pred, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        while not pred():
            if time.time() > deadline:
                raise SystemExit(f"timed out after {timeout:.0f}s waiting for {what}")
            time.sleep(0.002)

    def seen(kind: str) -> bool:
        return any(getattr(m, "name", "") == kind for m in msgs)

    for name in clip_names(args):
        prompt = args.prompt[clip_names(args).index(name)]
        frames = np.load(os.path.join(args.out, "frames", f"{name}.npy"), mmap_mode="r")
        msgs.clear(); emitted.clear()
        model.post_reactor_event(SessionStarted(session_id=f"prove-{name}"))
        model.post_reactor_event(ClientConnected(conn_id=conn, total=1))
        for cmd, cargs in (("set_prompt", {"prompt": prompt}), ("set_seed", {"seed": args.seed}), ("start", {})):
            model.submit_command(contract.validate(cmd, cargs), conn, None)
        wait(lambda: seen("generation_started"), 60, "generation_started")
        lat_s: list[float] = []
        t_next = time.perf_counter()
        i = 0
        while i < len(frames):
            n = 1 if i == 0 else 8  # one chunk: 1 frame, then ffactor_t (8)
            if args.feed == "gated":  # the next chunk's frames only once the previous chunk is out
                t0c = time.perf_counter()
                for k in range(i, i + n):
                    model.push_media("camera", InputFrame(data=np.ascontiguousarray(frames[k]), pts=k / args.feed_fps))
                wait(lambda: n_emitted() >= i + n, 120, f"chunk ending at frame {i + n}")
                lat_s.append(time.perf_counter() - t0c)
            else:  # realtime: one frame every 1/feed_fps, whatever the model does
                for k in range(i, i + n):
                    delay = t_next - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    model.push_media("camera", InputFrame(data=np.ascontiguousarray(frames[k]), pts=k / args.feed_fps))
                    t_next += 1.0 / args.feed_fps
            i += n
        if args.feed == "realtime":
            last = (-1, time.time())
            while time.time() - last[1] < 5.0 and n_emitted() < len(frames):
                if n_emitted() != last[0]:
                    last = (n_emitted(), time.time())
                time.sleep(0.05)
        model.post_reactor_event(ClientDisconnected(conn_id=conn, total=0))
        model.post_reactor_event(SessionEnded(session_id=f"prove-{name}", reason=EndReason.STOPPED))
        wait(lambda: seen("generation_complete"), 60, "generation_complete")
        np.savez_compressed(os.path.join(out, f"{name}_latents.npz"), **cap)
        cap.clear()
        got = np.concatenate(emitted) if emitted else np.zeros((0,) + frames.shape[1:], np.uint8)
        np.savez_compressed(os.path.join(out, f"{name}_frames.npz"), frames=got)
        info["clips"][name] = {"feed": args.feed, "frames_pushed": int(len(frames)), "frames_emitted": int(len(got)),
                               "chunk_latency_s": lat_s}
        print(f"  {name}: pushed {len(frames)} frames ({args.feed}), emitted {len(got)}", flush=True)
    model.stop()


def _control_vae_compile(args) -> None:
    if args.vae_compile == "on":
        return
    import xvideo.models.vae.vae_compile as VC

    for name in ("maybe_setup_decode", "maybe_setup_encode", "maybe_setup_encode_dynamic",
                 "warmup_encode", "warmup_decode", "warmup_encode_dynamic"):
        setattr(VC, name, lambda *a, **k: None)
    VC.encode_via_dynamic = lambda vae, x: vae.encode(x)


# =============================================================================== parent


def prepare_frames(args) -> dict[str, np.ndarray]:
    import av
    from PIL import Image

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    res = {}
    for path, name in zip(args.clip, clip_names(args)):
        frames = []
        with av.open(path) as c:
            for fr in c.decode(video=0):
                frames.append(np.asarray(fr.to_image().convert("RGB").resize(
                    (args.width, args.height), Image.Resampling.BICUBIC)))
        n = len(frames) if not args.max_frames else min(len(frames), args.max_frames)
        n = 1 + ((n - 1) // 8) * 8  # whole chunks only: the first chunk is 1 frame, the rest ffactor_t (8)
        res[name] = np.stack(frames[:n])
        np.save(os.path.join(args.out, "frames", f"{name}.npy"), res[name])
    return res


def spawn(args, side, tag) -> dict:
    out = os.path.join(args.out, tag)
    env = dict(os.environ)
    cache = os.path.join(args.out, "cache", side)
    env.update({"TORCHINDUCTOR_CACHE_DIR": os.path.join(cache, "inductor"), "TRITON_CACHE_DIR": os.path.join(cache, "triton"),
                "REACTOR_WEIGHTS_PATH": cache, "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    if side == "upstream":
        env.update(UPSTREAM_ENV)
        env["JOYOMNI_CUDA_GRAPH"] = "1" if args.cuda_graph == "on" else "0"
        env["PYTHONPATH"] = args.upstream
    else:
        env["PYTHONPATH"] = HERE
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--side", side, "--side-out", out]
    t0 = time.time()
    with open(os.path.join(args.out, f"{tag}.log"), "w") as log:
        rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"{tag} failed (exit {rc}); see {args.out}/{tag}.log")
    info = json.load(open(os.path.join(out, "info.json")))
    info["wall_s"] = time.time() - t0
    print(f"  {tag:<18} done in {info['wall_s']:.0f}s  attention={info['attention']}  "
          f"fp8 img/txt={info['fp8_img_blocks']}/{info['fp8_txt_blocks']}  vae objects={info['vae_objects']}", flush=True)
    return info


def rel(a, b) -> float:
    n = float(np.linalg.norm(a.ravel()))
    return float(np.linalg.norm((a - b).ravel())) / (n if n else 1.0)


def psnr(a, b) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def compare(args, a_tag, b_tag) -> dict:
    res = {"a": a_tag, "b": b_tag, "clips": {}}
    for name in clip_names(args):
        A = np.load(os.path.join(args.out, a_tag, f"{name}_latents.npz"))
        B = np.load(os.path.join(args.out, b_tag, f"{name}_latents.npz"))
        fa = np.load(os.path.join(args.out, a_tag, f"{name}_frames.npz"))["frames"]
        fb = np.load(os.path.join(args.out, b_tag, f"{name}_frames.npz"))["frames"]
        ps = [psnr(fa[i], fb[i]) for i in range(min(len(fa), len(fb)))]
        chunks = []
        for c in range(len([k for k in A.files if k.startswith("dit_")])):
            row = {"chunk": c}
            for st in ("enc", "dit", "pseudo"):
                key = f"{st}_{c + 1 if st == 'pseudo' else c:03d}"  # pseudo_{c+1} is made from chunk c's pixels
                if key in A.files and key in B.files:
                    row[f"{st}_rel"] = rel(A[key], B[key])
            lo = 0 if c == 0 else 1 + (c - 1) * 8
            hi = 1 if c == 0 else lo + 8
            row["psnr_min"] = min(ps[lo:hi])
            row["maxabs_px"] = int(np.abs(fa[lo:hi].astype(np.int16) - fb[lo:hi]).max())
            over = [st for st in ("enc", "dit", "pseudo") if row.get(f"{st}_rel", 0) > args.tol_latent]
            if not over and row["psnr_min"] < args.tol_psnr:
                over = ["decode"]
            row["first_divergent_stage"] = over[0] if over else None
            chunks.append(row)
        ok = len(fa) == len(fb) and all(r.get("dit_rel", 0) <= args.tol_latent and r["psnr_min"] >= args.tol_psnr
                                        for r in chunks)
        res["clips"][name] = {"pass": ok, "frames": len(ps), "psnr_mean": float(np.mean(np.minimum(ps, 99.0))),
                              "psnr_min": float(min(ps)), "bit_exact": all(r["maxabs_px"] == 0 for r in chunks)
                              and all(r.get(f"{s}_rel", 0) == 0 for r in chunks for s in ("enc", "dit", "pseudo")),
                              "chunks": chunks}
    res["pass"] = all(c["pass"] for c in res["clips"].values())
    return res


def print_table(res) -> None:
    print(f"\n{res['b']} vs {res['a']}:  {'PASS' if res['pass'] else 'FAIL'}")
    print(f"  {'clip':<16} {'frames':>6} {'bit-exact':>9} {'PSNR mean':>10} {'PSNR min':>9}  first divergent chunk/stage")
    for name, c in res["clips"].items():
        first = next((f"chunk {r['chunk']}: {r['first_divergent_stage']} (dit rel {r.get('dit_rel', 0):.2e})"
                      for r in c["chunks"] if r["first_divergent_stage"]), "-")
        print(f"  {name:<16} {c['frames']:>6} {str(c['bit_exact']):>9} {c['psnr_mean']:>10.2f} "
              f"{c['psnr_min'] if c['psnr_min'] != float('inf') else float('inf'):>9.2f}  {first}")


def write_video(args, srcs, res, labels) -> str:
    import av
    from PIL import Image, ImageDraw, ImageFont

    pw, ph, head = 624, 360, 56
    try:
        font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=18)
    except TypeError:
        font = small = ImageFont.load_default()
    path = os.path.join(args.out, "parity.mp4")
    with av.open(path, "w") as out:
        for codec in ("libx264", "h264", "mpeg4"):
            try:
                s = out.add_stream(codec, rate=24)
                break
            except Exception:  # noqa: BLE001  (codec not built into this PyAV)
                continue
        s.width, s.height, s.pix_fmt = pw * 3, ph + head, "yuv420p"
        s.options = {"crf": "18"} if s.codec_context.name == "libx264" else {}
        s.bit_rate = 20_000_000
        for name, prompt in zip(clip_names(args), args.prompt):
            fa = np.load(os.path.join(args.out, "upstream", f"{name}_frames.npz"))["frames"]
            fb = np.load(os.path.join(args.out, "integration", f"{name}_frames.npz"))["frames"]
            c = res["clips"][name]
            ptxt = "bit-exact" if c["bit_exact"] else f"PSNR mean {c['psnr_mean']:.2f} dB, min {c['psnr_min']:.2f} dB"
            for i in range(min(len(fa), len(fb))):
                canvas = Image.new("RGB", (pw * 3, ph + head), (16, 16, 16))
                d = ImageDraw.Draw(canvas)
                d.text((10, 6), f"{name}   frame {i:03d}/{len(fa) - 1}   {ptxt}   seed {args.seed}  "
                                f"{args.width}x{args.height}  posterior {args.posterior}", fill=(255, 255, 255), font=font)
                d.text((10, 32), f"instruction: {prompt}", fill=(200, 200, 200), font=small)
                for k, (fr, lab) in enumerate(zip((srcs[name][i], fa[i], fb[i]), labels)):
                    canvas.paste(Image.fromarray(np.asarray(fr)).resize((pw, ph), Image.Resampling.BILINEAR), (k * pw, head))
                    d.rectangle([k * pw, head, k * pw + 20 + int(d.textlength(lab, font=font)), head + 30], fill=(0, 0, 0))
                    d.text((k * pw + 10, head + 3), lab, fill=(255, 230, 80), font=font)
                for p in s.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    out.mux(p)
        for p in s.encode():
            out.mux(p)
    return path


def main() -> int:
    args = parse_args()
    if args.side:
        run_side(args)
        return 0
    os.makedirs(args.out, exist_ok=True)
    srcs = prepare_frames(args)
    print(f"{len(srcs)} clip(s), {sum(len(v) for v in srcs.values())} frames at {args.width}x{args.height}; "
          f"posterior {args.posterior}, vae-compile {args.vae_compile}, attention {args.attention}", flush=True)
    runs = [("upstream", "upstream"), ("integration", "integration")]
    if args.self_check:
        runs += [("upstream", "upstream-rerun"), ("integration", "integration-rerun")]
    infos = {tag: spawn(args, side, tag) for side, tag in runs}
    results = [compare(args, "upstream", "integration")]
    if args.self_check:
        results += [compare(args, "upstream", "upstream-rerun"), compare(args, "integration", "integration-rerun")]
    for r in results[1:] + results[:1]:
        print_table(r)
    video = write_video(args, srcs, results[0], ("source", f"upstream ({infos['upstream']['attention']})",
                                                 f"this model dir ({infos['integration']['attention']})"))
    report = {"args": {k: v for k, v in vars(args).items() if not k.startswith("side")},
              "sides": infos, "comparisons": results, "video": video}
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    verdict = results[0]["pass"]
    print(f"\n{'PASS' if verdict else 'FAIL'}: upstream vs this model dir  (tolerances: dit rel <= {args.tol_latent:g}, "
          f"pixels >= {args.tol_psnr:g} dB)\nvideo: {video}\nreport: {os.path.join(args.out, 'report.json')}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
