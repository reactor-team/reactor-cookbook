# Vendored: jd-opensource/JoyAI-Video-Edit (`deploy/xvideo`)

- Source:  https://github.com/jd-opensource/JoyAI-Video-Edit
- Commit:  ca17e1d1030f454cb98b0ed549b4d31a60139ceb (`deploy/xvideo`)
- License: Apache-2.0 (the repository root `LICENSE` carries the same text)

`xvideo` is the inference package of upstream's streaming server: the MMDiT,
the causal video VAE, the MiMo-VL condition encoder, and the streaming session
that edits a camera stream chunk by chunk. This copy is modified in how it is
served and how fast it runs; `scripts/prove_parity.py` runs it and upstream on
the same clips and compares their latents and pixels chunk by chunk.

Removed: the web server and its helpers (`serving/serve_joyomni_streaming.py`,
`serving/pe.py`), `lowvram.py`, and `fp8_status.py`. The Reactor application
replaces the server.

Changed:

- `serving/joyomni_streaming.py` runs every chunk on one persistent compute
  thread and the default CUDA stream (encode, DiT denoise and KV store,
  decode, pseudo-encode, postprocess), in place of upstream's stage threads
  and side streams. It adds `push_chunk()`, which runs exactly one chunk from
  the frames it is given, and `warmup_sessions()`, which runs whole sessions
  at load so that every shape a session can meet is planned and every CUDA
  graph captured before the first client. The text is padded to a fixed
  token count with the padding masked, and reference images are fitted into
  a fixed set of aspects, so that set of shapes is finite. Frames that are not
  at the session size are resized on a thread pool, scaled on the encode
  device, and moved through pinned host memory; output is raw uint8.
- `serving/graph_runner.py` extends upstream's `StreamingGraphRunner` to
  chunk 0, chunk 1, the steady window and the freeze-on-static window, with
  and without a reference image, in one shared memory pool.
- `models/dit/dit.py` calls cuDNN SDPA directly (FlashAttention-4 is not
  used), masks the padded text exactly and overflow-safely, writes q/k/v
  straight into the attention buffers, and fuses the FP8 quantisation, the
  RoPE copies and the padded-attention rescale into `joyomni_ops` kernels
  (`models/dit/sgl_fused_ops.py`, `models/dit/fp8_linear.py`).
- `models/vae/vae_compile.py` replays VAE encode and decode as CUDA graphs;
  `JOYOMNI_VAE_COMPILE_MODE` switches the compile mode (`default` restarts
  faster than the autotuned default).
- `models/models.py` memory-maps the DiT checkpoint instead of reading it
  into host memory.
- `models/pipeline.py`, `models/vae/vae.py`, `models/scheduler.py`,
  `config.py`, and `utils.py` carry the small changes the items above need.
