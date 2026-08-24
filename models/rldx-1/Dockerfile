# syntax=docker/dockerfile:1
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
# RLDX-1 serving image for NVIDIA RTX PRO 6000 (Blackwell, sm_120).

FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG RLDX_REPOSITORY=https://github.com/RLWRLD/RLDX-1.git
ARG RLDX_REVISION=ecbfaf80cd031dcc892186ed30465de3591047e6

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        git build-essential ninja-build ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# FlashAttention's source build imports these packages from the active
# environment, so install and pin them before compiling the extension.
RUN pip install --no-cache-dir --upgrade \
    pip==25.3 \
    setuptools==80.9.0 \
    wheel==0.45.1 \
    packaging==26.3 \
    psutil==7.2.2

# RLDX-1 pins PyTorch 2.7.0. Use the CUDA 12.8 wheels so Blackwell kernels are
# available for the RTX PRO 6000 instead of the default CUDA 12.6 build.
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# The published FlashAttention wheel does not contain sm_120 kernels, so compile
# it against the CUDA toolkit in this image. Lower the job count on a smaller
# Docker builder with --build-arg FLASH_ATTN_MAX_JOBS=<n>.
ARG FLASH_ATTN_MAX_JOBS=16
RUN MAX_JOBS=${FLASH_ATTN_MAX_JOBS} NVCC_THREADS=2 \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Keep upstream source isolated and pinned. Its Apache-2.0 license remains in
# the checkout at /opt/rldx-1/LICENSE.md.
ENV RLDX_PATH=/opt/rldx-1 \
    PYTHONPATH=/opt/rldx-1
RUN git init "$RLDX_PATH" \
    && git -C "$RLDX_PATH" fetch --depth 1 \
        "$RLDX_REPOSITORY" "$RLDX_REVISION" \
    && git -C "$RLDX_PATH" checkout --detach FETCH_HEAD

COPY . /app/

# Limit host thread pools; model inference is GPU-bound.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    HOST=0.0.0.0 \
    PORT=8080 \
    ORPHAN_TIMEOUT_SECONDS=30

EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
