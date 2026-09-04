# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential ca-certificates ffmpeg git ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# RLDX-1 uses PyTorch 2.7.0. The CUDA 12.8 wheel includes B200 support.
RUN pip install --no-cache-dir --upgrade \
    pip==25.3 setuptools==80.9.0 wheel==0.45.1 packaging==26.3 psutil==7.2.2
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# FlashAttention 2.7.4 has no prebuilt Blackwell wheel, so compile it for B200.
ARG FLASH_ATTN_MAX_JOBS=16
RUN MAX_JOBS=${FLASH_ATTN_MAX_JOBS} NVCC_THREADS=2 \
    TORCH_CUDA_ARCH_LIST="10.0+PTX" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1

ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    HOST=0.0.0.0 \
    PORT=8080 \
    ORPHAN_TIMEOUT_SECONDS=30

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/

ARG IMAGE_VERSION
ENV IMAGE_VERSION=${IMAGE_VERSION}

EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
