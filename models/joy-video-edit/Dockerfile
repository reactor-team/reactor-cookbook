# syntax=docker/dockerfile:1
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
#
# JoyAI-Video-Edit on Reactor Runtime. One image for NVIDIA B200 (sm_100) and
# B300 (sm_103). `reactor build` builds this file verbatim.

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    ca-certificates \
    build-essential \
    ninja-build \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv $VIRTUAL_ENV

RUN pip install --no-cache-dir \
    --upgrade pip==25.3 \
    setuptools==80.9.0 \
    wheel==0.45.1

# torch 2.14.0+cu132: sm_100 SASS (binary-compatible with sm_103) plus sm_100a/sm_103
# arch-specific kernels; its triton 3.8 ptxas accepts sm_103a, which inductor needs on B300.
RUN pip install --no-cache-dir \
    torch==2.14.0+cu132 \
    torchvision==0.29.0+cu132 \
    --index-url https://download.pytorch.org/whl/cu132

# cu13 wheels put the toolkit libraries under nvidia/cu13/lib; the pip cuDNN (the one torch was
# built against) must win over the base image's apt cuDNN in /usr/lib/x86_64-linux-gnu.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvshmem/lib:/usr/local/cuda/lib64

WORKDIR /app

# The model's dependencies and Reactor Runtime resolve together, so a conflict
# fails the build. Keep RUNTIME_VERSION equal to reactor.yaml's build.runtime_version.
ARG RUNTIME_VERSION=3.6.0
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt \
    "reactor-runtime==${RUNTIME_VERSION}"

# joyomni_ops: the fused DiT kernels and the FP8 GEMM, built from the vendored source.
COPY joyomni_ops /app/joyomni_ops

# CUTLASS v4.7.1 (SM100f / SM103 family targets), headers only.
RUN git clone --filter=blob:none --sparse \
    https://github.com/NVIDIA/cutlass.git \
    /opt/cutlass && \
    cd /opt/cutlass && \
    git checkout v4.7.1 && \
    git sparse-checkout set include tools/util/include && \
    rm -rf .git

# 100f is the Blackwell datacenter family: one SASS image that runs natively on
# sm_100 (B200) and sm_103 (B300). A build without the FP8 GEMM would change the
# model's numerics, so its absence fails the image.
RUN JOYOMNI_OPS_CUTLASS_DIR=/opt/cutlass \
    JOYOMNI_OPS_CUDA_ARCHS="90;100f" \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8 \
    pip install --no-cache-dir \
      --no-build-isolation \
      /app/joyomni_ops \
 && python3 -c "import joyomni_ops; assert joyomni_ops.has_fp8(), 'FP8 kernel missing'" \
 && echo "[OK] joyomni_ops FP8 OK"

COPY . /app/

ENV HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "reactor_runtime.serve"]
CMD ["run"]
