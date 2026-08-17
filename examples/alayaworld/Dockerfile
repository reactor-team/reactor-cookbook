# syntax=docker/dockerfile:1
# Reactor AlayaWorld image with the build tools required by torch.compile.

FROM python:3.12-slim

WORKDIR /app

# Git prepares the pinned public source checkouts at model load. TorchInductor
# compiles small host-side extensions for the selected CUDA device.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Keep the expensive dependency layer reusable across adapter-only edits.
COPY requirements.txt requirements-flash-attn.txt /app/
# Three passes because FlashAttention 4 and the runtime cannot be resolved
# together: the CuTe DSL caps protobuf below 7 and the runtime needs 7.35. The
# attention wheels land second, then the protobuf the runtime needs goes back on
# top — verified to serve both. requirements-flash-attn.txt has the detail.
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip install --no-cache-dir -r /app/requirements-flash-attn.txt \
    && python -m pip install --no-cache-dir --no-deps "protobuf==7.35.1"

COPY . /app/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTORCH_ALLOC_CONF=expandable_segments:True
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
