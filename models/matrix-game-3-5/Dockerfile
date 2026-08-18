# syntax=docker/dockerfile:1
# Reactor Matrix-Game-3.5 image with an isolated upstream Python toolchain.

FROM ghcr.io/astral-sh/uv:0.12.3 AS uv
FROM python:3.12-slim

WORKDIR /app

# Git prepares the pinned public source checkout. The media libraries support
# Matrix's OpenCV and video dependencies; compilers support CUDA extensions.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv installs Python 3.10 and the upstream model dependencies into the mounted
# weights cache on first load, outside the Runtime's Python 3.12 environment.
COPY --from=uv /uv /usr/local/bin/uv

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTORCH_ALLOC_CONF=expandable_segments:True
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
