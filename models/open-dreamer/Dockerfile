# syntax=docker/dockerfile:1
# OpenDreamer Reactor model image.
#
# `reactor build` / `reactor run` build and run this image. The Dockerfile
# fetches the pinned public OpenDreamer source and conditioning demo without
# copying either into this recipe repository.

FROM python:3.12-slim

ARG OPENDREAMER_REPOSITORY=https://github.com/next-state/open-dreamer.git
ARG OPENDREAMER_REVISION=797e41f052b5996740938fd2fe8161f1866de3a2

# Git installs the pinned Optax dependency and fetches OpenDreamer. FFmpeg
# supports runtime recording; ca-certificates covers both public downloads.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the runtime and model dependencies before copying adapter source so
# source-only edits preserve the expensive CUDA/JAX dependency layer.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Keep upstream source isolated under /opt and retain its Git metadata so the
# adapter can verify the exact revision before importing it.
ENV OPENDREAMER_PATH=/opt/open-dreamer
RUN git init "$OPENDREAMER_PATH" \
    && git -C "$OPENDREAMER_PATH" fetch --depth 1 \
        "$OPENDREAMER_REPOSITORY" "$OPENDREAMER_REVISION" \
    && git -C "$OPENDREAMER_PATH" checkout --detach FETCH_HEAD

COPY . /app/

# Bake the public 184 MB VPT sample into the image. The checkpoint remains in
# Reactor's mounted weights cache so it persists across local container runs.
RUN python /app/download_demo.py --output-dir "$OPENDREAMER_PATH/samples/vpt"

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
