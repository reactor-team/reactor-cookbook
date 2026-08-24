# syntax=docker/dockerfile:1
# Reactor PersonaPlex image. `reactor build` and `reactor run` invoke Docker for
# this workspace; reactor-runtime ships its own WebRTC media stack as a wheel,
# so a plain Python base needs no system media libraries.

FROM python:3.12-slim

WORKDIR /app

# PersonaPlex is published as a subdirectory of its Git repository rather than
# to PyPI, so pip drives the `git` binary to fetch the pinned commit.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Keep the expensive dependency layer reusable across adapter-only edits.
COPY requirements.txt requirements-personaplex.txt /app/
# Two passes because PersonaPlex's declared `torch < 2.5` and the CUDA 12.8
# PyTorch a B200 needs cannot be resolved together. The runtime, PyTorch, and
# every dependency PersonaPlex declares are resolved as one set first; the model
# code then lands --no-deps on top of it. requirements-personaplex.txt has the
# detail.
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps -r /app/requirements-personaplex.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
