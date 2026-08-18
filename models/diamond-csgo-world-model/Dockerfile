# DIAMOND CSGO Reactor model image. `reactor build` and `reactor run` invoke
# Docker for this workspace; the upstream source stays pinned and unmodified.

FROM alpine:3.22 AS diamond-source

ARG DIAMOND_REVISION=851cefb497733d27f1b85c804104638765860fca
RUN apk add --no-cache git \
    && git init /diamond \
    && git -C /diamond remote add origin https://github.com/eloialonso/diamond.git \
    && git -C /diamond fetch --depth 1 origin "${DIAMOND_REVISION}" \
    && git -C /diamond checkout --detach FETCH_HEAD \
    && rm -rf /diamond/.git

FROM python:3.12-slim

WORKDIR /app

# Install the runtime and serving dependencies in one cached layer.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Keep the upstream source outside the Reactor-owned adapter directory. The
# adapter validates this root before importing DIAMOND's CSGO modules.
COPY --from=diamond-source /diamond /opt/diamond
ENV DIAMOND_PATH=/opt/diamond

COPY . /app/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
