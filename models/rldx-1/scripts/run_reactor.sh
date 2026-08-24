#!/usr/bin/env bash
#
# rldx-reactor inference server container driver.
#
# Builds the workspace Dockerfile and runs the Reactor Runtime inference
# server for RLDX-1 (video-in -> action-out) on a CUDA GPU box. Iterate on
# the pipeline without re-building: edit config.yml, then `restart`.
#
# Container:
#   rldx-reactor — `python -m reactor_runtime.serve` on :${REACTOR_HTTP_PORT},
#                  built from Dockerfile (CUDA devel + the
#                  standalone reactor-runtime installed from PyPI, cu128 torch
#                  for Blackwell, source-built flash-attn for sm_120). The
#                  runtime takes no CLI arguments — HOST / PORT /
#                  ORPHAN_TIMEOUT_SECONDS are passed as environment variables.
#
# Subcommands:
#   build     build or rebuild the image
#   start     boot the container (builds the image on first run)
#   stop      stop + remove the container
#   restart   stop + start (use after editing config.yml)
#   status    container state + :PORT listener + last log lines
#   logs      tail -F ${LOG_DIR}/reactor.log
#
# Env knobs (defaults shown):
#   WEIGHTS_DIR         (required) host path to the RLDX-1-FT-ROBOCASA
#                       checkpoint dir; mounted read-only at /weights and
#                       exposed as REACTOR_WEIGHTS_PATH. config.yml's empty
#                       checkpoint_dir then resolves to the bundle root.
#   REACTOR_HTTP_PORT   8080
#   GPUS                all   (passed to docker --gpus; e.g. '"device=0"')
#   LOG_DIR             /tmp/rldx-reactor-logs
#   REACTOR_IMAGE       rldx-reactor:dev
#   REACTOR_NAME        rldx-reactor
#   FLASH_ATTN_MAX_JOBS 16
#
# See inference.md for the client-side flow (py-sdk / cpp_sdk).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CMD="${1:-start}"

REACTOR_HTTP_PORT="${REACTOR_HTTP_PORT:-8080}"
GPUS="${GPUS:-all}"
LOG_DIR="${LOG_DIR:-/tmp/rldx-reactor-logs}"
REACTOR_IMAGE="${REACTOR_IMAGE:-rldx-reactor:dev}"
REACTOR_NAME="${REACTOR_NAME:-rldx-reactor}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-16}"

mkdir -p "${LOG_DIR}"

log() { printf '[run_reactor] %s\n' "$*" >&2; }

build_image() {
    log "building ${REACTOR_IMAGE} from Dockerfile"
    log "  (the first build compiles FlashAttention from source)"
    docker build --provenance=false \
        --build-arg "FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS}" \
        -f "${REPO_ROOT}/Dockerfile" \
        -t "${REACTOR_IMAGE}" "${REPO_ROOT}" 2>&1 | tail -12
}

ensure_image() {
    if docker image inspect "${REACTOR_IMAGE}" >/dev/null 2>&1; then return 0; fi
    build_image
}

stop_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${REACTOR_NAME}$"; then
        log "stopping container '${REACTOR_NAME}'"
        docker rm -f "${REACTOR_NAME}" >/dev/null 2>&1 || true
    fi
}

start_reactor() {
    if [ -z "${WEIGHTS_DIR:-}" ]; then
        log "ERROR: set WEIGHTS_DIR to the host RLDX-1-FT-ROBOCASA checkpoint dir."
        log "  e.g.  WEIGHTS_DIR=~/rldx-robocasa $0 start"
        exit 2
    fi
    if [ ! -d "${WEIGHTS_DIR}" ]; then
        log "ERROR: WEIGHTS_DIR '${WEIGHTS_DIR}' is not a directory."
        exit 2
    fi
    ensure_image
    stop_container

    log "starting ${REACTOR_NAME} on :${REACTOR_HTTP_PORT} (gpus=${GPUS})"
    # --network host so WebRTC ICE candidates resolve to localhost on both ends.
    # config.yml is mounted so `restart` picks up edits without a rebuild.
    docker run -d --rm \
        --network host \
        --gpus "${GPUS}" \
        --name "${REACTOR_NAME}" \
        -e PYTHONUNBUFFERED=1 \
        -e REACTOR_WEIGHTS_PATH=/weights \
        -e HOST=0.0.0.0 \
        -e PORT="${REACTOR_HTTP_PORT}" \
        -e ORPHAN_TIMEOUT_SECONDS=30 \
        -v "${WEIGHTS_DIR}":/weights:ro \
        -v "${REPO_ROOT}/config.yml":/app/config.yml:ro \
        "${REACTOR_IMAGE}" >/dev/null

    ( docker logs -f "${REACTOR_NAME}" 2>&1 \
        | tee "${LOG_DIR}/reactor.log" >/dev/null ) &
    disown

    log "started"
    log "  logs:  docker logs -f ${REACTOR_NAME}   |   tail -F ${LOG_DIR}/reactor.log"
    log "  stop:  $0 stop"
}

status_reactor() {
    echo "=== docker container ==="
    docker ps --filter "name=${REACTOR_NAME}" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | head -5
    echo
    echo "=== :${REACTOR_HTTP_PORT} listener ==="
    ss -ltn "sport = :${REACTOR_HTTP_PORT}" 2>/dev/null | tail -n +2 || echo "(none)"
    echo
    echo "=== last 5 lines of ${LOG_DIR}/reactor.log ==="
    tail -5 "${LOG_DIR}/reactor.log" 2>/dev/null || echo "(no log)"
}

case "${CMD}" in
    build)   build_image ;;
    start)   start_reactor ;;
    stop)    stop_container ;;
    restart) stop_container; start_reactor ;;
    status)  status_reactor ;;
    logs)
        log "tailing ${LOG_DIR}/reactor.log (Ctrl-C to stop)"
        tail -F "${LOG_DIR}/reactor.log"
        ;;
    *)
        cat >&2 <<EOF
Usage: WEIGHTS_DIR=<checkpoint> $0 {build|start|stop|restart|status|logs}

  build     build or rebuild the rldx-reactor image
  start     boot the rldx-reactor container (builds the image once)
  stop      stop + remove the container
  restart   stop + start (after editing config.yml)
  status    container state + :${REACTOR_HTTP_PORT} listener + log tail
  logs      tail -F ${LOG_DIR}/reactor.log

Env knobs (defaults shown):
  WEIGHTS_DIR         (required — RLDX-1-FT-ROBOCASA checkpoint dir)
  REACTOR_HTTP_PORT   ${REACTOR_HTTP_PORT}
  GPUS                ${GPUS}
  LOG_DIR             ${LOG_DIR}
  REACTOR_IMAGE       ${REACTOR_IMAGE}
  REACTOR_NAME        ${REACTOR_NAME}
  FLASH_ATTN_MAX_JOBS ${FLASH_ATTN_MAX_JOBS}

See inference.md for the client-side flow.
EOF
        exit 2
        ;;
esac
