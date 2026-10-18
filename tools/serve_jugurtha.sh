#!/usr/bin/env bash
# ==============================================================================
# AƔBALU: Jugurtha-4B Production Daemon & OpenAI API Server (vLLM)
# ==============================================================================
# Usage:
#   ./tools/serve_jugurtha.sh start    # Start vLLM in background with monitoring
#   ./tools/serve_jugurtha.sh stop     # Gracefully stop the server
#   ./tools/serve_jugurtha.sh status   # Check server health and PID
#   ./tools/serve_jugurtha.sh restart  # Restart the server
#   ./tools/serve_jugurtha.sh logs     # Tail live server logs
# ==============================================================================

set -euo pipefail

# Configuration
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="/home/jovyan/envs/vllm/bin/python"
VENV_VLLM="/home/jovyan/envs/vllm/bin/vllm"
MODEL_DIR="${REPO_DIR}/artifacts/runs/jugurtha-4b-sft/model"
RUN_DIR="${REPO_DIR}/artifacts/runs/vllm-serve"
PID_FILE="${RUN_DIR}/vllm.pid"
LOG_FILE="${RUN_DIR}/serve.log"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-jugurtha-4b}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-32768}"

# Environment variables for optimal H100 vLLM execution
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="${CUDA_HOME:-/home/jovyan/envs/vllm/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"

get_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "${pid}"
            return 0
        fi
    fi
    # Fallback: check ps for any active vllm serve process for Jugurtha
    pgrep -f "vllm.*jugurtha" | head -n 1 || true
}

start_server() {
    local pid
    pid="$(get_pid)"
    if [[ -n "${pid}" ]]; then
        echo "[Jugurtha-4B] Server is ALREADY RUNNING (PID: ${pid})."
        echo "[Jugurtha-4B] Listening on http://${HOST}:${PORT}"
        return 0
    fi

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: Model directory does not exist: ${MODEL_DIR}" >&2
        exit 1
    fi

    if [[ ! -x "${VENV_VLLM}" ]]; then
        echo "ERROR: vLLM executable not found at ${VENV_VLLM}" >&2
        exit 1
    fi

    echo "=================================================================="
    echo " Starting Jugurtha-4B Production vLLM Server"
    echo " Model:     ${MODEL_DIR}"
    echo " Template:  ${MODEL_DIR}/chat_template.jinja"
    echo " Aliases:   Jugurtha-4B, agbalu/Jugurtha-4B, jugurtha-4b"
    echo " Endpoint:  http://${HOST}:${PORT}"
    echo " Context:   ${MAX_LEN} tokens"
    echo " GPU Util:  ${GPU_UTIL}"
    echo " Logs:      ${LOG_FILE}"
    echo "=================================================================="

    nohup "${VENV_VLLM}" serve "${MODEL_DIR}" \
        --served-model-name "Jugurtha-4B" "agbalu/Jugurtha-4B" "jugurtha-4b" \
        --chat-template "${MODEL_DIR}/chat_template.jinja" \
        --host "${HOST}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --max-model-len "${MAX_LEN}" \
        --dtype bfloat16 \
        --trust-remote-code \
        > "${LOG_FILE}" 2>&1 &

    local new_pid=$!
    echo "${new_pid}" > "${PID_FILE}"
    echo "[Jugurtha-4B] Process spawned with PID: ${new_pid}"

    echo -n "[Jugurtha-4B] Waiting for engine warmup & CUDA graph capture..."
    local attempts=0
    local max_attempts=90
    while [[ ${attempts} -lt ${max_attempts} ]]; do
        if ! kill -0 "${new_pid}" 2>/dev/null; then
            echo ""
            echo "ERROR: vLLM process died unexpectedly! See logs below:" >&2
            echo "----------------------------------------------------" >&2
            tail -n 25 "${LOG_FILE}" >&2
            echo "----------------------------------------------------" >&2
            rm -f "${PID_FILE}"
            exit 1
        fi

        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo " READY!"
            echo "=================================================================="
            echo " [Jugurtha-4B] Status: ONLINE and HEALTHY"
            echo " OpenAI Base URL: http://${HOST}:${PORT}/v1"
            echo " Chat Endpoint:   http://${HOST}:${PORT}/v1/chat/completions"
            echo " Models List:     http://${HOST}:${PORT}/v1/models"
            echo " Swagger Docs:    http://${HOST}:${PORT}/docs"
            echo "=================================================================="
            return 0
        fi

        echo -n "."
        sleep 2
        ((attempts++))
    done

    echo ""
    echo "WARNING: Server started but healthcheck timed out after 90s. Check logs: ${LOG_FILE}"
}

stop_server() {
    local pid
    pid="$(get_pid)"
    if [[ -z "${pid}" ]]; then
        echo "[Jugurtha-4B] No running server found."
        rm -f "${PID_FILE}"
        return 0
    fi

    echo "[Jugurtha-4B] Stopping server (PID: ${pid})..."
    kill -TERM "${pid}" 2>/dev/null || true

    local count=0
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 1
        ((count++))
        if [[ ${count} -gt 15 ]]; then
            echo "[Jugurtha-4B] Force killing server (PID: ${pid})..."
            kill -KILL "${pid}" 2>/dev/null || true
            break
        fi
    done

    # Clean up any residual child/worker processes
    pkill -f "vllm.*jugurtha" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "[Jugurtha-4B] Server stopped cleanly."
}

check_status() {
    local pid
    pid="$(get_pid)"
    if [[ -n "${pid}" ]]; then
        echo "[Jugurtha-4B] Status: RUNNING (PID: ${pid})"
        if curl -s -f "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[Jugurtha-4B] Health: OK (HTTP 200)"
            echo "[Jugurtha-4B] Models: Jugurtha-4B | agbalu/Jugurtha-4B | jugurtha-4b"
            echo "[Jugurtha-4B] API:    http://${HOST}:${PORT}/v1"
            echo "[Jugurtha-4B] Chat:   http://${HOST}:${PORT}/v1/chat/completions"
        else
            echo "[Jugurtha-4B] Health: INITIALIZING / UNHEALTHY (check logs: ${LOG_FILE})"
        fi
    else
        echo "[Jugurtha-4B] Status: STOPPED"
    fi
}

show_logs() {
    if [[ -f "${LOG_FILE}" ]]; then
        tail -f -n 50 "${LOG_FILE}"
    else
        echo "No log file found at ${LOG_FILE}"
    fi
}

case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
