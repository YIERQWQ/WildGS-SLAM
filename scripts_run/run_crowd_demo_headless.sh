#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ba3033/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="${ROOT_DIR}/.conda/envs:${CONDA_ENVS_PATH}"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
export DINOV3_WEIGHTS=/mnt/HDD1/ba3033/SLAM/wildgs-slam/pretrained/dinov3/vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
conda activate wildgs-slam
cd "${ROOT_DIR}"

: "${DINOV3_WEIGHTS:?Set DINOV3_WEIGHTS to the DINOv3 checkpoint path or URL}"
export DINOV3_WEIGHTS

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${ROOT_DIR}/output/run_logs/${RUN_TS}"
mkdir -p "${RUN_LOG_DIR}"
RUN_LOG_FILE="${RUN_LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG_FILE}") 2>&1

echo "[RUN] log_dir=${RUN_LOG_DIR}"

BETA_HOST="127.0.0.1"
BETA_PORT="5555"
BETA_LOG="${RUN_LOG_DIR}/dino_beta_service.log"

port_is_open() {
    local host="$1"
    local port="$2"
    (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout_s="${3:-300}"
    local start_s
    start_s="$(date +%s)"
    while ! port_is_open "${host}" "${port}"; do
        if [ $(( $(date +%s) - start_s )) -ge "${timeout_s}" ]; then
            echo "Timed out waiting for DINOv3 beta service on ${host}:${port}" >&2
            [ -f "${BETA_LOG}" ] && tail -n 50 "${BETA_LOG}" >&2 || true
            exit 1
        fi
        sleep 1
    done
}

if port_is_open "${BETA_HOST}" "${BETA_PORT}"; then
    echo "[DINO-BETA] Reusing existing service on ${BETA_HOST}:${BETA_PORT}"
    BETA_PID=""
else
    mkdir -p "$(dirname "${BETA_LOG}")"
    CUDA_VISIBLE_DEVICES=1 python src/utils/dino_beta_service.py \
        --device cuda:0 \
        --host "${BETA_HOST}" \
        --port "${BETA_PORT}" \
        --weights "${DINOV3_WEIGHTS}" \
        > "${BETA_LOG}" 2>&1 &
    BETA_PID=$!
    trap 'if [ -n "${BETA_PID}" ] && kill -0 "${BETA_PID}" 2>/dev/null; then kill "${BETA_PID}" 2>/dev/null || true; wait "${BETA_PID}" 2>/dev/null || true; fi' EXIT
    wait_for_port "${BETA_HOST}" "${BETA_PORT}" 300
fi

CUDA_VISIBLE_DEVICES=0 python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
