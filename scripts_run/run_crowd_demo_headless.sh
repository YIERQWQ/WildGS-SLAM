#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ba3033/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="${ROOT_DIR}/.conda/envs:${CONDA_ENVS_PATH}"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

conda activate wildgs-slam
cd "${ROOT_DIR}"

DATASET_SRC="${ROOT_DIR}/../wildgs-slam/datasets/Wild_SLAM_Mocap"
DATASET_DST="${ROOT_DIR}/datasets/Wild_SLAM_Mocap"
if [ ! -e "${DATASET_DST}" ] && [ -e "${DATASET_SRC}" ]; then
    mkdir -p "${ROOT_DIR}/datasets"
    ln -s "${DATASET_SRC}" "${DATASET_DST}"
fi

PRETRAINED_SRC="${ROOT_DIR}/../wildgs-slam/pretrained"
PRETRAINED_DST="${ROOT_DIR}/pretrained"
if [ ! -e "${PRETRAINED_DST}" ] && [ -e "${PRETRAINED_SRC}" ]; then
    ln -s "${PRETRAINED_SRC}" "${PRETRAINED_DST}"
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${ROOT_DIR}/output/run_logs/${RUN_TS}"
mkdir -p "${RUN_LOG_DIR}"
export RUN_LOG_DIR
exec > >(tee -a "${RUN_LOG_DIR}/run.log") 2>&1

python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
