#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ba3033/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="${ROOT_DIR}/.conda/envs:${CONDA_ENVS_PATH}"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

conda activate wildgs-slam
cd "${ROOT_DIR}"

python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
