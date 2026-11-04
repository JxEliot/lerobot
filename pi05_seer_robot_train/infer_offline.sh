#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

VENV_ROOT="${VENV_ROOT:-/root/venvs/lerobot-pi05}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPOSITORY_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee}"
DATASET_REPO_ID="${DATASET_REPO_ID:-seer_robot/lerobot_ee}"
HF_HOME="${HF_HOME:-/mnt/cfs/usr/jixiang/hf_models/hub}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${LEROBOT_ROOT}/src/lerobot" ]] || { echo "LeRobot source not found: ${LEROBOT_ROOT}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "Dataset root not found: ${DATASET_ROOT}" >&2; exit 1; }

export LEROBOT_ROOT DATASET_ROOT DATASET_REPO_ID HF_HOME
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${LEROBOT_ROOT}/src:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_offline.py" "$@"
