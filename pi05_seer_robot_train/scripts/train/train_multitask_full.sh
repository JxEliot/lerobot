#!/usr/bin/env bash
# Train PI05 from the base model on every episode in the merged Seer dataset.
# Usage: bash scripts/train/train_multitask_full.sh [check|smoke|full] [steps] [batch-per-GPU] [save-frequency] [num-gpus]
set -Eeuo pipefail

# Editable run configuration. Command-line arguments override these values.
TRAINING_STEPS="${TRAINING_STEPS:-60000}"
TRAINING_NUM_GPUS="${TRAINING_NUM_GPUS:-4}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
export TRAINING_STEPS TRAINING_NUM_GPUS TRAINING_BATCH_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
  unset ENV_FILE
fi

export DATASET_ROOT="${MULTITASK_DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_multitask_v3}"
export DATASET_REPO_ID="${MULTITASK_DATASET_REPO_ID:-seer/lerobot_ee_multitask_v3}"
export LEROBOT_ROOT="${LEROBOT_ROOT:-/mnt/cfs/usr/jixiang/vla_learning/lerobot}"
export EVAL_SPLIT=0
export EVAL_STEPS=0
export MODEL_ROOT="${BASE_MODEL_ROOT:-/mnt/cfs/usr/jixiang/hf_models/pi05_base}"
export TRAINING_PROFILE=rtc
export RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
export OUTPUT_BASE="${MULTITASK_OUTPUT_BASE:-${PACKAGE_ROOT}/checkpoints/close_box_checkpoints}"
export JOB_NAME="${MULTITASK_JOB_NAME:-seer-pi05-close-box-rtc}"
export MLFLOW_ENABLE="${MLFLOW_ENABLE:-true}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-seer-pi05-close-box}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
