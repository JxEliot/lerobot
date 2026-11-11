#!/usr/bin/env bash
# Continue full fine-tuning from the existing trained-RTC weights on every merged episode.
# Usage: bash train_multitask_from_rtc.sh [check|smoke|full] [steps] [batch-per-GPU] [save-frequency]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
  unset ENV_FILE
fi

RTC_INIT_CHECKPOINT="${RTC_INIT_CHECKPOINT:-${SCRIPT_DIR}/checkpoints/rtc/full-20261103-185442/checkpoints/010000/pretrained_model}"
if [[ ! -f "${RTC_INIT_CHECKPOINT}/config.json" ]]; then
  echo "RTC checkpoint config not found: ${RTC_INIT_CHECKPOINT}/config.json" >&2
  exit 1
fi

export DATASET_ROOT="${MULTITASK_DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_multitask_v3}"
export DATASET_REPO_ID="${MULTITASK_DATASET_REPO_ID:-seer/lerobot_ee_multitask_v3}"
export LEROBOT_ROOT="${LEROBOT_ROOT:-/mnt/cfs/usr/jixiang/vla_learning/lerobot}"
export EVAL_SPLIT=0
export EVAL_STEPS=0
export MODEL_ROOT="${RTC_INIT_CHECKPOINT}"
export TRAINING_PROFILE=rtc
export RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
export OUTPUT_BASE="${MULTITASK_OUTPUT_BASE:-${SCRIPT_DIR}/checkpoints/multitask-from-rtc}"
export JOB_NAME="${MULTITASK_JOB_NAME:-seer-pi05-multitask-from-rtc}"
export MLFLOW_ENABLE=true
export TRACKIO_ENABLE=true
export MLFLOW_HOME="${MLFLOW_HOME:-/mnt/cfs/usr/jixiang/vla_learning/mlflow/pi05_seer_robot_train}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///${MLFLOW_HOME}/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${MLFLOW_HOME}/artifacts}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-seer-pi05-multitask}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
