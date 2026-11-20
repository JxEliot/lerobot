#!/usr/bin/env bash
# Copy this file to train_<experiment>.sh and edit the configuration block.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${ENV_FILE:-}" && -f "${PACKAGE_ROOT}/config/server.env" ]]; then
  ENV_FILE="${PACKAGE_ROOT}/config/server.env"
fi
if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
  unset ENV_FILE
fi

# Editable run configuration. Command-line arguments override steps, batch size, and GPU count.
TRAINING_STEPS="${TRAINING_STEPS:-60000}"
TRAINING_NUM_GPUS="${TRAINING_NUM_GPUS:-4}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
TRAINING_PROFILE="${TRAINING_PROFILE:-rtc}" # rtc or no-rtc
RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
TRAINING_DATASET_ROOT="${TRAINING_DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee}"
TRAINING_DATASET_REPO_ID="${TRAINING_DATASET_REPO_ID:-seer_robot/lerobot_ee}"
TRAINING_OUTPUT_BASE="${TRAINING_OUTPUT_BASE:-${PACKAGE_ROOT}/checkpoints/custom-run}"
TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-seer-pi05-custom-run}"

export TRAINING_STEPS TRAINING_NUM_GPUS TRAINING_BATCH_SIZE
export TRAINING_PROFILE RTC_TRAIN_MAX_DELAY
export DATASET_ROOT="${TRAINING_DATASET_ROOT}"
export DATASET_REPO_ID="${TRAINING_DATASET_REPO_ID}"
export OUTPUT_BASE="${TRAINING_OUTPUT_BASE}"
export JOB_NAME="${TRAINING_JOB_NAME}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
