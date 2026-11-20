#!/usr/bin/env bash
# Full-finetune PI05 with trained RTC action-prefix conditioning.
# Usage: bash scripts/train/train_rtc.sh [check|smoke|full] [steps] [batch-per-GPU] [save-frequency] [num-gpus]
set -Eeuo pipefail

# Editable run configuration. Command-line arguments override these values.
TRAINING_STEPS="${TRAINING_STEPS:-60000}"
TRAINING_NUM_GPUS="${TRAINING_NUM_GPUS:-4}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
export TRAINING_STEPS TRAINING_NUM_GPUS TRAINING_BATCH_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export TRAINING_PROFILE=rtc
export RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/checkpoints/rtc}"
export JOB_NAME="${JOB_NAME:-seer-pi05-rtc}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
