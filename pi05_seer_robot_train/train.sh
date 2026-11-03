#!/usr/bin/env bash
# Full-finetune PI05 without RTC training prefixes.
# Usage: bash train.sh [check|smoke|full] [steps] [batch-per-GPU] [save-frequency]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_PROFILE=no-rtc
export RTC_TRAIN_MAX_DELAY=0
export OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/checkpoints/no-rtc}"
export JOB_NAME="${JOB_NAME:-seer-pi05-no-rtc}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
