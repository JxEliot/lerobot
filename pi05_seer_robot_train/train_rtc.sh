#!/usr/bin/env bash
# Full-finetune PI05 with trained RTC action-prefix conditioning.
# Usage: bash train_rtc.sh [check|smoke|full] [steps] [batch-per-GPU] [save-frequency]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_PROFILE=rtc
export RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
export OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/checkpoints/rtc}"
export JOB_NAME="${JOB_NAME:-seer-pi05-rtc}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
