#!/usr/bin/env bash
# Usage: bash run_pi05_finetune_rtc.sh [check|smoke|full] [steps] [batch-per-gpu] [save-freq]
# Start both comparison runs from the same MODEL_ROOT and dataset.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "ERROR: Comparison scripts require a fresh run; RESUME_FROM retains checkpoint training settings." >&2
  exit 1
fi
export RTC_TRAIN_MAX_DELAY=10
export OUTPUT_BASE="${OUTPUT_BASE:-$SCRIPT_DIR/checkpoints/rtc}"
echo "PI05 comparison: rtc; RTC_TRAIN_MAX_DELAY=$RTC_TRAIN_MAX_DELAY"
exec bash "$SCRIPT_DIR/run_pi05_finetune.sh" "$@"
