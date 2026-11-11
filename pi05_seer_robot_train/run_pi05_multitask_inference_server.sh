#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# MULTITASK_CHECKPOINT="${SCRIPT_DIR}/checkpoints/multitask-full-rtc/full-20261105-155455/checkpoints/010000/pretrained_model"

MULTITASK_CHECKPOINT="${SCRIPT_DIR}/checkpoints/multitask-full-rtc/full-20261106-170837/checkpoints/010000/pretrained_model"


export CHECKPOINT="${CHECKPOINT:-${MULTITASK_CHECKPOINT}}"
export RTC_MODE="${RTC_MODE:-trained}"

exec bash "${SCRIPT_DIR}/run_pi05_inference_server.sh" "$@"
