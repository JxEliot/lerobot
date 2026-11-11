#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NO_RTC_CHECKPOINT="${SCRIPT_DIR}/checkpoints/no-rtc/full-20261103-185856/checkpoints/010000/pretrained_model"
export CHECKPOINT="${CHECKPOINT:-${NO_RTC_CHECKPOINT}}"
export RTC_MODE=off
export EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"
exec bash "${SCRIPT_DIR}/run_pi05_inference_server.sh" "$@"
