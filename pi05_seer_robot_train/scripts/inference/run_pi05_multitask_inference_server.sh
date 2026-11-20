#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MULTITASK_CHECKPOINT="${PACKAGE_ROOT}/checkpoints/multitask-full-v1/full-20261118-071558/checkpoints/010000/pretrained_model"

export CHECKPOINT="${CHECKPOINT:-${MULTITASK_CHECKPOINT}}"
export RTC_MODE="${RTC_MODE:-trained}"

exec bash "${SCRIPT_DIR}/run_pi05_inference_server.sh" "$@"
