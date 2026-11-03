#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
export CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/no-rtc/full-20261013-132623/checkpoints/010000/pretrained_model}"
export RTC_MODE=guided
export EXECUTION_HORIZON="${EXECUTION_HORIZON:-15}"
exec bash "$SCRIPT_DIR/run_pi05_inference_server.sh" "$@"
