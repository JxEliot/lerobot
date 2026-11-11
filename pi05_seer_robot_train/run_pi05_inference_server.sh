#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

VENV_ROOT="${VENV_ROOT:-/root/venvs/lerobot-pi05}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}}"
RTC_CHECKPOINT="${SCRIPT_DIR}/checkpoints/rtc/full-20261103-185442/checkpoints/010000/pretrained_model"
# RTC_CHECKPOINT="${SCRIPT_DIR}/checkpoints/multitask-full-rtc/full-20261105-155455/checkpoints/010000/pretrained_model"
CHECKPOINT="${CHECKPOINT:-${RTC_CHECKPOINT}}"
RTC_MODE="${RTC_MODE:-trained}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
DEVICE="${DEVICE:-cuda}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-25}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
WARMUP_TASK="${WARMUP_TASK:-Pick up the part and place it into the box.}"
HF_HOME="${HF_HOME:-/mnt/cfs/usr/jixiang/hf_models/hub}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${LEROBOT_ROOT}/src/lerobot" ]] || { echo "LeRobot source not found: ${LEROBOT_ROOT}" >&2; exit 1; }
[[ -f "${CHECKPOINT}/config.json" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }

export HF_HOME
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${LEROBOT_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/pi05_robot_inference_server.py" \
  --rtc-mode "${RTC_MODE}" \
  --checkpoint "${CHECKPOINT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --action-horizon "${ACTION_HORIZON}" \
  --execution-horizon "${EXECUTION_HORIZON}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --warmup-task "${WARMUP_TASK}" \
  "$@"
