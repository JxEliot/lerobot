#!/usr/bin/env bash
# Resume an interrupted PI05 run from one of its local checkpoints.
# Usage: bash scripts/train/resume_training.sh <run-dir> <target-total-steps> [checkpoint-step|last]
set -Eeuo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 <run-dir> <target-total-steps> [checkpoint-step|last]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${PACKAGE_ROOT}/.." && pwd)"

RUN_DIR_INPUT="$1"
TARGET_STEPS="$2"
CHECKPOINT_SELECTOR="${3:-last}"

if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
fi
MLFLOW_CREDS_FILE="${MLFLOW_CREDS_FILE:-${PACKAGE_ROOT}/config/.mlflow.env}"
if [[ -f "${MLFLOW_CREDS_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${MLFLOW_CREDS_FILE}"
  set +a
fi

LEROBOT_ROOT="${LEROBOT_ROOT:-${REPOSITORY_ROOT}}"
VENV_ROOT="${VENV_ROOT:-/root/venvs/lerobot-pi05}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
TORCHRUN="${TORCHRUN:-${VENV_ROOT}/bin/torchrun}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MLFLOW_ENABLE="${MLFLOW_ENABLE:-true}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-seer-pi05-eef}"
MLFLOW_RUN_NAME_PREFIX="${MLFLOW_RUN_NAME_PREFIX:-seer-pi05-resume}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${RUN_DIR_INPUT}" ]] || fail "Run directory not found: ${RUN_DIR_INPUT}"
RUN_DIR="$(cd -- "${RUN_DIR_INPUT}" && pwd)"
[[ -d "${LEROBOT_ROOT}/src/lerobot" ]] || fail "LeRobot source not found: ${LEROBOT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ -x "${TORCHRUN}" || "$(command -v "${TORCHRUN}" || true)" ]] || fail "torchrun not found: ${TORCHRUN}"
[[ "${TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "target-total-steps must be a positive integer"
[[ "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]] || fail "SAVE_FREQ must be a positive integer to preserve recovery points"

CHECKPOINTS_DIR="${RUN_DIR}/checkpoints"
case "${CHECKPOINT_SELECTOR}" in
  last)
    CHECKPOINT_DIR="${CHECKPOINTS_DIR}/last"
    ;;
  *[!0-9]*)
    fail "Checkpoint selector must be 'last' or a numeric step, got: ${CHECKPOINT_SELECTOR}"
    ;;
  *)
    CHECKPOINT_NAME="${CHECKPOINT_SELECTOR}"
    while (( ${#CHECKPOINT_NAME} < 6 )); do
      CHECKPOINT_NAME="0${CHECKPOINT_NAME}"
    done
    CHECKPOINT_DIR="${CHECKPOINTS_DIR}/${CHECKPOINT_NAME}"
    ;;
esac

[[ -d "${CHECKPOINT_DIR}" ]] || fail "Checkpoint directory not found: ${CHECKPOINT_DIR}"
TRAIN_CONFIG="${CHECKPOINT_DIR}/pretrained_model/train_config.json"
TRAINING_STEP="${CHECKPOINT_DIR}/training_state/training_step.json"
[[ -f "${TRAIN_CONFIG}" ]] || fail "Checkpoint training config not found: ${TRAIN_CONFIG}"
[[ -f "${TRAINING_STEP}" ]] || fail "Checkpoint training state not found: ${TRAINING_STEP}"

IFS=$'\t' read -r CURRENT_STEP CHECKPOINT_GPUS < <(
  "${PYTHON_BIN}" - "${TRAINING_STEP}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
step = int(state["step"])
parallelism = state.get("parallelism") or {}
if parallelism:
    cp = int(parallelism.get("ring_degree", 1)) * int(parallelism.get("ulysses_degree", 1))
    world_size = (
        int(parallelism.get("dp_replicate", 1))
        * int(parallelism.get("dp_shard", 1))
        * cp
    )
else:
    world_size = state.get("dp_world_size") or state.get("num_processes")
if world_size is None or int(world_size) < 1:
    world_size = 0
print(f"{step}\t{int(world_size)}")
PY
)
if [[ -n "${RESUME_NUM_GPUS:-}" ]]; then
  [[ "${RESUME_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || fail "RESUME_NUM_GPUS must be a positive integer"
  if (( CHECKPOINT_GPUS > 0 && RESUME_NUM_GPUS != CHECKPOINT_GPUS )); then
    fail "RESUME_NUM_GPUS must match checkpoint topology (${CHECKPOINT_GPUS})"
  fi
  CHECKPOINT_GPUS="${RESUME_NUM_GPUS}"
fi
(( CHECKPOINT_GPUS > 0 )) || fail "Checkpoint has no usable process topology; set RESUME_NUM_GPUS explicitly"
(( TARGET_STEPS > CURRENT_STEP )) || fail "Target steps (${TARGET_STEPS}) must exceed checkpoint step (${CURRENT_STEP})"

export PYTHONPATH="${LEROBOT_ROOT}/src:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/cfs/usr/jixiang/hf_models/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MLFLOW_ENABLE MLFLOW_EXPERIMENT_NAME MLFLOW_RUN_NAME_PREFIX
export TRAINING_PROFILE="${TRAINING_PROFILE:-resume}"

if [[ "${MLFLOW_ENABLE,,}" =~ ^(1|true|yes)$ ]]; then
  [[ -n "${MLFLOW_TRACKING_URI:-}" ]] || fail "Set MLFLOW_TRACKING_URI to the official MLflow tracking server"
fi

cd -- "${LEROBOT_ROOT}"
echo "Resuming Seer PI05 training"
echo "  run: ${RUN_DIR}"
echo "  checkpoint: ${CHECKPOINT_DIR} (step ${CURRENT_STEP})"
echo "  target total steps: ${TARGET_STEPS}; save frequency: ${SAVE_FREQ}"
echo "  GPUs: ${CHECKPOINT_GPUS} (from checkpoint topology)"

"${TORCHRUN}" --standalone --nproc-per-node="${CHECKPOINT_GPUS}" "${PACKAGE_ROOT}/train.py" \
  --resume=true \
  --config_path="${TRAIN_CONFIG}" \
  --output_dir="${RUN_DIR}" \
  --steps="${TARGET_STEPS}" \
  --save_freq="${SAVE_FREQ}"
