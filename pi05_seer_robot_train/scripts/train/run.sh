#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPOSITORY_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
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

case "${MODE}" in
  check) DEFAULT_STEPS=0; DEFAULT_BATCH=1; DEFAULT_SAVE_FREQ=0 ;;
  smoke) DEFAULT_STEPS=2; DEFAULT_BATCH=1; DEFAULT_SAVE_FREQ=0 ;;
  full)
    DEFAULT_STEPS="${TRAINING_STEPS:-10000}"
    DEFAULT_BATCH="${TRAINING_BATCH_SIZE:-64}"
    DEFAULT_SAVE_FREQ=5000
    ;;
  *)
    echo "Usage: $0 [check|smoke|full] [steps] [batch-per-GPU] [save-frequency] [num-gpus]" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == full ]]; then
  STEPS="${2:-${TRAINING_STEPS:-${STEPS:-${DEFAULT_STEPS}}}}"
  BATCH_SIZE="${3:-${TRAINING_BATCH_SIZE:-${BATCH_SIZE:-${DEFAULT_BATCH}}}}"
else
  STEPS="${2:-${DEFAULT_STEPS}}"
  BATCH_SIZE="${3:-${DEFAULT_BATCH}}"
fi
if [[ "${MODE}" == full ]]; then
  SAVE_FREQ="${4:-${SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}}"
else
  SAVE_FREQ="${4:-${DEFAULT_SAVE_FREQ}}"
fi
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPOSITORY_ROOT}}"
VENV_ROOT="${VENV_ROOT:-/root/venvs/lerobot-pi05}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
TORCHRUN="${TORCHRUN:-${VENV_ROOT}/bin/torchrun}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/cfs/usr/jixiang/hf_models/pi05_base}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/mnt/cfs/usr/jixiang/hf_models/paligemma-3b-pt-224-tokenizer}"
OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/checkpoints}"
DATASET_REPO_ID="${DATASET_REPO_ID:-seer_robot/lerobot_ee}"
NUM_GPUS="${5:-${TRAINING_NUM_GPUS:-${NUM_GPUS:-1}}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
TRAINING_PROFILE="${TRAINING_PROFILE:-default}"
case "${TRAINING_PROFILE}" in
  no-rtc) RTC_TRAIN_MAX_DELAY=0 ;;
  rtc) RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}" ;;
  *) RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-0}" ;;
esac
OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-30000}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
LOG_FREQ="${LOG_FREQ:-200}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-0.05}"
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-true}"
MLFLOW_ENABLE="${MLFLOW_ENABLE:-true}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-seer-pi05-eef}"
MLFLOW_RUN_NAME_PREFIX="${MLFLOW_RUN_NAME_PREFIX:-seer-pi05}"
JOB_NAME="${JOB_NAME:-seer-pi05-${TRAINING_PROFILE}-${MODE}}"
IMAGE_TRANSFORMS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}}}'

export PYTHONPATH="${LEROBOT_ROOT}/src:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/cfs/usr/jixiang/hf_models/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MLFLOW_ENABLE MLFLOW_EXPERIMENT_NAME MLFLOW_RUN_NAME_PREFIX

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${LEROBOT_ROOT}/src/lerobot" ]] || fail "LeRobot source not found: ${LEROBOT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GPUS must be a positive integer"
[[ "${CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "CHUNK_SIZE must be a positive integer"
[[ "${N_ACTION_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "N_ACTION_STEPS must be a positive integer"
[[ "${RTC_TRAIN_MAX_DELAY}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "RTC_TRAIN_MAX_DELAY must be a non-negative integer"
(( N_ACTION_STEPS <= CHUNK_SIZE )) || fail "N_ACTION_STEPS cannot exceed CHUNK_SIZE"
(( RTC_TRAIN_MAX_DELAY < CHUNK_SIZE )) || fail "RTC_TRAIN_MAX_DELAY must be smaller than CHUNK_SIZE"
if [[ "${MLFLOW_ENABLE,,}" =~ ^(1|true|yes)$ ]]; then
  [[ -n "${MLFLOW_TRACKING_URI:-}" ]] || fail "Set MLFLOW_TRACKING_URI to the official MLflow tracking server"
fi
if [[ "${TRAINING_PROFILE}" == rtc ]] && (( RTC_TRAIN_MAX_DELAY == 0 )); then
  fail "RTC_TRAIN_MAX_DELAY must be greater than zero for train_rtc.sh"
fi
"${PYTHON_BIN}" "${PACKAGE_ROOT}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --tokenizer-root "${TOKENIZER_ROOT}" \
  --num-gpus "${NUM_GPUS}"

if [[ "${MODE}" == "check" ]]; then
  echo "Preflight checks passed; training was not started."
  exit 0
fi
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "steps must be a positive integer"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "batch-per-GPU must be a positive integer"
[[ "${GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "GRAD_ACCUM_STEPS must be a positive integer"
[[ -x "${TORCHRUN}" || "$(command -v "${TORCHRUN}" || true)" ]] || fail "torchrun not found: ${TORCHRUN}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_OUTPUT="${OUTPUT_DIR:-${OUTPUT_BASE}/${MODE}-${RUN_STAMP}}"
[[ ! -e "${RUN_OUTPUT}" ]] || fail "Output already exists: ${RUN_OUTPUT}"
mkdir -p "${OUTPUT_BASE}"
cd "${LEROBOT_ROOT}"

echo "Starting Seer PI05 ${MODE} run"
echo "  dataset: ${DATASET_ROOT} (${DATASET_REPO_ID})"
echo "  output: ${RUN_OUTPUT}"
echo "  GPUs: ${NUM_GPUS}; batch/GPU: ${BATCH_SIZE}; grad accumulation: ${GRAD_ACCUM_STEPS}"
echo "  effective batch: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))"
echo "  steps: ${STEPS}; chunk=${CHUNK_SIZE}; action steps=${N_ACTION_STEPS}; RTC max delay=${RTC_TRAIN_MAX_DELAY}"
echo "  optimizer LR: ${OPTIMIZER_LR}; full fine-tuning: vision encoder and action expert unfrozen"
echo "  image augmentation: ${IMAGE_AUGMENTATION} (brightness, contrast, sharpness)"

read -r TRACKIO_CONFIG_SUPPORTED IMAGE_BACKEND_SUPPORTED < <(
  "${PYTHON_BIN}" - <<'PY'
from dataclasses import fields

from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransformsConfig

config_fields = {field.name for field in fields(TrainPipelineConfig)}
image_fields = {field.name for field in fields(ImageTransformsConfig)}
print(int("trackio" in config_fields), int("backend" in image_fields))
PY
)
logger_args=()
if [[ "${TRACKIO_CONFIG_SUPPORTED}" == 1 ]]; then
  logger_args+=(--trackio.enable=false)
fi
image_backend_args=()
if [[ "${IMAGE_BACKEND_SUPPORTED}" == 1 ]]; then
  image_backend_args+=(--dataset.image_transforms.backend=gpu)
fi

"${TORCHRUN}" --standalone --nproc-per-node="${NUM_GPUS}" "${PACKAGE_ROOT}/train.py" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.streaming=false \
  --dataset.eval_split="${EVAL_SPLIT}" \
  --dataset.image_transforms.enable="${IMAGE_AUGMENTATION}" \
  "${image_backend_args[@]}" \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_ROOT}" \
  --policy.text_tokenizer_name="${TOKENIZER_ROOT}" \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.use_peft=false \
  --policy.use_relative_actions=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.rtc_training_max_delay="${RTC_TRAIN_MAX_DELAY}" \
  --policy.optimizer_lr="${OPTIMIZER_LR}" \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}" \
  --policy.push_to_hub=false \
  --output_dir="${RUN_OUTPUT}" \
  --job_name="${JOB_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_steps="${EVAL_STEPS}" \
  --seed=1000 \
  --accelerator.gradient_accumulation.steps="${GRAD_ACCUM_STEPS}" \
  "${logger_args[@]}"
