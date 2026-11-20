#!/usr/bin/env bash
# Fine-tune PI05 on only the right-flap episodes from the Seer v3 dataset.
# Usage: bash scripts/train/train_right_flap_only.sh [check|smoke|full] [steps] [batch-per-GPU] [save-frequency] [num-gpus]
set -Eeuo pipefail

# Editable run configuration. A zero step count keeps the data-exposure-matched default.
TRAINING_STEPS="${TRAINING_STEPS:-0}"
TRAINING_NUM_GPUS="${TRAINING_NUM_GPUS:-4}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"

MODE="${1:-full}"
case "${MODE}" in
  check) DEFAULT_STEPS=0; DEFAULT_BATCH=64; DEFAULT_SAVE_FREQ=0 ;;
  smoke) DEFAULT_STEPS=2; DEFAULT_BATCH=1; DEFAULT_SAVE_FREQ=0 ;;
  full) DEFAULT_BATCH=64; DEFAULT_SAVE_FREQ=5000 ;;
  *)
    echo "Usage: $0 [check|smoke|full] [steps] [batch-per-GPU] [save-frequency] [num-gpus]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${PACKAGE_ROOT}/.." && pwd)"

if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
  unset ENV_FILE
fi
MLFLOW_CREDS_FILE="${MLFLOW_CREDS_FILE:-${PACKAGE_ROOT}/config/.mlflow.env}"
if [[ -f "${MLFLOW_CREDS_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${MLFLOW_CREDS_FILE}"
  set +a
fi

TASK_TEXT="Fold the right flap of the box."
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPOSITORY_ROOT}}"
VENV_ROOT="${VENV_ROOT:-/root/venvs/lerobot-pi05}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
TORCHRUN="${TORCHRUN:-${VENV_ROOT}/bin/torchrun}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_close_box}"
DATASET_REPO_ID="${DATASET_REPO_ID:-seer/lerobot_ee_close_box}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/cfs/usr/jixiang/hf_models/pi05_base}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/mnt/cfs/usr/jixiang/hf_models/paligemma-3b-pt-224-tokenizer}"
OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/checkpoints/right-flap-only-rtc}"

NUM_GPUS="${5:-${TRAINING_NUM_GPUS:-${NUM_GPUS:-4}}}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
if [[ "${MODE}" == full ]]; then
  BATCH_SIZE="${3:-${TRAINING_BATCH_SIZE:-${DEFAULT_BATCH}}}"
else
  BATCH_SIZE="${3:-${DEFAULT_BATCH}}"
fi
RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-30000}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
LOG_FREQ="${LOG_FREQ:-20}"
if [[ "${MODE}" == full ]]; then
  SAVE_FREQ="${4:-${SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}}"
else
  SAVE_FREQ="${4:-${DEFAULT_SAVE_FREQ}}"
fi
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-true}"
JOB_NAME="${JOB_NAME:-seer-pi05-right-flap-only-full-rtc}"
IMAGE_TRANSFORMS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}}}'

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/cfs/usr/jixiang/hf_models/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONPATH="${LEROBOT_ROOT}/src:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export MLFLOW_ENABLE="${MLFLOW_ENABLE:-true}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-seer-pi05-single-task}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ -x "${TORCHRUN}" || "$(command -v "${TORCHRUN}" || true)" ]] || fail "torchrun not found: ${TORCHRUN}"
[[ -d "${LEROBOT_ROOT}/src/lerobot" ]] || fail "LeRobot source not found: ${LEROBOT_ROOT}"
[[ -d "${DATASET_ROOT}" ]] || fail "Dataset not found: ${DATASET_ROOT}"
[[ -d "${MODEL_ROOT}" ]] || fail "Base model not found: ${MODEL_ROOT}"
[[ -d "${TOKENIZER_ROOT}" ]] || fail "Tokenizer not found: ${TOKENIZER_ROOT}"
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GPUS must be a positive integer"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "batch-per-GPU must be a positive integer"
[[ "${GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "GRAD_ACCUM_STEPS must be a positive integer"
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || fail "NUM_WORKERS must be a non-negative integer"
[[ "${PREFETCH_FACTOR}" =~ ^[1-9][0-9]*$ ]] || fail "PREFETCH_FACTOR must be a positive integer"
[[ "${RTC_TRAIN_MAX_DELAY}" =~ ^[1-9][0-9]*$ ]] || fail "RTC_TRAIN_MAX_DELAY must be a positive integer"
[[ "${CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "CHUNK_SIZE must be a positive integer"
[[ "${N_ACTION_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "N_ACTION_STEPS must be a positive integer"
(( RTC_TRAIN_MAX_DELAY > 0 && RTC_TRAIN_MAX_DELAY < CHUNK_SIZE )) || fail "RTC_TRAIN_MAX_DELAY must be in [1, CHUNK_SIZE)"
(( N_ACTION_STEPS > 0 && N_ACTION_STEPS <= CHUNK_SIZE )) || fail "N_ACTION_STEPS must be in [1, CHUNK_SIZE]"

EPISODE_SELECTION="$("${PYTHON_BIN}" - "${DATASET_ROOT}" "${TASK_TEXT}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
task = sys.argv[2]
info = json.loads((root / "meta" / "info.json").read_text())
episodes = []
selected_frames = 0
for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
    for row in pq.read_table(path, columns=["episode_index", "tasks", "length"]).to_pylist():
        if task in (row["tasks"] or []):
            episodes.append(int(row["episode_index"]))
            selected_frames += int(row["length"])
episodes.sort()
if not episodes:
    raise SystemExit(f"No episodes found for exact task text: {task!r}")
if len(set(episodes)) != len(episodes):
    raise SystemExit("Duplicate episode indices found in episode metadata")
print(f"{len(episodes)}\t{selected_frames}\t{int(info['total_frames'])}\t{json.dumps(episodes, separators=(',', ':'))}")
PY
)"
IFS=$'\t' read -r SELECTED_EPISODES SELECTED_FRAMES DATASET_FRAMES EPISODE_IDS <<< "${EPISODE_SELECTION}"
[[ "${SELECTED_EPISODES}" =~ ^[1-9][0-9]*$ ]] || fail "Could not parse the selected episode count"
[[ "${DATASET_FRAMES}" =~ ^[1-9][0-9]*$ ]] || fail "Could not parse the dataset frame count"

REFERENCE_STEPS=10000
REFERENCE_BATCH_SIZE=64
REFERENCE_NUM_GPUS=4
REFERENCE_GRAD_ACCUM=1
REFERENCE_SAMPLES=$((REFERENCE_STEPS * REFERENCE_BATCH_SIZE * REFERENCE_NUM_GPUS * REFERENCE_GRAD_ACCUM))
TASK_SAMPLE_BUDGET=$((REFERENCE_SAMPLES * SELECTED_FRAMES / DATASET_FRAMES))
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))
MATCHED_STEPS=$(((TASK_SAMPLE_BUDGET + GLOBAL_BATCH_SIZE / 2) / GLOBAL_BATCH_SIZE))
if [[ "${MODE}" == "full" ]]; then
  DEFAULT_STEPS="${TRAINING_STEPS}"
  if (( DEFAULT_STEPS == 0 )); then
    DEFAULT_STEPS="${MATCHED_STEPS}"
  fi
fi
STEPS="${2:-${DEFAULT_STEPS:-${STEPS:-0}}}"

if [[ "${MODE}" != "check" ]]; then
  [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "steps must be a positive integer"
fi

"${PYTHON_BIN}" "${PACKAGE_ROOT}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --tokenizer-root "${TOKENIZER_ROOT}" \
  --num-gpus "${NUM_GPUS}"

echo "Selected task: ${TASK_TEXT}"
echo "Selected episodes: ${SELECTED_EPISODES}; selected frames: ${SELECTED_FRAMES}/${DATASET_FRAMES}"
echo "Episode IDs: ${EPISODE_IDS}"
echo "Data-exposure-matched steps: ${MATCHED_STEPS} (reference: 10,000 steps x 4 GPUs x batch 64)"
if [[ "${MODE}" == "check" ]]; then
  echo "Preflight passed; training was not started."
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  STEPS="${2:-2}"
fi

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_OUTPUT="${OUTPUT_DIR:-${OUTPUT_BASE}/right-flap-${RUN_STAMP}}"
[[ ! -e "${RUN_OUTPUT}" ]] || fail "Output already exists: ${RUN_OUTPUT}"
mkdir -p "${OUTPUT_BASE}"
cd "${LEROBOT_ROOT}"

echo "Starting Seer PI05 single-task full fine-tuning"
echo "  dataset: ${DATASET_ROOT} (${DATASET_REPO_ID})"
echo "  base model: ${MODEL_ROOT}"
echo "  output: ${RUN_OUTPUT}"
echo "  GPUs: ${NUM_GPUS}; batch/GPU: ${BATCH_SIZE}; grad accumulation: ${GRAD_ACCUM_STEPS}"
echo "  global batch: ${GLOBAL_BATCH_SIZE}; steps: ${STEPS}; save frequency: ${SAVE_FREQ}"
echo "  workers/GPU: ${NUM_WORKERS}; prefetch factor: ${PREFETCH_FACTOR}"
echo "  full fine-tuning; RTC max delay: ${RTC_TRAIN_MAX_DELAY}; image augmentation: ${IMAGE_AUGMENTATION}"

"${TORCHRUN}" --standalone --nproc-per-node="${NUM_GPUS}" "${PACKAGE_ROOT}/train.py" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="${EPISODE_IDS}" \
  --dataset.streaming=false \
  --dataset.eval_split=0.0 \
  --dataset.image_transforms.enable="${IMAGE_AUGMENTATION}" \
  --dataset.image_transforms.backend=gpu \
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
  --prefetch_factor="${PREFETCH_FACTOR}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_steps=0 \
  --seed=1000 \
  --accelerator.gradient_accumulation.steps="${GRAD_ACCUM_STEPS}" \
  --trackio.enable=false
