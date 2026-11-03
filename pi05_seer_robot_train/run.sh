#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-full}"
case "${MODE}" in
  check) DEFAULT_STEPS=0; DEFAULT_BATCH=1; DEFAULT_SAVE_FREQ=0 ;;
  smoke) DEFAULT_STEPS=2; DEFAULT_BATCH=1; DEFAULT_SAVE_FREQ=0 ;;
  full) DEFAULT_STEPS=10000; DEFAULT_BATCH=8; DEFAULT_SAVE_FREQ=0 ;;
  *)
    echo "Usage: $0 [check|smoke|full] [steps] [batch-per-GPU] [save-frequency]" >&2
    exit 2
    ;;
esac

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "Environment file not found: ${ENV_FILE}" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

STEPS="${2:-${STEPS:-${DEFAULT_STEPS}}}"
BATCH_SIZE="${3:-${BATCH_SIZE:-${DEFAULT_BATCH}}}"
SAVE_FREQ="${4:-${SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPOSITORY_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/cfs/usr/jixiang/hf_models/pi05_base}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/mnt/cfs/usr/jixiang/hf_models/paligemma-3b-pt-224-tokenizer}"
OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/checkpoints}"
DATASET_REPO_ID="${DATASET_REPO_ID:-seer_robot/lerobot_ee}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-0.05}"
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-true}"
TRACKIO_ENABLE="${TRACKIO_ENABLE:-false}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
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

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${LEROBOT_ROOT}/src/lerobot" ]] || fail "LeRobot source not found: ${LEROBOT_ROOT}"
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GPUS must be a positive integer"
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
echo "  steps: ${STEPS}; full fine-tuning: vision encoder and action expert unfrozen"
echo "  image augmentation: ${IMAGE_AUGMENTATION} (brightness, contrast, sharpness)"

"${TORCHRUN}" --standalone --nproc-per-node="${NUM_GPUS}" "${PACKAGE_ROOT}/train.py" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.streaming=false \
  --dataset.eval_split="${EVAL_SPLIT}" \
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
  --policy.chunk_size=50 \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.rtc_training_max_delay="${RTC_TRAIN_MAX_DELAY}" \
  --policy.push_to_hub=false \
  --output_dir="${RUN_OUTPUT}" \
  --job_name="seer-pi05-${MODE}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --seed=1000 \
  --accelerator.gradient_accumulation.steps="${GRAD_ACCUM_STEPS}" \
  --trackio.enable="${TRACKIO_ENABLE}" \
  --trackio.project=seer-pi05-eef

if [[ "${KEEP_CHECKPOINTS}" =~ ^[1-9][0-9]*$ ]]; then
  mapfile -t checkpoint_dirs < <(
    find "${RUN_OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\n' 2>/dev/null | sort -n
  )
  if (( ${#checkpoint_dirs[@]} > KEEP_CHECKPOINTS )); then
    for old in "${checkpoint_dirs[@]:0:${#checkpoint_dirs[@]}-KEEP_CHECKPOINTS}"; do
      rm -rf -- "${RUN_OUTPUT}/checkpoints/${old}"
    done
  fi
fi
