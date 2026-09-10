#!/usr/bin/env bash
# Full-parameter Pi0.5 finetuning for the dual-arm pick-and-place dataset.
# Usage: bash run_pi05_finetune.sh [check|smoke|full] [steps] [batch-per-gpu] [save-freq]
set -Eeuo pipefail

MODE="${1:-full}"
case "${MODE}" in
  check)
    DEFAULT_STEPS=0
    DEFAULT_BATCH=1
    DEFAULT_SAVE_FREQ=0
    ;;
  smoke)
    DEFAULT_STEPS=2
    DEFAULT_BATCH=1
    DEFAULT_SAVE_FREQ=2
    ;;
  full)
    DEFAULT_STEPS=8000
    DEFAULT_BATCH=64
    DEFAULT_SAVE_FREQ=2000
    ;;
  *)
    echo "Usage: $0 [check|smoke|full] [steps] [batch-per-gpu] [save-freq]" >&2
    exit 2
    ;;
esac

STEPS="${2:-${STEPS:-${DEFAULT_STEPS}}}"
BATCH_SIZE="${3:-${BATCH_SIZE:-${DEFAULT_BATCH}}}"
SAVE_FREQ="${4:-${SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}}"

LEROBOT_ROOT="${LEROBOT_ROOT:-/mnt/cfs/vyi1fr/usr/jixiang/lerobot}"
VENV="${VENV:-/root/venvs/lerobot-pi05}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/vyi1fr/usr/jixiang/Isaac-GR00T/seer_robot/datasets/pick_and_place_lerobotv2/pick_and_place_v3_pi05_gr00t_eef}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/cfs/vyi1fr/usr/jixiang/hf_models/pi05_base_local}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/mnt/cfs/vyi1fr/usr/jixiang/hf_models/paligemma-3b-pt-224-tokenizer}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/cfs/vyi1fr/usr/jixiang/lerobot/pi05_finetune/checkpoints}"
DATASET_REPO_ID="${DATASET_REPO_ID:-seer_robot/pick_and_place_v3}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
# Train hard action-prefix conditioning for RTC. At 20 FPS, 10 steps covers
# up to 0.5 seconds of controller/inference delay.
RTC_TRAIN_MAX_DELAY="${RTC_TRAIN_MAX_DELAY:-10}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-pi05-pick-and-place}"
RESUME_FROM="${RESUME_FROM:-}"

export PATH="${VENV}/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/cfs/vyi1fr/usr/jixiang/hf_models/hub}"
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

[[ -x "${VENV}/bin/python" ]] || fail "Python environment not found: ${VENV}"
[[ -x "${VENV}/bin/lerobot-train" ]] || fail "lerobot-train not found in ${VENV}"
[[ -d "${LEROBOT_ROOT}" ]] || fail "LeRobot source not found: ${LEROBOT_ROOT}"
[[ -f "${MODEL_ROOT}/config.json" ]] || fail "Pi0.5 config not found: ${MODEL_ROOT}/config.json"
[[ -f "${MODEL_ROOT}/model.safetensors" ]] || fail "Pi0.5 weights not found: ${MODEL_ROOT}/model.safetensors"
[[ -f "${TOKENIZER_ROOT}/tokenizer.json" ]] || fail "PaliGemma tokenizer not found: ${TOKENIZER_ROOT}"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || fail "Dataset metadata not found: ${DATASET_ROOT}/meta/info.json"
[[ -f "${DATASET_ROOT}/meta/stats.json" ]] || fail "Dataset statistics not found: ${DATASET_ROOT}/meta/stats.json"

"${VENV}/bin/python" - "${DATASET_ROOT}" "${TOKENIZER_ROOT}" "${NUM_GPUS}" <<'PY'
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

dataset_root, tokenizer_root, requested_gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
info = json.loads((Path(dataset_root) / "meta/info.json").read_text())
stats = json.loads((Path(dataset_root) / "meta/stats.json").read_text())
features = info["features"]

for key in ("observation.state", "action"):
    if key not in features:
        raise SystemExit(f"Missing dataset feature: {key}")
    if key not in stats or not {"q01", "q99"}.issubset(stats[key]):
        raise SystemExit(f"Missing Pi0.5 quantile statistics for: {key}")

state_dim = features["observation.state"]["shape"][0]
action_dim = features["action"]["shape"][0]
if state_dim > 32 or action_dim > 32:
    raise SystemExit(f"Pi0.5 supports at most 32 dims; got state={state_dim}, action={action_dim}")

image_keys = sorted(k for k, value in features.items() if value.get("dtype") in ("image", "video"))
if not image_keys:
    raise SystemExit("Dataset has no image/video observations")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
if len(tokenizer) < 250_000:
    raise SystemExit(f"Unexpected PaliGemma tokenizer vocabulary size: {len(tokenizer)}")

visible_gpus = torch.cuda.device_count()
if requested_gpus > visible_gpus:
    raise SystemExit(f"Requested {requested_gpus} GPUs, but PyTorch sees {visible_gpus}")
for index in range(requested_gpus):
    major, minor = torch.cuda.get_device_capability(index)
    if major < 8:
        raise SystemExit(f"GPU {index} does not provide suitable bfloat16 support: sm_{major}{minor}")

print(f"Preflight OK: episodes={info['total_episodes']}, frames={info['total_frames']}, fps={info['fps']}")
print(f"Features: state={state_dim}, action={action_dim}, cameras={image_keys}")
print(f"Tokenizer: {type(tokenizer).__name__}, vocab={len(tokenizer)}")
print(f"CUDA: {requested_gpus}/{visible_gpus} GPU(s) selected")
PY

if [[ "${MODE}" == "check" ]]; then
  echo "Configuration and data checks passed; no training was started."
  exit 0
fi

if [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  fail "steps must be a positive integer, got: ${STEPS}"
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  fail "batch-per-gpu must be a positive integer, got: ${BATCH_SIZE}"
fi

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  RUN_OUTPUT="${OUTPUT_DIR}"
else
  RUN_OUTPUT="${OUTPUT_BASE}/${MODE}-${RUN_STAMP}"
fi

if [[ -e "${RUN_OUTPUT}" && -z "${RESUME_FROM}" ]]; then
  fail "Output already exists; choose a new OUTPUT_DIR or set RESUME_FROM: ${RUN_OUTPUT}"
fi
if [[ -n "${RESUME_FROM}" && ! -e "${RESUME_FROM}" ]]; then
  fail "Resume checkpoint/config does not exist: ${RESUME_FROM}"
fi

mkdir -p "${OUTPUT_BASE}"
cd "${LEROBOT_ROOT}"

COMMON_ARGS=(
  "--output_dir=${RUN_OUTPUT}"
  "--job_name=pi05-pick-and-place-${MODE}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=10"
  "--seed=1000"
  "--accelerator.gradient_accumulation.steps=${GRAD_ACCUM_STEPS}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--wandb.project=${WANDB_PROJECT}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  POLICY_ARGS=(
    "--resume=true"
    "--config_path=${RESUME_FROM}"
  )
else
  POLICY_ARGS=(
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.return_uint8=true"
    "--policy.type=pi05"
    "--policy.pretrained_path=${MODEL_ROOT}"
    "--policy.text_tokenizer_name=${TOKENIZER_ROOT}"
    "--policy.freeze_vision_encoder=false"
    "--policy.train_expert_only=false"
    "--policy.gradient_checkpointing=true"
    "--policy.dtype=bfloat16"
    "--policy.device=cuda"
    "--policy.use_relative_actions=false"
    "--policy.use_gr00t_eef_relative_actions=true"
    "--policy.chunk_size=50"
    "--policy.n_action_steps=${N_ACTION_STEPS}"
    "--policy.rtc_training_max_delay=${RTC_TRAIN_MAX_DELAY}"
    "--policy.push_to_hub=false"
  )
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))
echo "Starting Pi0.5 ${MODE} run"
echo "  output:          ${RUN_OUTPUT}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  batch/GPU:       ${BATCH_SIZE}"
echo "  grad accumulation: ${GRAD_ACCUM_STEPS}"
echo "  effective batch: ${EFFECTIVE_BATCH}"
echo "  steps:           ${STEPS}"
echo "  RTC max delay:   ${RTC_TRAIN_MAX_DELAY}"

exec "${VENV}/bin/torchrun" \
  --standalone \
  --nproc-per-node="${NUM_GPUS}" \
  "${VENV}/bin/lerobot-train" \
  "${POLICY_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
