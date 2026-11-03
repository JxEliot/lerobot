#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/mnt/cfs/usr/jixiang/vla_learning/lerobot/pi05_finetune}"
# CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/full-20260930-162041/checkpoints/008000/pretrained_model}"
# CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/no-rtc/full-20261013-132623/checkpoints/010000/pretrained_model}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/rtc/full-20261013-133656/checkpoints/010000/pretrained_model}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_v21}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-25}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/lerobot-pi05/bin/python}"
exec "$PYTHON_BIN" /mnt/cfs/usr/jixiang/vla_learning/lerobot/pi05_finetune/pi05_inference_server.py \
  --rtc-mode "${RTC_MODE:-trained}" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --port "${PORT:-18080}" \
  --device "${DEVICE:-cuda}" \
  --action-horizon "$ACTION_HORIZON" \
  --execution-horizon "$EXECUTION_HORIZON" \
  --warmup-steps "$WARMUP_STEPS"
