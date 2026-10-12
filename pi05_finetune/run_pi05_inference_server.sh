#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/mnt/cfs/usr/jixiang/vla_learning/lerobot/pi05_finetune}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/full-20260930-162041/checkpoints/008000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_v21}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/lerobot-pi05/bin/python}"
exec "$PYTHON_BIN" /mnt/cfs/usr/jixiang/vla_learning/lerobot/pi05_finetune/pi05_inference_server.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --port "${PORT:-18080}" \
  --device "${DEVICE:-cuda}" \
  --action-horizon "$ACTION_HORIZON" \
  --warmup-steps "$WARMUP_STEPS"
