#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/lerobot-pi05/bin/python}"
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/checkpoints/full-20260930-162041/checkpoints/008000/pretrained_model}"
DATASET="${DATASET:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_v21}"
EPISODES="${EPISODES:-0 1 2 3 4}"
STEPS="${STEPS:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/inference_results/pi05.png}"

usage() {
  cat <<'EOF'
Usage:
  ./run_pi05_trajectory_verification.sh

Environment overrides:
  CHECKPOINT=/path/to/pretrained_model
  DATASET=/path/to/lerobot_dataset
  EPISODES="0 1 2 3 4"    Episode IDs to evaluate
  STEPS=0                  0 evaluates complete episodes; positive value caps frames
  DEVICE=cuda              Inference device
  OUTPUT=/path/to/pi05.png Output name prefix; per-episode plots and JSON go beside it
  PYTHON_BIN=/path/to/python

Examples:
  ./run_pi05_trajectory_verification.sh
  EPISODES="5 6" DEVICE=cuda ./run_pi05_trajectory_verification.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CHECKPOINT}/config.json" ]] || { echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${DATASET}/meta/info.json" ]] || { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 1; }

read -r -a EPISODE_ARGS <<< "${EPISODES}"
mkdir -p "$(dirname -- "${OUTPUT}")"

echo "PI05 complete-episode verification"
echo "  checkpoint: ${CHECKPOINT}"
echo "  dataset:    ${DATASET}"
echo "  episodes:   ${EPISODES}"
echo "  steps:      ${STEPS} (0 means complete episodes)"
echo "  device:     ${DEVICE}"
echo "  output:     ${OUTPUT}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_pi05_trajectory.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --episodes "${EPISODE_ARGS[@]}" \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --output "${OUTPUT}"
