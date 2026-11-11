#!/usr/bin/env bash
# Start the local-only MLflow dashboard. Use an SSH tunnel to access it remotely.
set -Eeuo pipefail

VENV_ROOT="${VENV_ROOT:-/root/venvs/lerobot-pi05}"
MLFLOW_BIN="${MLFLOW_BIN:-${VENV_ROOT}/bin/mlflow}"
MLFLOW_HOME="${MLFLOW_HOME:-/mnt/cfs/usr/jixiang/vla_learning/mlflow/pi05_seer_robot_train}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///${MLFLOW_HOME}/mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${MLFLOW_HOME}/artifacts}"
MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

[[ -x "${MLFLOW_BIN}" ]] || { echo "MLflow executable not found: ${MLFLOW_BIN}" >&2; exit 1; }
mkdir -p "${MLFLOW_HOME}" "${MLFLOW_ARTIFACT_ROOT}"

exec "${MLFLOW_BIN}" server \
  --backend-store-uri "${MLFLOW_TRACKING_URI}" \
  --default-artifact-root "file://${MLFLOW_ARTIFACT_ROOT}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
