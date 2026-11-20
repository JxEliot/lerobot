"""MLflow logging for policy training."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
from collections.abc import Mapping
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, Mapping):
        flattened: dict[str, str] = {}
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_params(item, child_prefix))
        return flattened

    if isinstance(value, (list, tuple)):
        serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    elif value is None:
        serialized = "null"
    else:
        serialized = str(value)
    return {prefix[:250]: serialized[:500]} if prefix else {}


def _parse_tags(raw: str | None) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in (raw or "").split(","):
        key, separator, value = item.partition("=")
        if key.strip():
            tags[key.strip()[:250]] = (value.strip() if separator else "")[:500]
    return tags


def _resolve_experiment(client: Any, experiment_name: str) -> tuple[str, str]:
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        experiment = client.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"MLflow experiment id does not exist: {experiment_id}")
        return experiment.experiment_id, experiment.name

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id, experiment.name

    fallback_id = os.environ.get("MLFLOW_FALLBACK_EXPERIMENT_ID")
    if fallback_id:
        experiment = client.get_experiment(fallback_id)
        if experiment is None:
            raise ValueError(f"MLFLOW_FALLBACK_EXPERIMENT_ID does not exist: {fallback_id}")
        return experiment.experiment_id, experiment.name

    try:
        experiment_id = client.create_experiment(experiment_name)
    except Exception:
        # Concurrent jobs can race while creating the same experiment.
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise
        return experiment.experiment_id, experiment.name
    return experiment_id, experiment_name


class MLflowLogger:
    """Log training metrics and configuration to a configured MLflow server."""

    def __init__(self, cfg: TrainPipelineConfig):
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI must point to the official MLflow tracking server")

        self._mlflow = mlflow
        self._finished = False
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg.job_name)
        client = mlflow.MlflowClient()
        experiment_id, resolved_experiment_name = _resolve_experiment(client, experiment_name)

        run_prefix = os.environ.get("MLFLOW_RUN_NAME_PREFIX", "seer-pi05").strip()
        run_name = os.environ.get("MLFLOW_RUN_NAME")
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{run_prefix}.{cfg.job_name}.{timestamp}" if run_prefix else f"{cfg.job_name}.{timestamp}"

        tags = _parse_tags(os.environ.get("MLFLOW_TAGS"))
        tags.update(
            {
                "job_name": str(cfg.job_name),
                "dataset.repo_id": cfg.dataset.repo_id if cfg.dataset else "",
                "dataset.root": str(cfg.dataset.root) if cfg.dataset else "",
                "training_profile": os.environ.get("TRAINING_PROFILE", "default"),
            }
        )
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags)
        self._run_id = run.info.run_id

        params = {
            key: value
            for key, value in _flatten_params(cfg.to_dict()).items()
            if key not in {"output_dir", "checkpoint_path"}
            and not key.endswith(".pretrained_path")
            and not key.startswith("trackio.")
        }
        params.update(
            {
                "mlflow.experiment_name": resolved_experiment_name,
                "mlflow.run_name": run_name,
            }
        )
        items = list(params.items())
        for offset in range(0, len(items), 100):
            mlflow.log_params(dict(items[offset : offset + 100]))

        atexit.register(self._finish_if_open)
        logging.info(
            "MLflow run %s: experiment=%s (id=%s)",
            self._run_id,
            resolved_experiment_name,
            experiment_id,
        )

    def log_dict(self, values: dict, step: int | None = None, mode: str = "train", **_: Any) -> None:
        if step is None:
            raise ValueError("MLflow metric logging requires a step")
        metrics = {}
        for key, value in values.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                scalar = float(value)
                if math.isfinite(scalar):
                    metrics[f"{mode}/{key}"] = scalar
        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "eval") -> None:
        path = Path(video_path)
        if path.is_file():
            try:
                self._mlflow.log_artifact(str(path), artifact_path=f"{mode}_videos/step_{step}")
            except Exception as error:  # noqa: BLE001
                logging.warning("Could not log video artifact %s to MLflow: %s", path, error)

    def log_policy(self, checkpoint_dir: Path) -> None:
        # LeRobot owns checkpoints; MLflow is used for metrics and optional evaluation videos.
        return None

    def _finish_if_open(self) -> None:
        if self._finished:
            return
        active_run = self._mlflow.active_run()
        if active_run is not None and active_run.info.run_id == self._run_id:
            try:
                self._mlflow.end_run(status="FAILED")
            except Exception as error:  # noqa: BLE001
                logging.warning("Could not mark MLflow run %s as failed: %s", self._run_id, error)
        self._finished = True

    def finish(self) -> None:
        active_run = self._mlflow.active_run()
        if active_run is not None and active_run.info.run_id == self._run_id:
            self._mlflow.end_run(status="FINISHED")
        self._finished = True
