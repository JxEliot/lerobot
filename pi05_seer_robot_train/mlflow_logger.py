"""MLflow adapter for the logging hooks used by LeRobot's training loop."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

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


class MLflowLogger:
    """Provide the TrackioLogger interface while storing runs in MLflow."""

    def __init__(self, cfg: TrainPipelineConfig):
        import mlflow

        self._mlflow = mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI must be set when MLflow logging is enabled")

        tracking_home = os.environ.get("MLFLOW_HOME")
        if tracking_home:
            Path(tracking_home).expanduser().mkdir(parents=True, exist_ok=True)
        artifact_root = Path(os.environ["MLFLOW_ARTIFACT_ROOT"]).expanduser()
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)

        logger_cfg = getattr(cfg, "trackio", None) or getattr(cfg, "wandb", None)
        project = getattr(logger_cfg, "project", None)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", project or "seer-pi05")
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                experiment_name,
                artifact_location=artifact_root.resolve().as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        run_name = getattr(logger_cfg, "run_name", None) or getattr(logger_cfg, "run_id", None) or cfg.job_name
        tags = {
            "job_name": cfg.job_name,
            "dataset.repo_id": cfg.dataset.repo_id if cfg.dataset else "",
            "dataset.root": str(cfg.dataset.root) if cfg.dataset else "",
            "training_profile": os.environ.get("TRAINING_PROFILE", "default"),
        }
        run = mlflow.start_run(run_name=run_name, tags=tags)
        self._run_id = run.info.run_id

        params = {
            key: value
            for key, value in _flatten_params(cfg.to_dict()).items()
            if key not in {"output_dir", "checkpoint_path"} and not key.endswith(".pretrained_path")
        }
        items = list(params.items())
        for offset in range(0, len(items), 100):
            mlflow.log_params(dict(items[offset : offset + 100]))

        logging.info(
            "MLflow run %s: experiment=%s tracking_uri=%s artifact_root=%s",
            self._run_id,
            experiment_name,
            tracking_uri,
            artifact_root,
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
        if not path.is_file():
            logging.warning("Skipping missing MLflow video artifact: %s", path)
            return
        try:
            self._mlflow.log_artifact(str(path), artifact_path=f"{mode}_videos/step_{step}")
        except Exception as error:  # noqa: BLE001
            logging.warning("Could not log video artifact %s to MLflow: %s", path, error)

    def log_policy(self, checkpoint_dir: Path) -> None:
        # LeRobot retains ownership of all checkpoints; MLflow is metrics-only.
        pass

    def finish(self) -> None:
        active_run = self._mlflow.active_run()
        if active_run is not None and active_run.info.run_id == self._run_id:
            self._mlflow.end_run(status="FINISHED")
