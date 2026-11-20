import sys
from types import SimpleNamespace

import pytest

from lerobot.common.mlflow_utils import MLflowLogger


def test_mlflow_logger_writes_metrics_to_configured_server(monkeypatch):
    calls = {"metrics": [], "params": [], "statuses": []}
    experiment = SimpleNamespace(experiment_id="17", name="seer-pi05-test")
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
    active_run = run

    class FakeClient:
        def get_experiment_by_name(self, name):
            assert name == experiment.name
            return experiment

    def start_run(**kwargs):
        assert kwargs["experiment_id"] == experiment.experiment_id
        assert kwargs["run_name"].startswith("test-job.")
        assert kwargs["tags"]["team"] == "robotics"
        return run

    def end_run(status):
        calls["statuses"].append(status)

    mlflow = SimpleNamespace(
        MlflowClient=FakeClient,
        set_tracking_uri=lambda uri: calls.setdefault("tracking_uri", uri),
        start_run=start_run,
        log_params=lambda params: calls["params"].append(params),
        log_metrics=lambda metrics, step: calls["metrics"].append((metrics, step)),
        active_run=lambda: active_run,
        end_run=end_run,
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment.name)
    monkeypatch.setenv("MLFLOW_RUN_NAME_PREFIX", "test")
    monkeypatch.setenv("MLFLOW_TAGS", "team=robotics")

    cfg = SimpleNamespace(
        job_name="test-job",
        dataset=SimpleNamespace(repo_id="seer/test", root="/datasets/test"),
        to_dict=lambda: {"batch_size": 8, "trackio": {"enable": False}},
    )
    logger = MLflowLogger(cfg)
    logger.log_dict({"loss": 1.25, "invalid": float("nan"), "enabled": True}, step=12)
    logger.finish()

    assert calls["tracking_uri"] == "https://mlflow.example.test"
    assert calls["metrics"] == [({"train/loss": 1.25}, 12)]
    assert all("trackio.enable" not in params for params in calls["params"])
    assert calls["statuses"] == ["FINISHED"]


def test_mlflow_logger_requires_remote_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace())
    cfg = SimpleNamespace(job_name="test-job", dataset=None, to_dict=lambda: {})

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        MLflowLogger(cfg)
