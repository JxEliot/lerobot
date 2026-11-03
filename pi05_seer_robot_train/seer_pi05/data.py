"""LeRobot v3 dataset adaptation and training-only statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import compact_action, compact_state, relative_action_windows, vector_stats

STATE_KEY = "observation.state"
ACTION_KEY = "action"
# The policy state order matches the legacy Seer end-effector training branch:
# joint positions, base positions, grippers, then left and right EEF poses.
STATE_NAMES = [
    *(f"left_arm_joint_{index}.position" for index in range(1, 8)),
    *(f"right_arm_joint_{index}.position" for index in range(1, 8)),
    "waist_position.position",
    "lift_position.position",
    "left_gripper",
    "right_gripper",
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.rx",
    "left_ee.ry",
    "left_ee.rz",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.rx",
    "right_ee.ry",
    "right_ee.rz",
]
ACTION_NAMES = [
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.rx",
    "left_ee.ry",
    "left_ee.rz",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.rx",
    "right_ee.ry",
    "right_ee.rz",
    "left_gripper",
    "right_gripper",
]


def compute_training_stats(dataset: Any, episodes: list[int], chunk_size: int) -> dict[str, dict]:
    """Read selected training episode parquet files without decoding or modifying videos."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("The LeRobot dataset extra (including pyarrow) is required") from error

    episodes_by_file: dict[Path, list[int]] = {}
    for episode in episodes:
        path = Path(dataset.root) / dataset.meta.get_data_file_path(int(episode))
        episodes_by_file.setdefault(path, []).append(int(episode))

    state_rows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    for path, file_episodes in episodes_by_file.items():
        table = pq.read_table(path, columns=["episode_index", STATE_KEY, ACTION_KEY])
        file_episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        file_states = table[STATE_KEY].to_pylist()
        file_actions = table[ACTION_KEY].to_pylist()
        for episode in file_episodes:
            indices = np.flatnonzero(file_episode_indices == episode)
            if not len(indices):
                raise ValueError(f"No parquet rows found for episode {episode} in {path}")
            state = compact_state(np.asarray([file_states[index] for index in indices], dtype=np.float32))
            action = compact_action(np.asarray([file_actions[index] for index in indices], dtype=np.float32))
            state_rows.append(state)
            action_windows.append(relative_action_windows(state, action, chunk_size).reshape(-1, 14))

    if not state_rows:
        raise ValueError("No training episodes were selected for Seer PI05 statistics")
    return {
        STATE_KEY: vector_stats(np.concatenate(state_rows, axis=0)),
        ACTION_KEY: vector_stats(np.concatenate(action_windows, axis=0)),
    }


class SeerDatasetView(Dataset):
    """Apply the 64/16 to 30/14 schema conversion while preserving LeRobot behavior."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __getattr__(self, name: str) -> Any:
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._dataset)

    def _convert(self, item: dict[str, Any]) -> dict[str, Any]:
        converted = dict(item)
        for key, converter in ((STATE_KEY, compact_state), (ACTION_KEY, compact_action)):
            value = converted.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                array = converter(value.detach().cpu().numpy())
                converted[key] = torch.as_tensor(array, dtype=value.dtype)
            else:
                converted[key] = converter(np.asarray(value))
        return converted

    def __getitem__(self, index: int | slice) -> Any:
        result = self._dataset[index]
        if isinstance(result, list):
            return [self._convert(item) for item in result]
        return self._convert(result)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        getitems = getattr(self._dataset, "__getitems__", None)
        items = getitems(indices) if getitems is not None else [self._dataset[index] for index in indices]
        return [self._convert(item) for item in items]


def _set_compact_metadata(dataset: Any, stats: dict[str, dict]) -> None:
    state_feature = dataset.meta.features[STATE_KEY]
    action_feature = dataset.meta.features[ACTION_KEY]
    state_feature["shape"] = [30]
    state_feature["names"] = list(STATE_NAMES)
    action_feature["shape"] = [14]
    action_feature["names"] = list(ACTION_NAMES)
    dataset.meta.stats.update(stats)


def make_seer_train_eval_datasets(cfg: Any):
    """Build standard v3 datasets, then expose compact EEF vectors and matching stats."""
    from lerobot.datasets.factory import make_train_eval_datasets

    if cfg.dataset.streaming:
        raise ValueError("Seer PI05 adaptation currently requires map-style LeRobot v3 datasets")

    train_dataset, eval_dataset = make_train_eval_datasets(cfg)
    info = train_dataset.meta.info
    if info.codebase_version != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 dataset, got {info.codebase_version!r}")
    state_shape = tuple(train_dataset.meta.features[STATE_KEY]["shape"])
    action_shape = tuple(train_dataset.meta.features[ACTION_KEY]["shape"])
    if state_shape != (64,) or action_shape != (16,):
        raise ValueError(f"Expected Seer state/action shapes 64/16, got {state_shape}/{action_shape}")

    episodes = train_dataset.episodes
    if episodes is None:
        episodes = list(range(train_dataset.meta.total_episodes))
    stats = compute_training_stats(
        train_dataset,
        [int(episode) for episode in episodes],
        cfg.trainable_config.chunk_size,
    )
    for dataset in (train_dataset, eval_dataset):
        if dataset is not None:
            _set_compact_metadata(dataset, stats)

    return SeerDatasetView(train_dataset), SeerDatasetView(eval_dataset) if eval_dataset is not None else None
