"""LeRobot v3 dataset adaptation and training-only statistics."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def _valid_video_prefix_length(
    timestamps: list[float], video_specs: dict[str, tuple[float, float, int]]
) -> tuple[int, dict[str, int]]:
    """Return the usable row prefix and the first invalid row for each camera.

    ``video_specs`` maps each camera to its source timestamp offset, average FPS,
    and decoded frame count. This mirrors the frame-index calculation in
    ``decode_video_frames_torchcodec``.
    """
    prefixes: dict[str, int] = {}
    for camera, (offset, fps, num_frames) in video_specs.items():
        if not np.isfinite(offset) or not np.isfinite(fps) or fps <= 0 or num_frames <= 0:
            raise ValueError(f"Camera {camera} has invalid video metadata: fps={fps}, frames={num_frames}")
        prefixes[camera] = next(
            (
                row
                for row, timestamp in enumerate(timestamps)
                if not 0 <= round((offset + timestamp) * fps) < num_frames
            ),
            len(timestamps),
        )
    return min(prefixes.values(), default=len(timestamps)), prefixes


def _inspect_video_safe_rows(
    dataset: Any, episodes: list[int]
) -> tuple[dict[int, int], dict[int, list[str]]]:
    """Find episode tails whose camera frames are absent from the MP4 files."""
    try:
        import pyarrow.parquet as pq
        from lerobot.datasets.video_utils import VideoDecoderCache
    except ImportError as error:
        message = "The LeRobot dataset extra (including pyarrow and torchcodec) is required"
        raise RuntimeError(message) from error

    episodes_by_file: dict[Path, list[int]] = {}
    for episode in episodes:
        path = Path(dataset.root) / dataset.meta.get_data_file_path(int(episode))
        episodes_by_file.setdefault(path, []).append(int(episode))

    timestamps_by_episode: dict[int, list[float]] = {}
    for path, file_episodes in episodes_by_file.items():
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp"])
        episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        for episode in file_episodes:
            rows = np.flatnonzero(episode_indices == episode)
            rows = rows[np.argsort(frame_indices[rows], kind="stable")]
            if not np.array_equal(frame_indices[rows], np.arange(len(rows), dtype=np.int64)):
                raise ValueError(f"Episode {episode} has non-contiguous frame_index values in {path}")
            timestamps_by_episode[episode] = timestamps[rows].tolist()

    valid_lengths: dict[int, int] = {}
    truncation_causes: dict[int, list[str]] = {}
    decoder_cache = VideoDecoderCache(max_size=4)
    try:
        for episode in episodes:
            ep = dataset.meta.episodes[int(episode)]
            timestamps = timestamps_by_episode[int(episode)]
            if len(timestamps) != int(ep["length"]):
                raise ValueError(
                    f"Episode {episode} has {len(timestamps)} parquet timestamps but "
                    f"metadata length is {ep['length']}"
                )

            video_specs = {}
            for camera in dataset.meta.video_keys:
                video_prefix = f"videos/{camera}"
                video_path = dataset.root / dataset.meta.get_video_file_path(int(episode), camera)
                try:
                    decoder = decoder_cache.get_decoder(str(video_path))
                except Exception as error:
                    message = f"Could not inspect episode {episode} camera {camera}: {video_path}"
                    raise RuntimeError(message) from error
                metadata = decoder.metadata
                video_specs[camera] = (
                    float(ep[f"{video_prefix}/from_timestamp"]),
                    float(metadata.average_fps),
                    int(metadata.num_frames),
                )

            valid_length, camera_prefixes = _valid_video_prefix_length(timestamps, video_specs)
            if valid_length <= 0:
                raise ValueError(f"Episode {episode} has no rows with valid video frames")
            valid_lengths[int(episode)] = valid_length
            if valid_length < len(timestamps):
                truncation_causes[int(episode)] = [
                    camera for camera, prefix in camera_prefixes.items() if prefix == valid_length
                ]
    finally:
        decoder_cache.clear()

    return valid_lengths, truncation_causes


def _safe_relative_indices(dataset: Any, episodes: list[int], valid_lengths: dict[int, int]) -> list[int]:
    """Map valid absolute row prefixes into the selected dataset's index space."""
    absolute_to_relative = dataset.absolute_to_relative_idx
    indices = []
    for episode in episodes:
        ep = dataset.meta.episodes[int(episode)]
        absolute_start = int(ep["dataset_from_index"])
        for offset in range(valid_lengths[int(episode)]):
            absolute_index = absolute_start + offset
            if absolute_to_relative is None:
                indices.append(absolute_index)
            else:
                try:
                    indices.append(absolute_to_relative[absolute_index])
                except KeyError as error:
                    raise ValueError(
                        f"Selected dataset is missing absolute row {absolute_index} from episode {episode}"
                    ) from error
    return sorted(indices)


def _log_video_tail_drops(
    label: str, dataset: Any, episodes: list[int], valid_lengths: dict[int, int], causes: dict[int, list[str]]
) -> None:
    if not causes:
        return
    details = []
    dropped_total = 0
    for episode in episodes:
        if episode not in causes:
            continue
        length = int(valid_lengths[episode])
        full_length = int(dataset.meta.episodes[int(episode)]["length"])
        dropped = full_length - length
        dropped_total += dropped
        details.append(f"{episode}:{dropped}[{','.join(causes[episode])}]")
    logger.warning(
        "%s: skipped %d terminal sample(s) from %d episode(s) because camera video frames are missing "
        "(episode:rows[cameras]): %s",
        label,
        dropped_total,
        len(causes),
        ", ".join(details),
    )


def compute_training_stats(
    dataset: Any, episodes: list[int], chunk_size: int, valid_lengths: dict[int, int] | None = None
) -> dict[str, dict]:
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
        table = pq.read_table(path, columns=["episode_index", "frame_index", STATE_KEY, ACTION_KEY])
        file_episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        file_frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        file_states = table[STATE_KEY].to_pylist()
        file_actions = table[ACTION_KEY].to_pylist()
        for episode in file_episodes:
            indices = np.flatnonzero(file_episode_indices == episode)
            if not len(indices):
                raise ValueError(f"No parquet rows found for episode {episode} in {path}")
            indices = indices[np.argsort(file_frame_indices[indices], kind="stable")]
            if not np.array_equal(file_frame_indices[indices], np.arange(len(indices), dtype=np.int64)):
                raise ValueError(f"Episode {episode} has non-contiguous frame_index values in {path}")
            state = compact_state(np.asarray([file_states[index] for index in indices], dtype=np.float32))
            action = compact_action(np.asarray([file_actions[index] for index in indices], dtype=np.float32))
            valid_length = valid_lengths[episode] if valid_lengths is not None else len(indices)
            if not 0 < valid_length <= len(indices):
                raise ValueError(
                    f"Episode {episode} has invalid video-safe length {valid_length}/{len(indices)}"
                )
            state_rows.append(state[:valid_length])
            windows = relative_action_windows(state, action, chunk_size)
            action_windows.append(windows[:valid_length].reshape(-1, 14))

    if not state_rows:
        raise ValueError("No training episodes were selected for Seer PI05 statistics")
    return {
        STATE_KEY: vector_stats(np.concatenate(state_rows, axis=0)),
        ACTION_KEY: vector_stats(np.concatenate(action_windows, axis=0)),
    }


class SeerDatasetView(Dataset):
    """Apply the 64/16 to 30/14 schema conversion while preserving LeRobot behavior."""

    def __init__(
        self,
        dataset: Any,
        valid_indices: list[int] | None = None,
        valid_lengths: dict[int, int] | None = None,
    ):
        self._dataset = dataset
        self._valid_indices = list(range(len(dataset))) if valid_indices is None else valid_indices
        self.meta = _SafeDatasetMetadata(dataset.meta, valid_lengths)

        source_mapping = dataset.absolute_to_relative_idx
        relative_to_absolute = (
            {relative: absolute for absolute, relative in source_mapping.items()}
            if source_mapping is not None
            else None
        )
        self.absolute_to_relative_idx = {
            (relative_to_absolute[index] if relative_to_absolute is not None else index): view_index
            for view_index, index in enumerate(self._valid_indices)
        }

    def __getattr__(self, name: str) -> Any:
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._valid_indices)

    @property
    def hf_dataset(self) -> Any:
        """Expose the filtered Arrow view for training helpers that inspect columns."""
        return self._dataset.hf_dataset.select(self._valid_indices)

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
        if isinstance(index, slice):
            return [self[item_index] for item_index in range(*index.indices(len(self)))]
        return self._convert(self._dataset[self._valid_indices[index]])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        getitems = getattr(self._dataset, "__getitems__", None)
        mapped_indices = [self._valid_indices[index] for index in indices]
        items = (
            getitems(mapped_indices)
            if getitems is not None
            else [self._dataset[index] for index in mapped_indices]
        )
        return [self._convert(item) for item in items]


class _SafeEpisodeTable:
    def __init__(self, episodes: Any, valid_lengths: dict[int, int] | None):
        self._episodes = episodes
        self._safe_to_indices = None
        if valid_lengths is not None:
            starts = np.asarray(episodes["dataset_from_index"], dtype=np.int64)
            ends = np.asarray(episodes["dataset_to_index"], dtype=np.int64)
            self._safe_to_indices = ends.copy()
            for episode, length in valid_lengths.items():
                if not 0 <= episode < len(starts) or not 0 < length <= ends[episode] - starts[episode]:
                    raise ValueError(f"Episode {episode} has invalid video-safe length {length}")
                self._safe_to_indices[episode] = starts[episode] + length

    def __getitem__(self, key: Any) -> Any:
        if key == "dataset_to_index" and self._safe_to_indices is not None:
            return self._safe_to_indices
        return self._episodes[key]

    def __getattr__(self, name: str) -> Any:
        if name == "_episodes":
            raise AttributeError(name)
        return getattr(self._episodes, name)


class _SafeDatasetMetadata:
    def __init__(self, metadata: Any, valid_lengths: dict[int, int] | None):
        self._metadata = metadata
        self.episodes = _SafeEpisodeTable(metadata.episodes, valid_lengths)

    def __getattr__(self, name: str) -> Any:
        if name in {"_metadata", "episodes"}:
            raise AttributeError(name)
        return getattr(self._metadata, name)


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
    train_valid_lengths, train_truncations = _inspect_video_safe_rows(train_dataset, episodes)
    train_valid_indices = _safe_relative_indices(train_dataset, episodes, train_valid_lengths)
    _log_video_tail_drops("Train data", train_dataset, episodes, train_valid_lengths, train_truncations)

    eval_valid_lengths = None
    eval_valid_indices = None
    if eval_dataset is not None:
        eval_episodes = eval_dataset.episodes
        if eval_episodes is None:
            eval_episodes = list(range(eval_dataset.meta.total_episodes))
        eval_valid_lengths, eval_truncations = _inspect_video_safe_rows(eval_dataset, eval_episodes)
        eval_valid_indices = _safe_relative_indices(eval_dataset, eval_episodes, eval_valid_lengths)
        _log_video_tail_drops("Eval data", eval_dataset, eval_episodes, eval_valid_lengths, eval_truncations)

    stats = compute_training_stats(
        train_dataset,
        [int(episode) for episode in episodes],
        cfg.trainable_config.chunk_size,
        train_valid_lengths,
    )
    for dataset in (train_dataset, eval_dataset):
        if dataset is not None:
            _set_compact_metadata(dataset, stats)

    return (
        SeerDatasetView(train_dataset, train_valid_indices, train_valid_lengths),
        SeerDatasetView(eval_dataset, eval_valid_indices, eval_valid_lengths)
        if eval_dataset is not None
        else None,
    )
