from types import SimpleNamespace

import numpy as np
import pytest

from pi05_seer_robot_train.seer_pi05.data import (
    ACTION_KEY,
    STATE_KEY,
    SeerDatasetView,
    _safe_relative_indices,
    _valid_video_prefix_length,
    compute_training_stats,
)


def test_video_safe_prefix_uses_torchcodec_frame_index_rounding():
    timestamps = [frame / 20 for frame in range(331)]

    valid_length, camera_prefixes = _valid_video_prefix_length(
        timestamps,
        {
            "head": (0.0, 20.0, 328),
            "left_wrist": (0.0, 20.0, 331),
            "right_wrist": (0.0, 20.0, 331),
        },
    )

    assert valid_length == 328
    assert camera_prefixes == {"head": 328, "left_wrist": 331, "right_wrist": 331}


def test_safe_indices_map_episode_prefixes_to_filtered_dataset_rows():
    dataset = SimpleNamespace(
        absolute_to_relative_idx={1: 0, 2: 1, 3: 2, 7: 3, 8: 4, 9: 5},
        meta=SimpleNamespace(
            episodes={
                1: {"dataset_from_index": 1},
                3: {"dataset_from_index": 7},
            }
        ),
    )

    indices = _safe_relative_indices(dataset, [3, 1], {3: 2, 1: 1})

    assert indices == [0, 3, 4]


def test_dataset_view_maps_filtered_rows_and_supports_python_indexing():
    from lerobot.datasets.sampler import EpisodeAwareSampler

    class FakeDataset:
        def __len__(self):
            return 5

        episodes = [0, 1]
        absolute_to_relative_idx = None
        meta = SimpleNamespace(
            episodes={
                "dataset_from_index": [0, 3],
                "dataset_to_index": [3, 5],
            }
        )

        def __getitem__(self, index):
            state = np.zeros(64, dtype=np.float32)
            state[0] = index
            state[54] = state[61] = 1.0
            action = np.zeros(16, dtype=np.float32)
            action[6] = action[13] = 1.0
            return {STATE_KEY: state, ACTION_KEY: action}

        def __getitems__(self, indices):
            return [self[index] for index in indices]

    view = SeerDatasetView(FakeDataset(), [0, 1, 3], {0: 2, 1: 1})

    assert len(view) == 3
    assert view.absolute_to_relative_idx == {0: 0, 1: 1, 3: 2}
    assert list(view.meta.episodes["dataset_to_index"]) == [2, 4]
    assert view[-1][STATE_KEY][0] == 3
    assert [item[STATE_KEY][0] for item in view.__getitems__([2, 0])] == [3, 0]

    sampler = EpisodeAwareSampler(
        view.meta.episodes["dataset_from_index"],
        view.meta.episodes["dataset_to_index"],
        episode_indices_to_use=view.episodes,
        shuffle=False,
        absolute_to_relative_idx=view.absolute_to_relative_idx,
    )
    assert list(sampler) == [0, 1, 2]


def test_training_stats_keep_future_actions_for_valid_video_samples(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    states = np.zeros((4, 64), dtype=np.float32)
    states[:, 54] = states[:, 61] = 1.0
    actions = np.zeros((4, 16), dtype=np.float32)
    actions[:, 0] = [1.0, 3.0, 5.0, 7.0]
    actions[:, 6] = actions[:, 13] = 1.0
    table = pyarrow.table(
        {
            "episode_index": [0] * 4,
            "frame_index": list(range(4)),
            STATE_KEY: pyarrow.array(states.tolist()),
            ACTION_KEY: pyarrow.array(actions.tolist()),
        }
    )
    parquet.write_table(table, tmp_path / "episode.parquet")

    class FakeMeta:
        episodes = {0: {"length": 4}}

        @staticmethod
        def get_data_file_path(episode):
            assert episode == 0
            return "episode.parquet"

    dataset = SimpleNamespace(root=tmp_path, meta=FakeMeta())

    stats = compute_training_stats(dataset, [0], chunk_size=2, valid_lengths={0: 2})

    assert stats[STATE_KEY]["count"] == [2] * 30
    assert stats[ACTION_KEY]["count"] == [4] * 14
    assert stats[ACTION_KEY]["mean"][0] == pytest.approx(3.0)
