"""Read-only statistics for the exact Seer PI05 training action windows."""
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def compact_seer_vectors(state, action):
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    def pose(x):
        return np.concatenate((x[..., :3], Rotation.from_quat(x[..., 3:7]).as_rotvec()), axis=-1)
    if state.shape[-1] == 64 and action.shape[-1] == 16:
        state = np.concatenate((state[:, :16], state[:, 62:64], pose(state[:, 48:55]), pose(state[:, 55:62])), axis=-1)
        action = np.concatenate((pose(action[:, :7]), pose(action[:, 7:14]), action[:, 14:16]), axis=-1)
    elif state.shape[-1] != 30 or action.shape[-1] != 14:
        raise ValueError(f"Expected Seer 64/16 or compact 30/14 vectors, got {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite Seer training vectors")
    return state.astype(np.float32), action.astype(np.float32)


def relative_action_windows(state, action, chunk_size):
    """One anchor per row; repeat the last action at the episode boundary."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n = len(state)
    if not n or len(action) != n:
        raise ValueError("Episode requires matching, non-empty states and actions")
    indices = np.minimum(np.arange(n)[:, None] + np.arange(chunk_size), n - 1)
    windows = action[indices].copy()
    for a0, s0 in ((0, 18), (6, 24)):
        current = Rotation.from_rotvec(np.repeat(state[:, s0 + 3:s0 + 6], chunk_size, axis=0))
        target = Rotation.from_rotvec(windows[..., a0 + 3:a0 + 6].reshape(-1, 3))
        windows[..., a0:a0 + 3] = current.inv().apply(
            (windows[..., a0:a0 + 3] - state[:, None, s0:s0 + 3]).reshape(-1, 3)
        ).reshape(n, chunk_size, 3)
        windows[..., a0 + 3:a0 + 6] = (current.inv() * target).as_rotvec().reshape(n, chunk_size, 3)
    return windows


def vector_stats(x):
    return {key: np.asarray(value, dtype=np.float32).tolist() for key, value in {
        "min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0),
        "q01": np.quantile(x, .01, axis=0), "q99": np.quantile(x, .99, axis=0),
    }.items()} | {"count": [len(x)] * x.shape[1]}


def compute_seer_pi05_stats(paths, chunk_size=50):
    states, windows = [], []
    for path in paths:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state, action = compact_seer_vectors(table["observation.state"].to_pylist(), table["action"].to_pylist())
        states.append(state)
        windows.append(relative_action_windows(state, action, chunk_size).reshape(-1, 14))
    if not states:
        raise ValueError("No training episodes selected for Seer PI05 statistics")
    return {"observation.state": vector_stats(np.concatenate(states)), "action": vector_stats(np.concatenate(windows))}


def apply_seer_pi05_training_stats(dataset, policy_config):
    """Only fresh PI05 training opts in; existing checkpoint statistics stay authoritative."""
    if not getattr(policy_config, "use_gr00t_eef_relative_actions", False):
        return dataset
    if dataset.meta.info.codebase_version != "v2.1":
        return dataset
    episodes = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
    paths = [dataset.root / dataset.meta.get_data_file_path(int(ep)) for ep in episodes]
    dataset.meta.stats.update(compute_seer_pi05_stats(paths, int(policy_config.chunk_size)))
    return dataset
