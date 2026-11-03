import numpy as np

from pi05_seer_robot_train.seer_pi05.geometry import (
    absolute_eef_actions,
    compact_action,
    compact_state,
    relative_action_windows,
    relative_eef_actions,
)


def _raw_vectors(length: int = 3):
    state = np.zeros((length, 64), dtype=np.float32)
    action = np.zeros((length, 16), dtype=np.float32)
    state[:, 54] = 1.0
    state[:, 61] = 1.0
    action[:, 6] = 1.0
    action[:, 13] = 1.0
    state[:, 48] = np.arange(length, dtype=np.float32)
    state[:, 55] = np.arange(length, dtype=np.float32) * -1
    action[:, 0] = np.arange(length, dtype=np.float32) + 5
    action[:, 7] = np.arange(length, dtype=np.float32) - 3
    return state, action


def test_compacts_v3_seer_vectors_to_pi05_dimensions():
    state, action = _raw_vectors()

    compact_states = compact_state(state)
    compact_actions = compact_action(action)

    assert compact_states.shape == (3, 30)
    assert compact_actions.shape == (3, 14)
    np.testing.assert_allclose(compact_states[:, 18], state[:, 48])
    np.testing.assert_allclose(compact_actions[:, 12:], action[:, 14:])


def test_relative_and_absolute_eef_transforms_round_trip():
    state, action = _raw_vectors(2)
    state[:, 51:54] = np.array([0.0, 0.0, 0.7], dtype=np.float32)
    action[:, 3:6] = np.array([0.1, -0.2, 1.0], dtype=np.float32)
    compact_states = compact_state(state)
    compact_actions = compact_action(action)

    relative = relative_eef_actions(compact_states, compact_actions)
    restored = absolute_eef_actions(compact_states, relative)

    np.testing.assert_allclose(restored, compact_actions, atol=1e-5)
    assert not np.allclose(relative[:, :3], compact_actions[:, :3])


def test_action_windows_repeat_at_episode_end_without_crossing_boundary():
    state, action = _raw_vectors()
    windows = relative_action_windows(state, action, chunk_size=2)

    assert windows.shape == (3, 2, 14)
    np.testing.assert_allclose(windows[0, :, 0], [5.0, 6.0])
    np.testing.assert_allclose(windows[1, :, 0], [5.0, 6.0])
    np.testing.assert_allclose(windows[2, :, 0], [5.0, 5.0])
