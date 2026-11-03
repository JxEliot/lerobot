import numpy as np
import pytest

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
    np.testing.assert_allclose(compact_states[:, 16:18], state[:, 62:64])
    np.testing.assert_allclose(compact_actions[:, 12:], action[:, 14:])


def test_quaternion_compaction_matches_xyzw_reference_for_edge_rotations():
    rotation_cls = pytest.importorskip("scipy.spatial.transform").Rotation

    rotvecs = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.pi - 1e-6, 0.0, 0.0],
            [0.0, -(np.pi - 1e-5), 0.0],
            [0.0, 0.0, 1.7],
            [0.31, -0.82, 0.47],
        ],
        dtype=np.float32,
    )
    quaternions = rotation_cls.from_rotvec(rotvecs).as_quat().astype(np.float32)
    quaternions[1::2] *= -1.0

    state = np.zeros((len(rotvecs), 64), dtype=np.float32)
    action = np.zeros((len(rotvecs), 16), dtype=np.float32)
    state[:, 54] = state[:, 61] = 1.0
    action[:, 6] = action[:, 13] = 1.0
    state[:, 51:55] = quaternions
    state[:, 58:62] = -quaternions
    action[:, 3:7] = -quaternions
    action[:, 10:14] = quaternions

    compact_states = compact_state(state)
    compact_actions = compact_action(action)
    expected = rotation_cls.from_quat(quaternions)
    expected_inverse = rotation_cls.from_quat(-quaternions)

    for actual in (compact_states[:, 21:24], compact_actions[:, 3:6]):
        np.testing.assert_allclose(
            rotation_cls.from_rotvec(actual).as_matrix(), expected.as_matrix(), atol=2e-5
        )
    for actual in (compact_states[:, 27:30], compact_actions[:, 9:12]):
        np.testing.assert_allclose(
            rotation_cls.from_rotvec(actual).as_matrix(), expected_inverse.as_matrix(), atol=2e-5
        )


def test_exact_pi_quaternions_have_a_deterministic_axis_sign():
    state = np.zeros((2, 64), dtype=np.float32)
    action = np.zeros((2, 16), dtype=np.float32)
    state[:, 54] = state[:, 61] = 1.0
    action[:, 6] = action[:, 13] = 1.0
    quaternion = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)
    state[0, 51:55], state[1, 51:55] = quaternion, -quaternion
    action[0, 3:7], action[1, 3:7] = quaternion, -quaternion

    compact_states = compact_state(state)
    compact_actions = compact_action(action)

    np.testing.assert_allclose(compact_states[0, 21:24], compact_states[1, 21:24])
    np.testing.assert_allclose(compact_actions[0, 3:6], compact_actions[1, 3:6])


def test_dual_arm_relative_pose_matches_independent_se3_reference():
    rotation_cls = pytest.importorskip("scipy.spatial.transform").Rotation

    state = np.zeros(64, dtype=np.float32)
    action = np.zeros((2, 16), dtype=np.float32)
    left_current = rotation_cls.from_euler("z", 90, degrees=True)
    right_current = rotation_cls.from_euler("x", -60, degrees=True)
    left_target = rotation_cls.from_rotvec([0.4, -0.2, 0.7])
    right_target = rotation_cls.from_rotvec([-0.3, 0.8, 0.1])
    state[48:51] = [1.0, 2.0, 3.0]
    state[51:55] = left_current.as_quat()
    state[55:58] = [-1.0, 0.5, 2.0]
    state[58:62] = right_current.as_quat()
    state[62:64] = [0.2, 0.8]
    action[:, :3] = [[1.0, 3.0, 3.5], [0.5, 1.0, 2.0]]
    action[0, 3:7] = left_target.as_quat()
    action[1, 7:14] = [0.3, 0.5, -0.2, *right_target.as_quat()]
    action[:, 14:16] = [[0.9, 0.1], [0.4, 0.6]]

    compact_state_vector = compact_state(state)
    compact_action_vector = compact_action(action)
    relative = relative_eef_actions(compact_state_vector, compact_action_vector)

    expected_positions = [
        left_current.inv().apply(action[0, :3] - state[48:51]),
        right_current.inv().apply(action[1, 7:10] - state[55:58]),
    ]
    expected_rotations = [left_current.inv() * left_target, right_current.inv() * right_target]
    np.testing.assert_allclose(relative[0, :3], expected_positions[0], atol=2e-5)
    np.testing.assert_allclose(relative[1, :3], expected_positions[1], atol=2e-5)
    for actual, expected in zip((relative[0, 3:6], relative[1, 9:12]), expected_rotations, strict=True):
        np.testing.assert_allclose(
            rotation_cls.from_rotvec(actual).as_matrix(), expected.as_matrix(), atol=2e-5
        )
    np.testing.assert_allclose(relative[:, 12:14], compact_action_vector[:, 12:14])

    restored = absolute_eef_actions(compact_state_vector, relative)
    np.testing.assert_allclose(restored[:, :3], compact_action_vector[:, :3], atol=2e-5)
    np.testing.assert_allclose(restored[:, 6:9], compact_action_vector[:, 6:9], atol=2e-5)
    np.testing.assert_allclose(restored[:, 12:14], compact_action_vector[:, 12:14], atol=1e-6)


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
