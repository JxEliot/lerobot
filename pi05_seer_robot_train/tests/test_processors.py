import numpy as np
import torch

from pi05_seer_robot_train.seer_pi05.geometry import compact_action, compact_state
from pi05_seer_robot_train.seer_pi05.processors import (
    _absolute_actions,
    _compact_action,
    _compact_state,
    _relative_actions,
)


def test_torch_compaction_matches_dataset_conversion():
    state = np.zeros((2, 64), dtype=np.float32)
    action = np.zeros((2, 16), dtype=np.float32)
    state[:, 54] = state[:, 61] = 1.0
    action[:, 6] = action[:, 13] = 1.0

    torch_state = _compact_state(torch.from_numpy(state))
    torch_action = _compact_action(torch.from_numpy(action))

    assert torch_state.shape == (2, 30)
    assert torch_action.shape == (2, 14)
    np.testing.assert_allclose(torch_state.numpy(), compact_state(state), atol=1e-6)
    np.testing.assert_allclose(torch_action.numpy(), compact_action(action), atol=1e-6)


def test_torch_relative_action_conversion_round_trips():
    state = torch.zeros((2, 30), dtype=torch.float32)
    action = torch.zeros((2, 3, 14), dtype=torch.float32)
    state[:, 20] = 0.4
    state[:, 21] = -0.2
    state[:, 23] = 0.7
    action[..., 0] = torch.tensor([[0.8], [1.0]])
    action[..., 1] = 0.3
    action[..., 3] = 0.2
    action[..., 7] = -0.5
    action[..., 9] = -0.3

    relative = _relative_actions(state, action)
    restored = _absolute_actions(state, relative)

    torch.testing.assert_close(restored, action, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(relative[..., :3], action[..., :3])
