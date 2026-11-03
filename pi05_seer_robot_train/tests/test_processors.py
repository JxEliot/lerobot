import numpy as np
import torch

from pi05_seer_robot_train.seer_pi05.geometry import compact_action, compact_state
from pi05_seer_robot_train.seer_pi05.processors import (
    SeerEefAbsoluteActionsProcessorStep,
    SeerEefRelativeActionsProcessorStep,
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


def test_torch_quaternion_compaction_is_sign_stable_at_pi():
    state = torch.zeros((2, 64), dtype=torch.float32)
    action = torch.zeros((2, 16), dtype=torch.float32)
    state[:, 54] = state[:, 61] = 1.0
    action[:, 6] = action[:, 13] = 1.0
    quaternion = torch.tensor([0.0, -1.0, 0.0, 0.0])
    state[0, 51:55], state[1, 51:55] = quaternion, -quaternion
    action[0, 3:7], action[1, 3:7] = quaternion, -quaternion

    compact_states = _compact_state(state)
    compact_actions = _compact_action(action)

    torch.testing.assert_close(compact_states[0, 21:24], compact_states[1, 21:24])
    torch.testing.assert_close(compact_actions[0, 3:6], compact_actions[1, 3:6])


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


def test_inference_keeps_chunk_anchor_until_queued_actions_are_drained():
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor import bind_relative_anchor
    from lerobot.utils.constants import OBS_STATE

    class FakePolicy:
        queued_actions = 0

        def count_queued_actions(self):
            return self.queued_actions

    state_at_prediction = torch.zeros((1, 30), dtype=torch.float32)
    state_at_prediction[:, 23] = torch.pi / 2
    state_after_first_action = state_at_prediction.clone()
    state_after_first_action[:, 18] = 3.0
    state_after_first_action[:, 23] = -0.4

    absolute_target = torch.zeros((1, 14), dtype=torch.float32)
    absolute_target[:, 0:3] = torch.tensor([[0.5, 1.0, -0.2]])
    absolute_target[:, 3:6] = torch.tensor([[0.2, -0.1, 0.6]])
    absolute_target[:, 6:12] = torch.tensor([[2.0, 0.5, 0.1, -0.3, 0.4, 0.2]])
    absolute_target[:, 12:14] = torch.tensor([[0.25, 0.75]])

    relative_step = SeerEefRelativeActionsProcessorStep()
    absolute_step = SeerEefAbsoluteActionsProcessorStep(relative_step=relative_step)
    policy = FakePolicy()
    pipeline = type("Pipeline", (), {"steps": [relative_step]})()
    assert bind_relative_anchor(policy, pipeline) is relative_step

    relative_step({TransitionKey.OBSERVATION: {OBS_STATE: state_at_prediction}})
    relative_target = _relative_actions(state_at_prediction, absolute_target)

    policy.queued_actions = 2
    relative_step({TransitionKey.OBSERVATION: {OBS_STATE: state_after_first_action}})
    restored_during_chunk = absolute_step({TransitionKey.ACTION: relative_target})[TransitionKey.ACTION]
    torch.testing.assert_close(restored_during_chunk, absolute_target, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(relative_step.get_cached_state(), state_at_prediction)

    policy.queued_actions = 0
    relative_step({TransitionKey.OBSERVATION: {OBS_STATE: state_after_first_action}})
    torch.testing.assert_close(relative_step.get_cached_state(), state_after_first_action)
