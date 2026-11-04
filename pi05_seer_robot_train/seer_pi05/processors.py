"""PI05 processors for compact Seer state and dual-arm relative EEF actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _normalize_quaternion(value: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8) or not torch.isfinite(value).all():
        raise ValueError("Seer end-effector quaternions must be finite and non-zero")
    return value / norm


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lv, lw = left[..., :3], left[..., 3:4]
    rv, rw = right[..., :3], right[..., 3:4]
    return torch.cat(
        (lw * rv + rw * lv + torch.linalg.cross(lv, rv, dim=-1), lw * rw - (lv * rv).sum(-1, keepdim=True)),
        dim=-1,
    )


def _rotate_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qv = quaternion[..., :3]
    first = torch.linalg.cross(qv, vector, dim=-1)
    return vector + 2.0 * (quaternion[..., 3:4] * first + torch.linalg.cross(qv, first, dim=-1))


def _rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half_angle = angle * 0.5
    scale = torch.where(angle > 1e-8, torch.sin(half_angle) / angle.clamp_min(1e-8), 0.5 - angle.square() / 48)
    return _normalize_quaternion(torch.cat((rotvec * scale, torch.cos(half_angle)), dim=-1))


def _quaternion_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _normalize_quaternion(quaternion)
    vector = quaternion[..., :3]
    first_nonzero = torch.where(
        vector[..., 0:1] != 0,
        vector[..., 0:1],
        torch.where(vector[..., 1:2] != 0, vector[..., 1:2], vector[..., 2:3]),
    )
    flip = (quaternion[..., 3:4] < 0) | ((quaternion[..., 3:4] == 0) & (first_nonzero < 0))
    quaternion = torch.where(flip, -quaternion, quaternion)
    vector = quaternion[..., :3]
    sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quaternion[..., 3:4].clamp(0.0, 1.0))
    scale = torch.where(sin_half > 1e-8, angle / sin_half.clamp_min(1e-8), 2.0)
    return vector * scale


def _compact_state(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] == 30:
        return state
    if state.shape[-1] != 64:
        raise ValueError(f"Expected a 64D or compact 30D state, got {tuple(state.shape)}")
    dtype = state.dtype
    state = state.float()

    def pose(start: int) -> torch.Tensor:
        return torch.cat((state[..., start : start + 3], _quaternion_to_rotvec(state[..., start + 3 : start + 7])), dim=-1)

    return torch.cat((state[..., :16], state[..., 62:64], pose(48), pose(55)), dim=-1).to(dtype)


def _compact_action(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] == 14:
        return action
    if action.shape[-1] != 16:
        raise ValueError(f"Expected a 16D or compact 14D action, got {tuple(action.shape)}")
    dtype = action.dtype
    action = action.float()

    def pose(start: int) -> torch.Tensor:
        return torch.cat((action[..., start : start + 3], _quaternion_to_rotvec(action[..., start + 3 : start + 7])), dim=-1)

    return torch.cat((pose(0), pose(7), action[..., 14:16]), dim=-1).to(dtype)


def _relative_actions(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    output_dtype = action.dtype
    state, action = state.float(), action.float()
    converted = action.clone()
    for action_start, state_start in ((0, 18), (6, 24)):
        current = state[..., state_start : state_start + 6]
        target = action[..., action_start : action_start + 6]
        while current.ndim < target.ndim:
            current = current.unsqueeze(-2)
        current_quaternion = _rotvec_to_quaternion(current[..., 3:6])
        target_quaternion = _rotvec_to_quaternion(target[..., 3:6])
        inverse_current = torch.cat((-current_quaternion[..., :3], current_quaternion[..., 3:4]), dim=-1)
        relative_position = _rotate_vector(inverse_current, target[..., :3] - current[..., :3])
        relative_quaternion = _quaternion_multiply(inverse_current, target_quaternion)
        converted[..., action_start : action_start + 3] = relative_position
        converted[..., action_start + 3 : action_start + 6] = _quaternion_to_rotvec(relative_quaternion)
    return converted.to(output_dtype)


def _absolute_actions(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    output_dtype = action.dtype
    state, action = state.float(), action.float()
    converted = action.clone()
    for action_start, state_start in ((0, 18), (6, 24)):
        current = state[..., state_start : state_start + 6]
        relative = action[..., action_start : action_start + 6]
        while current.ndim < relative.ndim:
            current = current.unsqueeze(-2)
        current_quaternion = _rotvec_to_quaternion(current[..., 3:6])
        relative_quaternion = _rotvec_to_quaternion(relative[..., 3:6])
        absolute_position = current[..., :3] + _rotate_vector(current_quaternion, relative[..., :3])
        absolute_quaternion = _quaternion_multiply(current_quaternion, relative_quaternion)
        converted[..., action_start : action_start + 3] = absolute_position
        converted[..., action_start + 3 : action_start + 6] = _quaternion_to_rotvec(absolute_quaternion)
    return converted.to(output_dtype)


@ProcessorStepRegistry.register(name="seer_state_action_compaction_processor")
@dataclass
class SeerStateActionCompactionProcessorStep(ProcessorStep):
    """Convert raw inference inputs to the compact state/action representation."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        observation = dict(result.get(TransitionKey.OBSERVATION) or {})
        state = observation.get(OBS_STATE)
        if state is not None:
            observation[OBS_STATE] = _compact_state(state)
            result[TransitionKey.OBSERVATION] = observation
        action = result.get(TransitionKey.ACTION)
        if action is not None:
            result[TransitionKey.ACTION] = _compact_action(action)
        return result

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="seer_eef_relative_actions_processor")
@dataclass
class SeerEefRelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """Apply GR00T-style SE(3) relative targets and retain the inference anchor."""

    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        observation = result.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        action = result.get(TransitionKey.ACTION)
        if state is None:
            raise ValueError("observation.state is required for Seer EEF action conversion")
        if state.shape[-1] == 64:
            compacted = dict(observation)
            compacted[OBS_STATE] = _compact_state(state)
            result[TransitionKey.OBSERVATION] = compacted
            state = compacted[OBS_STATE]
            if action is not None:
                action = _compact_action(action)
                result[TransitionKey.ACTION] = action
        if state.shape[-1] != 30:
            raise ValueError(f"Expected compact 30D Seer state, got {tuple(state.shape)}")
        # Keep the observation that anchored the current prediction until PI05 has
        # served every action in that chunk. Postprocessing uses the same state.
        if not self._chunk_in_flight():
            self._last_state = state
        if self.enabled and action is not None:
            if action.shape[-1] != 14:
                raise ValueError(f"Expected compact 14D Seer action, got {tuple(action.shape)}")
            result[TransitionKey.ACTION] = _relative_actions(state, action)
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


@ProcessorStepRegistry.register(name="seer_eef_absolute_actions_processor")
@dataclass
class SeerEefAbsoluteActionsProcessorStep(AbsoluteActionsProcessorStep):
    """Invert the relative transform after PI05 action unnormalization."""

    relative_step: SeerEefRelativeActionsProcessorStep | None = None
    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled or self.relative_step is None:
            return transition
        state = self.relative_step.get_cached_state()
        action = transition.get(TransitionKey.ACTION)
        if state is None or action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"Expected a tensor action, got {type(action)}")
        state = state.to(device=action.device, dtype=action.dtype)
        result = transition.copy()
        result[TransitionKey.ACTION] = _absolute_actions(state, action)
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


def make_seer_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    steps = make_default_policy_processor_steps(config, dataset_stats)
    relative_step = SeerEefRelativeActionsProcessorStep(enabled=True)
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        SeerStateActionCompactionProcessorStep(),
        relative_step,
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        SeerEefAbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
