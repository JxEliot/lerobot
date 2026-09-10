#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
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
from lerobot.processor.relative_action_processor import AbsoluteActionsProcessorStep as BaseAbsoluteActionsProcessorStep
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep as BaseRelativeActionsProcessorStep
from lerobot.lerobot_types import TransitionKey
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STATE

from .configuration_pi05 import PI05Config


def _hat(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def _rotvec_to_matrix(v: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    theta2 = theta.square()
    k = _hat(v)
    a = torch.where(theta2 < 1e-8, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / theta.clamp_min(1e-8))
    b = torch.where(theta2 < 1e-8, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(k)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def _matrix_to_rotvec(m: torch.Tensor) -> torch.Tensor:
    skew = torch.stack((m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), -1)
    sin_theta = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cos_theta = ((m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    theta = torch.atan2(sin_theta, cos_theta)
    scale = torch.where(sin_theta < 1e-6, 0.5 + theta.square() / 12, theta / (2 * sin_theta).clamp_min(1e-6))
    return scale.unsqueeze(-1) * skew


def _relative_eef_pose(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    current_r = _rotvec_to_matrix(current[..., 3:6])
    target_r = _rotvec_to_matrix(target[..., 3:6])
    rel_t = (current_r.transpose(-1, -2) @ (target[..., :3] - current[..., :3]).unsqueeze(-1)).squeeze(-1)
    rel_r = current_r.transpose(-1, -2) @ target_r
    return torch.cat((rel_t, _matrix_to_rotvec(rel_r)), dim=-1)


def _absolute_eef_pose(relative: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    current_r = _rotvec_to_matrix(current[..., 3:6])
    rel_r = _rotvec_to_matrix(relative[..., 3:6])
    abs_t = current[..., :3] + (current_r @ relative[..., :3].unsqueeze(-1)).squeeze(-1)
    abs_r = current_r @ rel_r
    return torch.cat((abs_t, _matrix_to_rotvec(abs_r)), dim=-1)


@ProcessorStepRegistry.register("gr00t_eef_relative_actions_processor")
@dataclass
class Gr00tEefRelativeActionsProcessorStep(BaseRelativeActionsProcessorStep):
    """GR00T-compatible dual-arm SE(3) relative action conversion."""

    enabled: bool = True
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        action = transition.get(TransitionKey.ACTION)
        if state is None:
            raise ValueError("observation.state is required for GR00T EEF conversion")
        self._last_state = state
        if not self.enabled or action is None:
            return transition
        converted = action.clone()
        for a0, s0 in ((0, 18), (6, 24)):
            current = state[..., s0 : s0 + 6]
            if action.ndim == 3:
                current = current.unsqueeze(-2)
            converted[..., a0 : a0 + 6] = _relative_eef_pose(action[..., a0 : a0 + 6], current)
        result = transition.copy()
        result[TransitionKey.ACTION] = converted
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


@ProcessorStepRegistry.register("gr00t_eef_absolute_actions_processor")
@dataclass
class Gr00tEefAbsoluteActionsProcessorStep(BaseAbsoluteActionsProcessorStep):
    """Inverse of :class:`Gr00tEefRelativeActionsProcessorStep`."""

    relative_step: Gr00tEefRelativeActionsProcessorStep | None = field(default=None, repr=False)
    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled or self.relative_step is None:
            return transition
        state = self.relative_step._last_state
        action = transition.get(TransitionKey.ACTION)
        if state is None or action is None:
            return transition
        converted = action.clone()
        for a0, s0 in ((0, 18), (6, 24)):
            current = state[..., s0 : s0 + 6]
            if action.ndim == 3:
                current = current.unsqueeze(-2)
            converted[..., a0 : a0 + 6] = _absolute_eef_pose(action[..., a0 : a0 + 6], current)
        result = transition.copy()
        result[TransitionKey.ACTION] = converted
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    # MEM section III-D represents proprioception with a linear projection into the
    # backbone instead of discretized prompt tokens, so the state is carried once.
    # Set from `PI05Config.use_proprioceptive_memory`; stock PI0.5 keeps it in the prompt.
    include_state_in_prompt: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        discretized_states = None
        if self.include_state_in_prompt:
            # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
            # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
            prompt_state = state[:, -1] if state.ndim == 3 else state
            state_np = prompt_state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            if discretized_states is None:
                full_prompt = f"Task: {cleaned_text};\nAction: "
            else:
                state_str = " ".join(map(str, discretized_states[i]))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = (
        Gr00tEefRelativeActionsProcessorStep(enabled=True)
        if config.use_gr00t_eef_relative_actions
        else RelativeActionsProcessorStep(enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        )
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
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
        (Gr00tEefAbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
         if config.use_gr00t_eef_relative_actions
         else AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step)),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
