# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from mobility_es.mdp.action.action_visualization import ActionVisualizer
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


@configclass
class LocomotionPolicyActionCfg(mdp.JointPositionActionCfg):
    """Configuration for the action term with locomotion policies.

    See :class:`LocomotionPolicyActionCfg` for more details.
    """

    class_type: type[ActionTerm] = MISSING
    policy_ckpt_path: str = MISSING


class LocomotionPolicyAction(mdp.JointPositionAction):
    """Action item for locomotion policies."""

    cfg: LocomotionPolicyActionCfg
    """The articulation asset on which the action term is applied."""
    _asset: Articulation

    def __init__(self, cfg: LocomotionPolicyActionCfg, env: RLESEnvWrapper):
        super().__init__(cfg, env)

        # Load the env and locomotion policy.
        self.env = env
        self._locomotion_policy = torch.jit.load(self.cfg.policy_ckpt_path).to(self.device)

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._last_joint_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Visualization
        self._visualizer = ActionVisualizer(cfg, env)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions

    def apply_actions(self):
        # Get locomotion observation.
        obs_dict = self.env.observation_manager.compute_group('locomotion')

        base_lin_vel = obs_dict['base_lin_vel']
        base_ang_vel = obs_dict['base_ang_vel']
        projected_gravity = obs_dict['projected_gravity']
        velocity_commands = self._processed_actions[:, [0, 1, 5]]
        joint_pos = obs_dict['joint_pos']
        joint_vel = obs_dict['joint_vel']
        last_joint_actions = self._last_joint_actions
        # Concatenate to compose the input obs tensor. (Order preserved)
        obs = torch.cat([
            base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel,
            last_joint_actions
        ],
                        dim=1)
        joint_actions = self._locomotion_policy(obs)
        self._last_joint_actions = joint_actions

        # Process and set the joint pose targets
        joint_actions = joint_actions * self._scale + self._offset
        self._asset.set_joint_position_target(joint_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._last_joint_actions[env_ids] = 0.0

    def visualize(self, base_action, residual_action):
        self._visualizer.visualize(base_action, residual_action)
