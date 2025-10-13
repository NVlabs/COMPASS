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
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from mobility_es.mdp.action.action_visualization import ActionVisualizer
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Configuration for the differential drive action term with left and right wheel joints.

    See :class:`DifferentialDriveAction` for more details.
    """

    class_type: type[ActionTerm] = MISSING
    # Name of the left wheel joint.
    left_joint_name: str = MISSING
    # Name of the right wheel joint.
    right_joint_name: str = MISSING
    # Differential drive wheel parameters.
    wheel_base: float = MISSING
    wheel_radius: float = MISSING


class DifferentialDriveAction(ActionTerm):
    """Action item for differential drive."""

    cfg: DifferentialDriveActionCfg
    """The articulation asset on which the action term is applied."""
    _asset: Articulation

    def __init__(self, cfg: DifferentialDriveActionCfg, env: RLESEnvWrapper):
        # Initialize the action term
        super().__init__(cfg, env)

        # Parse the joint information
        # -- Left joint
        left_joint_id, _ = self._asset.find_joints(self.cfg.left_joint_name)
        if len(left_joint_id) != 1:
            raise ValueError(
                f"Found more than one joint for the left joint name: {self.cfg.left_joint_name}")
        # -- Right joint
        right_joint_id, _ = self._asset.find_joints(self.cfg.right_joint_name)
        if len(right_joint_id) != 1:
            raise ValueError(
                f"Found more than one joint for the right joint name: {self.cfg.right_joint_name}")

        # Process into a list of joint ids
        self._joint_ids = [left_joint_id[0], right_joint_id[0]]

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel_command = torch.zeros(self.num_envs, 2, device=self.device)

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
        self._joint_vel_command[:, 0] = (
            self._processed_actions[:, 0] -
            self._processed_actions[:, 5] * self.cfg.wheel_base * 0.5) / self.cfg.wheel_radius
        self._joint_vel_command[:, 1] = (
            self._processed_actions[:, 0] +
            self._processed_actions[:, 5] * self.cfg.wheel_base * 0.5) / self.cfg.wheel_radius
        # Set the joint velocity targets
        self._asset.set_joint_velocity_target(self._joint_vel_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def visualize(self, base_action, residual_action):
        self._visualizer.visualize(base_action, residual_action)
