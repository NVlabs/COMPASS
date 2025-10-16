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

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg

from mobility_es.mdp.action.action_visualization import ActionVisualizer
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


class NonHolonomicPerfectControlAction(ActionTerm):
    """Action item for non-holonomic robot with perfect controller."""

    cfg: ActionTermCfg
    """The articulation asset on which the action term is applied."""
    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env: RLESEnvWrapper):
        # Initialize the action term
        super().__init__(cfg, env)

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

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
        self._processed_actions = self.raw_actions[:, (0, 5)]

    def apply_actions(self):
        # Get current asset state
        root_state = self._asset.data.root_state_w

        # Extract linear_x and angular_z from actions
        linear_x = self.processed_actions[:, 0]
        angular_z = self.processed_actions[:, 1]

        # Get current position and orientation quaternion
        pos = root_state[:, :3]
        quat = root_state[:, 3:7]

        # Convert quaternion to yaw angle
        yaw = 2 * torch.atan2(quat[:, 3], quat[:, 0])

        # Update yaw with angular velocity
        dt = self._env.physics_dt
        new_yaw = yaw + angular_z * dt

        # Compute new position based on forward velocity and orientation
        pos[:, 0] += linear_x * torch.cos(new_yaw) * dt
        pos[:, 1] += linear_x * torch.sin(new_yaw) * dt

        # Convert new yaw to quaternion (rotation around z-axis only)
        new_quat = torch.zeros_like(quat)
        new_quat[:, 0] = torch.cos(new_yaw / 2)
        new_quat[:, 3] = torch.sin(new_yaw / 2)

        # Update root state
        root_state[:, :3] = pos
        root_state[:, 3:7] = new_quat
        root_state[:, 7:10] = torch.stack([
            linear_x * torch.cos(new_yaw), linear_x * torch.sin(new_yaw),
            torch.zeros_like(linear_x)
        ],
                                          dim=1)
        root_state[:, 10:13] = torch.stack(
            [torch.zeros_like(angular_z),
             torch.zeros_like(angular_z), angular_z], dim=1)
        self._asset.write_root_state_to_sim(root_state)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def visualize(self, base_action, residual_action):
        self._visualizer.visualize(base_action, residual_action)
