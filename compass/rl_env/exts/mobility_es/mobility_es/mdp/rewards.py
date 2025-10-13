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
from typing import Optional, List

from isaaclab.managers import SceneEntityCfg

from mobility_es.mdp.utils import goal_pose_in_robot_frame
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


def goal_reaching(
    env: RLESEnvWrapper,
    command_name: str = "goal_pose",
    offset: tuple[float, float] = (-0, 0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for moving towards the goal pose. """
    goal_pose = goal_pose_in_robot_frame(env, command_name, offset, robot_cfg)
    goal_pose_abs = -torch.norm(goal_pose, p=2, dim=1).reshape((goal_pose.shape[0], 1))

    return torch.flatten(goal_pose_abs)


def goal_heading_alignment(
    env: RLESEnvWrapper,
    command_name: str = "goal_pose",
    offset: tuple[float, float] = (-0.0, 0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 1.0    # Distance at which heading alignment becomes important
) -> torch.Tensor:
    """Reward the agent for aligning with the goal heading when reached destination.

    The heading alignment reward is scaled based on the distance to the goal.
    As the robot gets closer to the goal, the heading alignment becomes more important.
    """
    # Get the goal pose in robot frame to calculate distance
    goal_pose = goal_pose_in_robot_frame(env, command_name, offset, robot_cfg)
    distance_to_goal = torch.norm(goal_pose[:, :2], p=2, dim=1)

    # Get the goal heading
    goal_heading = env.command_manager.get_command(command_name)[:, 3]
    heading_error = goal_heading.abs()

    # Scale the heading alignment based on distance
    # As distance approaches 0, the weight approaches 1
    # When distance > threshold, the weight is small but non-zero
    distance_weight = torch.exp(-distance_to_goal / distance_threshold)

    # Apply the distance-based scaling to the heading error
    scaled_heading_error = heading_error * distance_weight

    return -scaled_heading_error


def masked_action_rate_l2(env: RLESEnvWrapper,
                          idx_masks: Optional[List[int]] = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using an L2 squared kernel."""

    # Extract current and previous actions
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Calculate the difference in actions
    if idx_masks is not None:
        current_action = current_action[:, idx_masks]
        prev_action = prev_action[:, idx_masks]

    action_rate = current_action - prev_action

    # Return the sum of squared differences for the L2 penalty
    return torch.sum(action_rate.pow(2), dim=1)


def masked_action_l2(env: RLESEnvWrapper, idx_masks: Optional[List[int]] = None) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    current_action = env.action_manager.action

    # Calculate the difference in actions
    if idx_masks is not None:
        current_action = current_action[:, idx_masks]

    return torch.sum(current_action.pow(2), dim=1)
