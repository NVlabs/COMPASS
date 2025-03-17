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
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def goal_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    offset: tuple[float, float] = (-0, 0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Get the goal pose in the robot frame with offset. """

    # Get the goal pose relative to the root(i.e., robot) frame.
    goal_pose = env.command_manager.get_command(command_name)[:, :2]

    # Compute offset.
    robot = env.scene[robot_cfg.name]
    robot_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2].reshape(
        (goal_pose.shape[0], 1))
    offset_vec = torch.cat([
        offset[0] * torch.cos(robot_yaw) - offset[1] * torch.sin(robot_yaw),
        offset[0] * torch.sin(robot_yaw) + offset[1] * torch.cos(robot_yaw)
    ],
                           dim=1)
    # Goal pose in robot frame with offset.
    goal_pose_r = goal_pose - offset_vec
    return goal_pose_r
