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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from mobility_es.mdp.utils import goal_pose_in_robot_frame
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


def goal_reached(
    env: RLESEnvWrapper,
    dist_threshold: float = 0.5,
    linear_vel_threshold: float = 0.1,
    angular_vel_threshold: float = 0.1,
    heading_threshold: float = 0.1,
    command_name: str = 'goal_pose',
    offset: tuple[float, float] = (-0.0, 0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]

    # Check if the robot velocity is below the threshold
    root_linear_vel = torch.norm(robot.data.root_lin_vel_w, dim=1)
    root_angular_vel = robot.data.root_ang_vel_w[:, 2].abs()
    vel_threshold = torch.logical_and(root_linear_vel < linear_vel_threshold,
                                      root_angular_vel < angular_vel_threshold)

    # Check if the robot is close to the goal and has low velocity
    goal_pose = goal_pose_in_robot_frame(env, command_name, offset, robot_cfg)
    goal_pose_abs = torch.norm(goal_pose, p=2, dim=1)
    pose_vel_threshold = torch.logical_and(goal_pose_abs < dist_threshold, vel_threshold)

    # Check if the robot's heading is close to the goal heading and has low velocity
    goal_heading = env.command_manager.get_command(command_name)[:, 3].abs()
    reached_goal = torch.logical_and(goal_heading < heading_threshold, pose_vel_threshold)

    return reached_goal


def nan_pose(env: RLESEnvWrapper, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Check if the robot pose is nan, which can happen when the robot explodes in simulation.
    nan_pos = torch.isnan(env.scene[robot_cfg.name].data.root_pos_w)
    nan_quat = torch.isnan(env.scene[robot_cfg.name].data.root_quat_w)
    return torch.logical_or(torch.any(nan_pos, dim=1), torch.any(nan_quat, dim=1))
