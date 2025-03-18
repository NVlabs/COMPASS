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

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

from mobility_es.mdp.utils import goal_pose_in_robot_frame
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


def goal_reached(
    env: RLESEnvWrapper,
    dist_threshold: float = 0.5,
    vel_threshold: float = 0.1,
    command_name: str = 'goal_pose',
    offset: tuple[float, float] = (-0.0, 0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]

    # Goal pose
    goal_pose = goal_pose_in_robot_frame(env, command_name, offset, robot_cfg)

    # Distance to goal
    goal_pose_abs = torch.norm(goal_pose, p=2, dim=1)
    # Robot velocity
    root_vel = torch.norm(robot.data.root_lin_vel_w, dim=1)

    return torch.logical_and(goal_pose_abs < dist_threshold, root_vel < vel_threshold)


def nan_pose(env: RLESEnvWrapper, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Check if the robot pose is nan, which can happen when the robot explodes in simulation.
    nan_pos = torch.isnan(env.scene[robot_cfg.name].data.root_pos_w)
    nan_quat = torch.isnan(env.scene[robot_cfg.name].data.root_quat_w)
    return torch.logical_or(torch.any(nan_pos, dim=1), torch.any(nan_quat, dim=1))
