# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
