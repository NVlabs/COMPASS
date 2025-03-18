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


def root_speed(
        env: RLESEnvWrapper,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The speed of the root, which is the normalization of the linear velocity"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_lin_vel_w, dim=1).unsqueeze(1)


def root_yaw_rate(
        env: RLESEnvWrapper,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The yaw rate of the root"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2].unsqueeze(1)


def camera_img(env: RLESEnvWrapper, data_type: str = "rgb"):
    supported_data_types = ["rgb", "depth"]
    if data_type not in supported_data_types:
        raise ValueError(f"data type {data_type} is not in the support list {supported_data_types}")
    camera = env.scene["camera"]
    return camera.data.output[data_type].reshape((camera.data.output[data_type].shape[0], -1))


def upsample_segments(start_pose, end_pose, d_max=1.0, n_segments=10):
    num_env = start_pose.size(0)
    all_segments = torch.zeros((num_env, n_segments, 4)).to(start_pose.device)

    for i in range(num_env):
        s_x, s_y = start_pose[i]
        e_x, e_y = end_pose[i]
        distance = torch.sqrt((e_x - s_x)**2 + (e_y - s_y)**2)
        num_segments = torch.ceil(distance / d_max).int().item()

        # Generate interpolation steps for segments
        t = torch.linspace(0, 1, num_segments + 1).unsqueeze(1)
        points = (1 - t) * torch.tensor([s_x, s_y]) + t * torch.tensor([e_x, e_y])
        segments = torch.cat([points[:-1], points[1:]], dim=1)

        if num_segments < n_segments:
            # Pad with segments where the start and end points are the last point
            last_point = torch.tensor([e_x, e_y])
            last_segment = torch.cat([last_point, last_point], dim=0)
            padding_segments = last_segment.repeat(n_segments - num_segments, 1)
            segments = torch.cat([segments, padding_segments], dim=0)
        else:
            # Truncate to n_segments if necessary
            segments = segments[:n_segments]
        all_segments[i] = segments

    return all_segments


def routing_to_goal_simplified(
    env: RLESEnvWrapper,
    command_name: str = "goal_pose",
    offset: tuple[float, float] = (-0.0, 0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    goal_pose = goal_pose_in_robot_frame(env, command_name, offset, robot_cfg)
    start_pose = goal_pose.new_zeros(goal_pose.shape[0], 2)
    return upsample_segments(start_pose, goal_pose)


def foot_print(
    env: RLESEnvWrapper,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_length: float = 1.0,
    asset_width: float = 1.0,
) -> torch.Tensor:
    """Represent the foot print of the asset as a rectangle that is offset to the asset root."""
    robot: RigidObject = env.scene[asset_cfg.name]
    root_pos = robot.data.root_pos_w - env.scene.env_origins
    root_pos_2d = root_pos[:, :2]

    def create_footprint_tensor(length, width, device):
        tensor = torch.tensor([[length / 2, width / 2], [-length / 2, width / 2],
                               [-length / 2, -width / 2], [length / 2, -width / 2]],
                              device=device)
        return tensor

    foot_print_poly = create_footprint_tensor(asset_length, asset_width, root_pos.device)
    foot_print_poly = foot_print_poly.unsqueeze(0).repeat(root_pos.shape[0], 1, 1)

    foot_print_poly_offset = root_pos_2d.unsqueeze(1) + foot_print_poly
    return foot_print_poly_offset
