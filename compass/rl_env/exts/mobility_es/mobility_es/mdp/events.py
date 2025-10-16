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

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


def sample_root_state_uniform(env: RLESEnvWrapper,
                              env_ids: torch.Tensor,
                              pose_range: dict[str, tuple[float, float]],
                              velocity_range: dict[str, tuple[float, float]],
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # Sample poses
    pose_range_list = [
        pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_ranges = torch.tensor(pose_range_list, device=asset.device)

    # poses
    rand_samples = math_utils.sample_uniform(pose_ranges[:, 0],
                                             pose_ranges[:, 1], (len(env_ids), 6),
                                             device=asset.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4],
                                                        rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # Sample velocities
    velocity_range_list = [
        velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    velocity_ranges = torch.tensor(velocity_range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(velocity_ranges[:, 0],
                                             velocity_ranges[:, 1], (len(env_ids), 6),
                                             device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    return positions, orientations, velocities


def reset_root_state_uniform_collision_free(env: RLESEnvWrapper,
                                            env_ids: torch.Tensor,
                                            pose_range: dict[str, tuple[float, float]],
                                            velocity_range: dict[str, tuple[float, float]],
                                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                            max_resample_trial=10):
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Sample states
    positions, orientations, velocities = sample_root_state_uniform(env, env_ids, pose_range,
                                                                    velocity_range, asset_cfg)
    # Check collisions if collision checker is initialized.
    if env.collision_checker.is_initialized():
        # Index map with key as env_id and value as the index.
        env_ids_index_map = {int(val.item()): idx for idx, val in enumerate(env_ids)}
        resample_env_ids = env_ids
        resample_indices = torch.tensor(
            [env_ids_index_map[int(env_id.item())] for env_id in resample_env_ids])
        num_resample_trials = 0
        while num_resample_trials < max_resample_trial:
            positions_2d = positions[resample_indices, :2]
            # Get the points relative to the environment origin if the env prime is not global.
            if env.scene.cfg.environment.prim_path.startswith('/World/envs/'):
                positions_2d = positions_2d - env.scene.env_origins[resample_env_ids, :2]
            in_collisions = env.collision_checker.is_in_collision(positions_2d)
            resample_env_ids = resample_env_ids[torch.where(in_collisions == 1)[0]]
            if len(resample_env_ids) < 1:
                break
            # Resample new states for the collided root.
            resample_indices = torch.tensor(
                [env_ids_index_map[int(env_id.item())] for env_id in resample_env_ids])
            positions[resample_indices], orientations[resample_indices], velocities[
                resample_indices] = sample_root_state_uniform(env, resample_env_ids, pose_range,
                                                              velocity_range, asset_cfg)
            num_resample_trials += 1

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
