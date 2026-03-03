# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
                                            max_resample_trial=100,
                                            collision_distance=0.75):
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Check if precomputed start poses are available
    if env.collision_checker.is_initialized() and env.collision_checker.has_precomputed_start_poses(
    ):
        # Use precomputed valid locations (these are in world coordinates)
        num_samples = len(env_ids)

        # Try to use orientation-aware sampling if available
        if hasattr(env.collision_checker, 'sample_start_pose_with_orientation'):
            try:
                sampled_positions_2d, sampled_yaws = (
                    env.collision_checker.sample_start_pose_with_orientation(
                        num_samples, max_iteration=max_resample_trial))
                use_precomputed_orientations = True
            except (ValueError, AttributeError):
                # Fall back to position-only sampling (will use random orientation)
                sampled_positions_2d = env.collision_checker.sample_start_pose(num_samples)
                use_precomputed_orientations = False
        else:
            sampled_positions_2d = env.collision_checker.sample_start_pose(num_samples)
            use_precomputed_orientations = False

        # Convert to torch tensor
        sampled_positions_2d_torch = torch.tensor(sampled_positions_2d,
                                                  device=asset.device,
                                                  dtype=torch.float32)

        # Get default root states
        root_states = asset.data.default_root_state[env_ids].clone()
        positions = root_states[:, 0:3].clone()
        orientations = root_states[:, 3:7].clone()
        velocities = root_states[:, 7:13].clone()

        # Precomputed positions are in world coordinates from occupancy map origin
        # The map origin is typically at (0,0) in world space
        # Each environment has its own origin (env_origins), so we need to use precomputed
        # positions as relative offsets from the map origin,
        # then apply them relative to each environment's origin
        # The original sampling does: root_states + env_origins + random_offset
        # Precomputed positions are already relative to map origin (0,0),
        # so use them directly as offsets
        origins = env.scene.env_origins[env_ids, :2]
        # Precomputed positions are relative to map origin, use them as offsets (same as pose_range)
        # Apply same pattern as sample_root_state_uniform: root_states + env_origins + offset
        positions[:, 0] = root_states[:, 0] + origins[:, 0] + sampled_positions_2d_torch[:, 0]
        positions[:, 1] = root_states[:, 1] + origins[:, 1] + sampled_positions_2d_torch[:, 1]

        # Use precomputed orientations if available, otherwise sample from pose_range
        if use_precomputed_orientations:
            sampled_yaws_torch = torch.tensor(sampled_yaws,
                                              device=asset.device,
                                              dtype=torch.float32)
            orientations = math_utils.quat_from_euler_xyz(torch.zeros_like(sampled_yaws_torch),
                                                          torch.zeros_like(sampled_yaws_torch),
                                                          sampled_yaws_torch)
        elif "yaw" in pose_range:
            yaw_range = pose_range["yaw"]
            yaw_samples = torch.empty(len(env_ids), device=asset.device)
            yaw_samples.uniform_(*yaw_range)
            orientations = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw_samples),
                                                          torch.zeros_like(yaw_samples),
                                                          yaw_samples)

        # Set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        return

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
            in_collisions = env.collision_checker.is_in_collision(positions_2d,
                                                                  distance=collision_distance)
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
