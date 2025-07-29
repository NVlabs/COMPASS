# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import random

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.envs.mdp import commands

from mobility_es.wrapper.env_wrapper import RLESEnvWrapper

if TYPE_CHECKING:
    from mobility_es.mdp.command.commands_cfg import UniformCollisionFreePose2dCommandCfg


class UniformCollisionFreePoseCommand(commands.pose_2d_command.UniformPose2dCommand):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.

    It will then run collision checker to make sure the sampled pose is not in collision. If it
    is in collision, the sampled pose will be discarded and a new pose will be sampled until
    either a collision pose is sampled or reaches maximum number of sampling.

    Note: the occupancy map based collision checker only works for static environments and
    doesn't support randomization.
    """

    cfg: UniformCollisionFreePose2dCommandCfg
    """Configuration for the command term."""

    # pylint: disable=useless-parent-delegation
    def __init__(self, cfg: UniformCollisionFreePose2dCommandCfg, env: RLESEnvWrapper):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # Create the occupancy map for collision check.
        self.collision_checker = env.collision_checker

    def __str__(self) -> str:
        msg = "UniformCollisionFreePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        resample_env_ids = env_ids
        num_resample_trials = 0
        while num_resample_trials < self.cfg.max_resample_trial:
            # Obtain origins for the robots
            self.pos_command_w[resample_env_ids] = self.robot.data.root_pos_w[resample_env_ids]
            # Offset the position command by the current root position
            r = torch.empty(len(resample_env_ids), device=self.device)
            # Randomize the position
            self.pos_command_w[resample_env_ids,
                               0] += r.uniform_(*self._get_sampling_range(self.cfg.ranges.pos_x))
            self.pos_command_w[resample_env_ids,
                               1] += r.uniform_(*self._get_sampling_range(self.cfg.ranges.pos_y))
            self.pos_command_w[resample_env_ids, 2] = 0.0
            # Randomize the heading
            self.heading_command_w[resample_env_ids] = r.uniform_(*self.cfg.ranges.heading)

            resample_env_ids = self._check_collision(resample_env_ids)
            num_resample_trials += 1
            if len(resample_env_ids) < 1:
                # Finished resampling for all the envs
                break

        # Print a warning message.
        if num_resample_trials >= self.cfg.max_resample_trial:
            print(f"Envs with id: {resample_env_ids} failed to sample collision free"
                  f"poses within the maxum iteration:  {self.cfg.max_resample_trial}")

    # Apply minimum distance constraint to the input range, and randomly choose a feasible one.
    def _get_sampling_range(self, input_range: tuple[float, float]) -> tuple[float, float]:
        if not self.cfg.minimum_distance or random.random() > self.cfg.minimum_distance_prob:
            return input_range
        low, high = input_range
        if low > -self.cfg.minimum_distance and high < self.cfg.minimum_distance:
            raise ValueError('Sampling range is too small for minimum distance.')

        sampling_ranges = []
        if low < -self.cfg.minimum_distance:
            sampling_ranges.append((low, -self.cfg.minimum_distance))
        if high > self.cfg.minimum_distance:
            sampling_ranges.append((self.cfg.minimum_distance, high))
        return random.choice(sampling_ranges)

    # pylint: disable=unused-argument
    def _check_collision(self, env_ids: Sequence[int]):
        """ Checks whether the resampled command is collision free for the environments in env_ids.
        Returns a list of env_ids that are in collision.
        """
        if not self.collision_checker.is_initialized():
            return []

        sampled_points_r = self.pos_command_w[env_ids, :2]
        # Get the points relative to the environment origin if the env prime is not global.
        if self._env.scene.cfg.environment.prim_path.startswith('/World/envs/'):
            sampled_points_r = sampled_points_r - self._env.scene.env_origins[env_ids, :2]
        in_collisions = self.collision_checker.is_in_collision(sampled_points_r, distance=0.5)
        return env_ids[torch.where(in_collisions == 1)[0]]
