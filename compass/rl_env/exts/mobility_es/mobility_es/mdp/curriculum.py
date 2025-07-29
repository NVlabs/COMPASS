# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from collections.abc import Sequence

from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


# Increase the prob of applying min command distance constraint based the training iteration.
def increase_command_minimum_distance_prob(
        env: RLESEnvWrapper,
        env_ids: Sequence[int],    # pylint: disable=unused-argument
        command_name: str = 'goal_pose',
        minimum_distance_prob_range=(0.1, 1.0),
        total_iterations=1000,
        num_steps_per_iteration=256):
    cur_iteration = env.common_step_counter // num_steps_per_iteration
    start_prob, end_prob = minimum_distance_prob_range
    prob_range = end_prob - start_prob
    minimum_distance_prob = start_prob + prob_range * (cur_iteration / total_iterations)
    env.command_manager.get_term(command_name).cfg.minimum_distance_prob = minimum_distance_prob
