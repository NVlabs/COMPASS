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
