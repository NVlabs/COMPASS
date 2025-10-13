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

from isaaclab.envs.mdp import commands
from isaaclab.utils import configclass
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

from mobility_es.mdp.command.uniform_collision_free_pose_command import UniformCollisionFreePoseCommand


@configclass
class UniformCollisionFreePose2dCommandCfg(commands.UniformPose2dCommandCfg):
    '''Config for UniformCollisionFreePose2dCommand'''
    class_type: type = UniformCollisionFreePoseCommand

    # Maximum number of trials for the reject sampling.
    max_resample_trial: int = 100

    # Minimum distance to asset for command sampling.
    minimum_distance: float = None

    # Probability of applying minimum distance sampling.
    minimum_distance_prob: float = None

    goal_pose_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal")

    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.2)
