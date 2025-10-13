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

import os

from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from mobility_es.config import scene_assets
from mobility_es.config import robots
from mobility_es.config.env_cfg import GoalReachingEnvCfg
from mobility_es.mdp.action.locomotion_policy_action import LocomotionPolicyAction, LocomotionPolicyActionCfg


@configclass
class H1ActionsCfg:
    """Action specifications for the MDP."""

    drive_joints = LocomotionPolicyActionCfg(class_type=LocomotionPolicyAction,
                                             asset_name="robot",
                                             policy_ckpt_path=os.path.join(
                                                 os.path.dirname(__file__),
                                                 "../ckpt/h1_locomotion_policy.pt"),
                                             joint_names=[".*"],
                                             scale=0.5,
                                             use_default_offset=True)


@configclass
class H1GoalReachingEnvCfg(GoalReachingEnvCfg):
    """Configuration for the H1 to reach a 2D target pose environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = robots.h1.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera = scene_assets.camera.replace(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/front_cam")
        self.scene.camera.offset = CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.65),
                                                       rot=(-0.2705, 0.6532, -0.6532, 0.2705981),
                                                       convention="ros")

        self.actions = H1ActionsCfg()
